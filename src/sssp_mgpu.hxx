// sssp_mgpu.hxx

// Notes:
// - Checking for convergence after iteration slows us down.  Could do better by checking every
//   N iterations.  Or possibly there's a better way to check if the output frontier is empty.

#pragma once
#pragma GCC diagnostic ignored "-Wunused-result"

#include "nccl.h"
#include "thrust/device_vector.h"

#include "helpers.hxx"

template <typename type_t>
void scatter(type_t** out, type_t* h_x, int n, int n_gpus) {
    #pragma omp parallel for num_threads(n_gpus)
    for(int gid = 0; gid < n_gpus; gid++) {
        cudaSetDevice(gid);
        
        type_t* d_x;
        cudaMalloc(&d_x, n * sizeof(type_t));
        cudaMemcpy(d_x, h_x, n * sizeof(type_t),  cudaMemcpyHostToDevice);
        
        out[gid] = d_x;
        cudaDeviceSynchronize();
    }
    cudaSetDevice(0);
}

template <typename Int, typename Real>
long long sssp_mgpu(Real* h_dist, Int n_seeds, Int* seeds, Int n_nodes, Int n_edges, Int* rindices, Int* cindices, Real* data, Int n_gpus) {
    std::cout << "sssp_mgpu" << std::endl;
    
    // --
    // Setup devices
    
    cudaSetDevice(0);
    cudaStream_t master_stream;
    cudaStreamCreateWithFlags(&master_stream, cudaStreamNonBlocking);

    struct gpu_info {
        cudaStream_t stream;
        cudaEvent_t  event;
    };
    
    std::vector<gpu_info> infos;
    
    for(int i = 0 ; i < n_gpus ; i++) {
        gpu_info info;
        cudaSetDevice(i);
        cudaStreamCreateWithFlags(&info.stream, cudaStreamNonBlocking);
        cudaEventCreate(&info.event);
        infos.push_back(info);
    }
    
    // Enable peer access
    for(int i = 0; i < n_gpus; i++) {
        cudaSetDevice(i);
        for(int j = 0; j < n_gpus; j++) {
            if(i == j) continue;
            cudaDeviceEnablePeerAccess(j, 0);
        }
    }
    
    cudaSetDevice(0);

    // Setup NCCL
    ncclComm_t comms[n_gpus];
    int devs[n_gpus];
    for(int i = 0; i < n_gpus; i++) devs[i] = i;
    ncclCommInitAll(comms, n_gpus, devs);

    // --
    // Setup chunks
        
    Int* starts    = (Int*)malloc(n_gpus * sizeof(Int));
    Int* ends      = (Int*)malloc(n_gpus * sizeof(Int));
    Int chunk_size = (n_edges + n_gpus - 1) / n_gpus;
    for(Int i = 0; i < n_gpus; i++) {
        starts[i] = i * chunk_size;
        ends[i]   = (i + 1) * chunk_size;
    }
    ends[n_gpus - 1] = n_edges;

    // --
    // Setup frontiers
    
    char* h_frontier_in  = (char*)malloc(n_nodes * sizeof(char));
    char* h_frontier_out = (char*)malloc(n_nodes * sizeof(char));
    char* h_keep_going   = (char*)malloc(1       * sizeof(char));
    
    for(Int i = 0; i < n_nodes; i++) h_dist[i]          = std::numeric_limits<Real>::max();
    for(Int i = 0; i < n_nodes; i++) h_frontier_in[i]   = -1;
    for(Int i = 0; i < n_nodes; i++) h_frontier_out[i]  = -1;
    
    for(Int seed = 0; seed < n_seeds; seed++) {
        h_dist[seed]         = 0;
        h_frontier_in[seed]  = 0;
    }
    
    h_keep_going[0] = true;
        
    // Local data, frontier + dist
    Int* all_cindices[n_gpus];
    Int* all_rindices[n_gpus];
    Real* all_data[n_gpus];
    char* all_frontier_in[n_gpus];
    char* all_frontier_out[n_gpus];
    char* all_keep_going[n_gpus];
    Real* all_dist[n_gpus];

    scatter(all_cindices,     cindices,       n_edges, n_gpus);
    scatter(all_rindices,     rindices,       n_edges, n_gpus);
    scatter(all_data,         data,           n_edges, n_gpus);
    scatter(all_frontier_in,  h_frontier_in,  n_nodes, n_gpus);
    scatter(all_frontier_out, h_frontier_out, n_nodes, n_gpus);
    scatter(all_keep_going,   h_keep_going,   1,       n_gpus);
    scatter(all_dist,         h_dist,         n_nodes, n_gpus);

    cuda_timer_t timer;
    timer.start();
    
    char iter       = 0;
    char keep_going = 0;
    while(true) {
        char next_iter = iter + 1;
                
        // Advance
        #pragma omp parallel for num_threads(n_gpus)
        for(int gid = 0; gid < n_gpus; gid++) {
            
            cudaSetDevice(gid);
            
            Int* l_cindices      = all_cindices[gid];
            Int* l_rindices      = all_rindices[gid];
            Real* l_data         = all_data[gid];
            char* l_frontier_in  = all_frontier_in[gid];
            char* l_frontier_out = all_frontier_out[gid];
            char* l_keep_going   = all_keep_going[gid];
            Real* l_dist         = all_dist[gid];

            // Advance
            auto edge_op = [=] __device__(int const& offset) -> void {
                Int src = l_rindices[offset];
                Int dst = l_cindices[offset];
                
                if(l_frontier_in[src] != iter) return;
                l_keep_going[0] = next_iter;
                
                Real new_dist = l_dist[src] + l_data[offset];     
                Real old_dist = atomicMin(l_dist + dst, new_dist);
                if(new_dist < old_dist)
                    l_frontier_out[dst] = next_iter;
            };
            
            thrust::for_each(
                thrust::cuda::par.on(infos[gid].stream),
                thrust::make_counting_iterator<Int>(starts[gid]),
                thrust::make_counting_iterator<Int>(ends[gid]),
                edge_op
            );
            
            cudaEventRecord(infos[gid].event, infos[gid].stream);
        }
        
        for(int gid = 0; gid < n_gpus; gid++)
            cudaStreamWaitEvent(master_stream, infos[gid].event, 0);
        cudaStreamSynchronize(master_stream);
        
        // Merge
        ncclGroupStart();
        for (int i = 0; i < n_gpus; i++) {
            ncclAllReduce((const void*)all_dist[i],         (void*)all_dist[i],        n_nodes, ncclFloat, ncclMin, comms[i], infos[i].stream);    // min-reduce distance
            ncclAllReduce((const void*)all_frontier_out[i], (void*)all_frontier_in[i], n_nodes, ncclChar,  ncclMax,  comms[i], infos[i].stream);   // swap frontiers
            ncclReduce((const void*)all_keep_going[i],      (void*)all_keep_going[i],  n_gpus, ncclChar,  ncclMax, 0, comms[i], infos[i].stream); // check convergence criteria
        }
        ncclGroupEnd();

        for(int gid = 0; gid < n_gpus; gid++)
            cudaStreamWaitEvent(master_stream, infos[gid].event, 0);
        cudaStreamSynchronize(master_stream);
        
        cudaSetDevice(0);
        cudaMemcpy(&keep_going, all_keep_going[0], 1 * sizeof(char), cudaMemcpyDeviceToHost);
        if(keep_going != next_iter) break;

        iter++;
    }
    
    // Merge
    ncclGroupStart();
    for (int i = 0; i < n_gpus; i++)
        ncclReduce((const void*)all_dist[i], (void*)all_dist[i], n_nodes, ncclFloat, ncclMin, 0, comms[i], infos[i].stream);
    ncclGroupEnd();

    for(int gid = 0; gid < n_gpus; gid++)
        cudaStreamWaitEvent(master_stream, infos[gid].event, 0);
    cudaStreamSynchronize(master_stream);
    
    cudaSetDevice(0);
    cudaMemcpy(h_dist, all_dist[0], n_nodes * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    return timer.stop();
}