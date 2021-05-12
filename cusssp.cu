#pragma GCC diagnostic ignored "-Wunused-result"

#include <cstdlib>  // EXIT_SUCCESS
#include "omp.h"
#include "nvToolsExt.h"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"

#include <thrust/transform_scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <string.h>
#include <omp.h>

#include <queue>
#include <vector>

using namespace std;
using namespace std::chrono;

// #define VERBOSE
// #define NODE_BALANCED

// --
// Global defs

typedef int Int;
typedef float Real;

// graph
Int n_nodes;
Int n_edges;
Int* indptr;
Int* rindices;
Int* cindices;
Real* data;

__device__ static float atomicMin(float* address, float value) {
  int* addr_as_int = reinterpret_cast<int*>(address);
  int old = *addr_as_int;
  int expected;
  do {
    expected = old;
    old = atomicCAS(addr_as_int, expected, __float_as_int(::fminf(value, __int_as_float(expected))));
  } while (expected != old);
  return __int_as_float(old);
}

// --
// IO

void loal_data(std::string inpath) {
    FILE *ptr;
    ptr = fopen(inpath.c_str(), "rb");

    fread(&n_nodes,   sizeof(Int), 1, ptr);
    fread(&n_nodes,   sizeof(Int), 1, ptr);
    fread(&n_edges,    sizeof(Int), 1, ptr);

    indptr   = (Int*)  malloc(sizeof(Int)  * (n_nodes + 1)  );
    cindices = (Int*)  malloc(sizeof(Int)  * n_edges         );
    rindices = (Int*)  malloc(sizeof(Int)  * n_edges         );
    data     = (Real*) malloc(sizeof(Real) * n_edges         );

    fread(indptr,  sizeof(Int),   n_nodes + 1 , ptr);  // send directy to the memory since thats what the thing is.
    fread(cindices, sizeof(Int),  n_edges      , ptr);
    fread(data,    sizeof(Real),  n_edges      , ptr);
    
    for(Int src = 0; src < n_nodes; src++) {
        for(Int offset = indptr[src]; offset < indptr[src + 1]; offset++) {
            rindices[offset] = src;
        }
    }
    
#ifdef VERBOSE
        printf("----------------------------\n");
        printf("n_nodes   = %d\n", n_nodes);
        printf("n_nodes   = %d\n", n_nodes);
        printf("n_edges    = %d\n", n_edges);
        printf("----------------------------\n");
#endif
}

// --
// Run

class prioritize {
    public:
        bool operator()(pair<Int, Real> &p1, pair<Int, Real> &p2) {
            return p1.second > p2.second;
        }
};

long long dijkstra_sssp(Real* dist, Int src) {
    for(Int i = 0; i < n_nodes; i++) dist[i] = 999.0;
    dist[src] = 0;

    auto t = high_resolution_clock::now();
    priority_queue<pair<Int,Real>, vector<pair<Int,Real>>, prioritize> pq;
    pq.push(make_pair(src, 0));
    
    while(!pq.empty()) {
        pair<Int, Real> curr = pq.top();
        pq.pop();

        Int curr_node  = curr.first;
        Real curr_dist = curr.second;
        if(curr_dist == dist[curr_node]) {
            for(Int offset = indptr[curr_node]; offset < indptr[curr_node + 1]; offset++) {
                Int neib      = cindices[offset];
                Real new_dist = curr_dist + data[offset];
                if(new_dist < dist[neib]) {
                    dist[neib] = new_dist;
                    pq.push(make_pair(neib, new_dist));
                }
            }
        }
    }
    auto elapsed = high_resolution_clock::now() - t;
    return duration_cast<microseconds>(elapsed).count();
}

long long frontier_sssp(Real* h_dist, Int src, Int n_gpus) {    
    // Create chunks
    Int* starts    = (Int*)malloc(n_gpus * sizeof(Int));
    Int* ends      = (Int*)malloc(n_gpus * sizeof(Int));
    Int chunk_size = (n_edges + n_gpus - 1) / n_gpus;
    for(Int i = 0; i < n_gpus; i++) {
        starts[i] = i * chunk_size;
        ends[i]   = (i + 1) * chunk_size;
    }
    ends[n_gpus - 1] = n_edges;

    // Create GPUs
    cudaSetDevice(0);
    cudaStream_t master_stream;
    cudaStreamCreateWithFlags(&master_stream, cudaStreamNonBlocking);

    struct gpu_info {
        cudaStream_t stream;
        cudaEvent_t  event;
    };
    
    std::vector<gpu_info> infos;
    
    std::cout << "n_gpus: " << n_gpus << std::endl;
    
    for(int i = 0 ; i < n_gpus ; i++) {
        std::cout << "creating " << i << std::endl;
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
    
    // Setup frontiers
    
    char* h_frontier_in  = (char*)malloc(n_nodes * sizeof(char));
    char* h_frontier_out = (char*)malloc(n_nodes * sizeof(char));
    
    for(Int i = 0; i < n_nodes; i++) h_dist[i]          = 999.0;
    for(Int i = 0; i < n_nodes; i++) h_frontier_in[i]   = -1;
    for(Int i = 0; i < n_nodes; i++) h_frontier_out[i]  = -1;
    
    h_dist[src]        = 0;
    h_frontier_in[src] = 0;
    
    // Global frontier + dist
    char* g_frontier_in;
    char* g_frontier_out;
    Real* g_dist;
    
    cudaMalloc(&g_frontier_in,  n_nodes * sizeof(char));
    cudaMalloc(&g_frontier_out, n_nodes * sizeof(char));
    cudaMalloc(&g_dist,         n_nodes * sizeof(Real));

    cudaMemcpy(g_frontier_in,  h_frontier_in,  n_nodes * sizeof(char),  cudaMemcpyHostToDevice);
    cudaMemcpy(g_frontier_out, h_frontier_out, n_nodes * sizeof(char),  cudaMemcpyHostToDevice);
    cudaMemcpy(g_dist,         h_dist,         n_nodes * sizeof(Real),  cudaMemcpyHostToDevice);
    
    // Local data, frontier + dist
    Int* all_cindices[n_gpus];
    Int* all_rindices[n_gpus];
    Real* all_data[n_gpus];
    char* all_frontier_in[n_gpus];
    char* all_frontier_out[n_gpus];
    Real* all_dist[n_gpus];
    
    for(int gid = 0; gid < n_gpus; gid++) {
        cudaSetDevice(gid);
        
        Int* l_cindices;
        Int* l_rindices;
        Real* l_data;
        char* l_frontier_in;
        char* l_frontier_out;
        Real* l_dist;

        cudaMalloc(&l_cindices,     n_edges * sizeof(Int));
        cudaMalloc(&l_rindices,     n_edges * sizeof(Int));
        cudaMalloc(&l_data,         n_edges * sizeof(Real));
        cudaMalloc(&l_frontier_in,  n_nodes * sizeof(char));
        cudaMalloc(&l_frontier_out, n_nodes * sizeof(char));
        cudaMalloc(&l_dist,         n_nodes * sizeof(Real));
        
        cudaMemcpy(l_cindices,     cindices,       n_edges * sizeof(Int),  cudaMemcpyHostToDevice);
        cudaMemcpy(l_rindices,     rindices,       n_edges * sizeof(Int),  cudaMemcpyHostToDevice);
        cudaMemcpy(l_data,         data,           n_edges * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpy(l_frontier_in,  h_frontier_in,  n_nodes * sizeof(char), cudaMemcpyHostToDevice);
        cudaMemcpy(l_frontier_out, h_frontier_out, n_nodes * sizeof(char), cudaMemcpyHostToDevice);
        cudaMemcpy(l_dist,         h_dist,         n_nodes * sizeof(Real), cudaMemcpyHostToDevice);

        all_cindices[gid]     = l_cindices;
        all_rindices[gid]     = l_rindices;
        all_data[gid]         = l_data;
        all_frontier_in[gid]  = l_frontier_in;
        all_frontier_out[gid] = l_frontier_out;
        all_dist[gid]         = l_dist;
    }

    for(int i = 0 ; i < n_gpus ; i++) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }
    
    cudaSetDevice(0);

    int iteration = 0;
    
    auto t = high_resolution_clock::now();
    while(iteration <= 7) { // hardcode convergence criteria, for now
        
        Int iteration1 = iteration + 1;
        
        // --
        // ADVANCE
        
        if(iteration >= 3 && iteration <= 5) {
            
            // Broadcast data to workers
            // Could do this better -- shaped like tree instead of start
            #pragma omp parallel for num_threads(n_gpus)
            for(int gid = 0; gid < n_gpus; gid++) {
                cudaSetDevice(gid);
                cudaMemcpyAsync(all_frontier_in[gid],  g_frontier_in,  n_nodes * sizeof(char), cudaMemcpyDeviceToDevice, infos[gid].stream);
                cudaMemcpyAsync(all_dist[gid],         g_dist,         n_nodes * sizeof(Real), cudaMemcpyDeviceToDevice, infos[gid].stream);
                cudaEventRecord(infos[gid].event, infos[gid].stream);
            }
            for(int gid = 0; gid < n_gpus; gid++)
                cudaStreamWaitEvent(master_stream, infos[gid].event, 0);
            cudaStreamSynchronize(master_stream);
            
            // Advance
            #pragma omp parallel for num_threads(n_gpus)
            for(int gid = 0; gid < n_gpus; gid++) {
                
                cudaSetDevice(gid);
                
                Int* l_cindices      = all_cindices[gid];
                Int* l_rindices      = all_rindices[gid];
                Real* l_data         = all_data[gid];
                char* l_frontier_in  = all_frontier_in[gid];
                char* l_frontier_out = all_frontier_out[gid];
                Real* l_dist         = all_dist[gid];
                
                auto edge_op = [=] __device__(int const& offset) -> void {
                    Int src = l_rindices[offset];
                    if(l_frontier_in[src] != iteration) return;
                    
                    Int dst = l_cindices[offset];
                    
                    Real new_dist = l_dist[src] + l_data[offset];     
                    Real old_dist = atomicMin(l_dist + dst, new_dist);
                    if(new_dist < old_dist)
                        l_frontier_out[dst] = iteration1;
                };
                
                thrust::for_each(
                    thrust::cuda::par.on(infos[gid].stream),
                    thrust::make_counting_iterator<Int>(starts[gid]),
                    thrust::make_counting_iterator<Int>(ends[gid]),
                    edge_op
                );
                
                auto merge_op = [=] __device__(int const& dst) -> void {
                    if(l_frontier_out[dst] != iteration1) return;
                    if(g_frontier_out[dst] != iteration1) g_frontier_out[dst] = iteration1; // Could write directly to sharded array
                    atomicMin(g_dist + dst, l_dist[dst]);                                   // Could write directly to sharded array (maybe?)
                };
                
                thrust::for_each(
                    thrust::cuda::par.on(infos[gid].stream),
                    thrust::make_counting_iterator<Int>(0),
                    thrust::make_counting_iterator<Int>(n_nodes),
                    merge_op
                );
                
                cudaEventRecord(infos[gid].event, infos[gid].stream);
            }
            
            for(int gid = 0; gid < n_gpus; gid++)
                cudaStreamWaitEvent(master_stream, infos[gid].event, 0);
            cudaStreamSynchronize(master_stream);
        } else {
            // Single-GPU mode
            
            int gid = 0;
            cudaSetDevice(gid);
            
            Int* l_cindices = all_cindices[gid];
            Int* l_rindices = all_rindices[gid];
            Real* l_data    = all_data[gid];

            auto edge_op = [=] __device__(int const& offset) -> void {
                Int src = l_rindices[offset];               // local
                if(g_frontier_in[src] != iteration) return; // local
                
                Int dst = l_cindices[offset]; // local
                
                Real new_dist = g_dist[src] + l_data[offset];      // (global, local)
                Real old_dist = atomicMin(g_dist + dst, new_dist); // (global write)
                if(new_dist < old_dist)
                    g_frontier_out[dst] = iteration1; // global
            };
            
            thrust::for_each(
                thrust::cuda::par.on(infos[gid].stream),
                thrust::make_counting_iterator<Int>(0),
                thrust::make_counting_iterator<Int>(n_edges),
                edge_op
            );
            
            cudaEventRecord(infos[gid].event, infos[gid].stream);
            cudaStreamWaitEvent(master_stream, infos[gid].event, 0);
            cudaStreamSynchronize(master_stream);
        }
        
        char* tmp      = g_frontier_in;
        g_frontier_in  = g_frontier_out;
        g_frontier_out = tmp;
                
        iteration++;
    }
    
    cudaMemcpy(h_dist, g_dist, n_nodes * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    auto elapsed = high_resolution_clock::now() - t;
    return duration_cast<microseconds>(elapsed).count();
}


int main(int n_args, char** argument_array) {
    int n_gpus = 1;
    cudaGetDeviceCount(&n_gpus);
    
    // ---------------- INPUT ----------------

    loal_data(argument_array[1]);

    int src = 0;
    // ---------------- DIJKSTRA ----------------
    
    Real* dijkstra_dist = (Real*)malloc(n_nodes * sizeof(Real));
    auto ms1 = dijkstra_sssp(dijkstra_dist, src);
    
    // ---------------- FRONTIER ----------------
    
    Real* frontier_dist = (Real*)malloc(n_nodes * sizeof(Real));
    auto ms2 = frontier_sssp(frontier_dist, src, n_gpus);

    for(Int i = 0; i < 40; i++) std::cout << dijkstra_dist[i] << " ";
    std::cout << std::endl;
    for(Int i = 0; i < 40; i++) std::cout << frontier_dist[i] << " ";
    std::cout << std::endl;

    int n_errors = 0;
    for(Int i = 0; i < n_nodes; i++) {
        if(dijkstra_dist[i] != frontier_dist[i]) n_errors++;
    }
    
    std::cout << "ms1=" << ms1 << " | ms2=" << ms2 << " | n_errors=" << n_errors << std::endl;
    
    return 0;
}
