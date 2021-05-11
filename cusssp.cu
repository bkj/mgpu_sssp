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
Int n_rows, n_cols, n_nnz;
Int* indptr;
Int* rindices;
Int* indices;
Real* data;

Int n_nodes;
Int n_edges;

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

__global__ void edge_kernel(Int start, Int end, Real* d_dist, Int* d_rindices, Int* d_indices, Real* d_data, bool* d_frontier_in, bool* d_frontier_out) {
    Int offset = start + blockIdx.x * blockDim.x + threadIdx.x;
    if(offset >= end) return;
    
    Int src = d_rindices[offset];
    Int dst = d_indices[offset];
    
    if(!d_frontier_in[src]) return;
    
    Real new_dist = d_dist[src] + d_data[offset];
    Real old_dist = atomicMin(d_dist + dst, new_dist);
    if(new_dist < old_dist)
        d_frontier_out[dst] = true;
};

// --
// IO

void load_data(std::string inpath) {
    FILE *ptr;
    ptr = fopen(inpath.c_str(), "rb");

    fread(&n_rows,   sizeof(Int), 1, ptr);
    fread(&n_cols,   sizeof(Int), 1, ptr);
    fread(&n_nnz,    sizeof(Int), 1, ptr);

    indptr   = (Int*)  malloc(sizeof(Int)  * (n_rows + 1)  );
    indices  = (Int*)  malloc(sizeof(Int)  * n_nnz         );
    data     = (Real*) malloc(sizeof(Real) * n_nnz         );

    fread(indptr,  sizeof(Int),   n_rows + 1 , ptr);  // send directy to the memory since thats what the thing is.
    fread(indices, sizeof(Int),   n_nnz      , ptr);
    fread(data,    sizeof(Real),  n_nnz      , ptr);

    n_nodes = n_rows;
    n_edges = n_nnz;
    
    rindices = (Int*) malloc(sizeof(Int) * n_nnz);
    for(Int src = 0; src < n_nodes; src++) {
        for(Int offset = indptr[src]; offset < indptr[src + 1]; offset++) {
            rindices[offset] = src;
        }
    }
    
#ifdef VERBOSE
        printf("----------------------------\n");
        printf("n_rows   = %d\n", n_rows);
        printf("n_cols   = %d\n", n_cols);
        printf("n_nnz    = %d\n", n_nnz);
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
                Int neib      = indices[offset];
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

long long frontier_sssp(Real* dist, Int src, Int n_gpus) {
    bool* frontier_in  = (bool*)malloc(n_nodes * sizeof(bool));
    bool* frontier_out = (bool*)malloc(n_nodes * sizeof(bool));
    
    for(Int i = 0; i < n_nodes; i++) dist[i]          = 999.0;
    for(Int i = 0; i < n_nodes; i++) frontier_in[i]   = false;
    for(Int i = 0; i < n_nodes; i++) frontier_out[i]  = false;
    
    dist[src]        = 0;
    frontier_in[src] = true;
    
    int iteration = 0;
    
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
    
    // Data
    Int* all_d_indptr[n_gpus];
    Int* all_d_indices[n_gpus];
    Int* all_d_rindices[n_gpus];
    Real* all_d_data[n_gpus];

    for(int i = 0; i < n_gpus; i++) {
        Int* d_indptr;
        Int* d_indices;
        Int* d_rindices;
        Real* d_data;

        cudaMalloc(&d_indptr,  (n_nodes + 1) * sizeof(Int));
        cudaMalloc(&d_indices,  n_edges * sizeof(Int));
        cudaMalloc(&d_rindices, n_edges * sizeof(Int));
        cudaMalloc(&d_data,     n_edges * sizeof(Real));
        
        cudaMemcpy(d_indptr,   indptr,   (n_nodes + 1) * sizeof(Int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_indices,  indices,  n_edges * sizeof(Int),       cudaMemcpyHostToDevice);
        cudaMemcpy(d_rindices, rindices, n_edges * sizeof(Int),       cudaMemcpyHostToDevice);
        cudaMemcpy(d_data,     data,     n_edges * sizeof(Real),      cudaMemcpyHostToDevice);
        
        all_d_indptr[i] = d_indptr;
        all_d_indices[i] = d_indices;
        all_d_rindices[i] = d_rindices;
        all_d_data[i] = d_data;
    }
    
    // for(int i = 0; i < n_gpus; i++) {
    //     cudaMemAdvise(d_indptr,   (n_nodes + 1) * sizeof(Int), cudaMemAdviseSetReadMostly,  i);
    //     cudaMemAdvise(d_indices,  n_edges       * sizeof(Int), cudaMemAdviseSetReadMostly,  i);
    //     cudaMemAdvise(d_rindices, n_edges       * sizeof(Int), cudaMemAdviseSetReadMostly,  i);
    //     cudaMemAdvise(d_data,     n_edges       * sizeof(Real), cudaMemAdviseSetReadMostly, i);
    // }
    
    // Frontiers
    bool* d_frontier_in;
    bool* d_frontier_out;
    Real* d_dist;
    
    cudaMalloc(&d_frontier_in,  n_nodes * sizeof(bool));
    cudaMalloc(&d_frontier_out, n_nodes * sizeof(bool));
    cudaMalloc(&d_dist,         n_nodes * sizeof(Real));

    cudaMemcpy(d_frontier_in,  frontier_in,  n_nodes * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frontier_out, frontier_out, n_nodes * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist,         dist,         n_nodes * sizeof(Real), cudaMemcpyHostToDevice);

    for(int i = 0 ; i < n_gpus ; i++) cudaDeviceSynchronize();

    auto t = high_resolution_clock::now();
    for(int it = 0; it < 1; it++) {

        // auto edge_op = [=] __device__(int const& offset) -> bool {
        //     Int src = d_rindices[offset];
        //     Int dst = d_indices[offset];
            
        //     if(!d_frontier_in[src]) return false;
            
        //     Real new_dist = d_dist[src] + d_data[offset];
        //     Real old_dist = atomicMin(d_dist + dst, new_dist);
        //     if(new_dist < old_dist)
        //         d_frontier_out[dst] = true;
            
        //     return false;
        // };
        
        #pragma omp parallel for num_threads(n_gpus)
        for(int tid = 0; tid < n_gpus; tid++) {
            cudaSetDevice(tid);
            // thrust::transform(
            //     thrust::cuda::par.on(infos[tid].stream),
            //     thrust::make_counting_iterator<int>(starts[tid]),
            //     thrust::make_counting_iterator<int>(ends[tid]),
            //     thrust::make_discard_iterator(),
            //     edge_op
            // );
            edge_kernel<<<(n_edges + 255) / 256, 256, 0, infos[tid].stream>>>(
                starts[tid],
                ends[tid],
                d_dist, 
                all_d_rindices[tid], 
                all_d_indices[tid], 
                all_d_data[tid],
                d_frontier_in,
                d_frontier_out
            );
            cudaEventRecord(infos[tid].event, infos[tid].stream);
        }
        
        for(int tid = 0; tid < n_gpus; tid++) {
            cudaStreamWaitEvent(master_stream, infos[tid].event, 0);
        }
        cudaStreamSynchronize(master_stream);
          
        thrust::fill_n(
            thrust::cuda::par.on(infos[0].stream),
            d_frontier_in,
            n_nodes,
            false
        );
        cudaEventRecord(infos[0].event, infos[0].stream);
        cudaStreamWaitEvent(master_stream, infos[0].event, 0);
        cudaStreamSynchronize(master_stream);

        bool* tmp      = d_frontier_in;
        d_frontier_in  = d_frontier_out;
        d_frontier_out = tmp;
                
        iteration++;
    }
    
    cudaMemcpy(dist, d_dist, n_nodes * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    auto elapsed = high_resolution_clock::now() - t;
    return duration_cast<microseconds>(elapsed).count();
}


int main(int n_args, char** argument_array) {
    int n_gpus = 1;
    cudaGetDeviceCount(&n_gpus);
    
    // ---------------- INPUT ----------------

    load_data(argument_array[1]);

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
