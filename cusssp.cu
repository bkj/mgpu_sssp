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

void load_data(std::string inpath) {
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

long long frontier_sssp(Real* dist, Int src, Int n_gpus) {
    Int* frontier_in  = (Int*)malloc(n_nodes * sizeof(Int));
    Int* frontier_out = (Int*)malloc(n_nodes * sizeof(Int));
    
    for(Int i = 0; i < n_nodes; i++) dist[i]          = 999.0;
    for(Int i = 0; i < n_nodes; i++) frontier_in[i]   = -1;
    for(Int i = 0; i < n_nodes; i++) frontier_out[i]  = -1;
    
    dist[src]        = 0;
    frontier_in[src] = 0;
    
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
    
    Int* g_cindices[n_gpus];
    Int* g_rindices[n_gpus];
    Real* g_data[n_gpus];
    for(int gid = 0; gid < n_gpus; gid++) {
        cudaSetDevice(gid);
        
        Int* d_cindices;
        Int* d_rindices;
        Real* d_data;

        cudaMalloc(&d_cindices, n_edges * sizeof(Int));
        cudaMalloc(&d_rindices, n_edges * sizeof(Int));
        cudaMalloc(&d_data,     n_edges * sizeof(Real));
        
        cudaMemcpy(d_cindices, cindices,  n_edges * sizeof(Int),      cudaMemcpyHostToDevice);
        cudaMemcpy(d_rindices, rindices, n_edges * sizeof(Int),       cudaMemcpyHostToDevice);
        cudaMemcpy(d_data,     data,     n_edges * sizeof(Real),      cudaMemcpyHostToDevice);
        
        g_cindices[gid] = d_cindices;
        g_rindices[gid] = d_rindices;
        g_data[gid]     = d_data;
    }
    
    cudaSetDevice(0);
    
    // Frontiers
    Int* d_frontier_in;
    Int* d_frontier_out;
    Real* d_dist;
    
    cudaMalloc(&d_frontier_in,  n_nodes * sizeof(Int));
    cudaMalloc(&d_frontier_out, n_nodes * sizeof(Int));
    cudaMalloc(&d_dist,         n_nodes * sizeof(Real));

    cudaMemcpy(d_frontier_in,  frontier_in,  n_nodes * sizeof(Int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frontier_out, frontier_out, n_nodes * sizeof(Int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist,         dist,         n_nodes * sizeof(Real), cudaMemcpyHostToDevice);

    for(int i = 0 ; i < n_gpus ; i++) cudaDeviceSynchronize();

    auto t = high_resolution_clock::now();
    while(iteration < 5) {
        
        Int iteration1 = iteration + 1;
        
        #pragma omp parallel for num_threads(n_gpus)
        for(int tid = 0; tid < n_gpus; tid++) {
            cudaSetDevice(tid);
            
            Int* d_cindices = g_cindices[tid];
            Int* d_rindices = g_rindices[tid];
            Real* d_data    = g_data[tid];

            auto edge_op = [=] __device__(int const& offset) -> void {
                Int src = d_rindices[offset];
                if(d_frontier_in[src] != iteration) return;
                
                Int dst = d_cindices[offset];
                
                Real new_dist = d_dist[src] + d_data[offset];
                Real old_dist = atomicMin(d_dist + dst, new_dist);
                if(new_dist < old_dist)
                    d_frontier_out[dst] = iteration1;
            };
            
            thrust::for_each(
                thrust::cuda::par.on(infos[tid].stream),
                thrust::make_counting_iterator<Int>(starts[tid]),
                thrust::make_counting_iterator<Int>(ends[tid]),
                edge_op
            );
            cudaEventRecord(infos[tid].event, infos[tid].stream);
        }
        
        for(int tid = 0; tid < n_gpus; tid++)
            cudaStreamWaitEvent(master_stream, infos[tid].event, 0);
        cudaStreamSynchronize(master_stream);

        Int* tmp       = d_frontier_in;
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
