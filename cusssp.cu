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
#include <limits>
#include <vector>

#include "timer.hxx"

using namespace std;
using namespace std::chrono;

// --
// Global defs

typedef int Int;
typedef float Real;

// graph
Int n_nodes, n_edges;
Int* indptr;
Int* rindices;
Int* cindices;
Real* data;

int src = 0;

// --
// Helpers

__device__ static float atomicMin(float* address, float value) {
  int* addr_as_int = reinterpret_cast<int*>(address);
  int old = *addr_as_int;
  int expected;
  do {
    expected = old;
    old = atomicCAS(addr_as_int, expected,
                      __float_as_int(::fminf(value, __int_as_float(expected))));
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
    fread(&n_edges,   sizeof(Int), 1, ptr);

    indptr    = (Int*)  malloc(sizeof(Int)  * (n_nodes + 1)  );
    cindices  = (Int*)  malloc(sizeof(Int)  * n_edges        );
    rindices  = (Int*)  malloc(sizeof(Int)  * n_edges        );
    data      = (Real*) malloc(sizeof(Real) * n_edges        );

    fread(indptr,  sizeof(Int),   n_nodes + 1 , ptr);  // send directy to the memory since thats what the thing is.
    fread(cindices, sizeof(Int),  n_edges     , ptr);
    fread(data,    sizeof(Real),  n_edges     , ptr);
    
    for(Int src = 0; src < n_nodes; src++) {
        for(Int offset = indptr[src]; offset < indptr[src + 1]; offset++) {
            rindices[offset] = src;
        }
    }
}

// --
// Run

class prioritize {
    public:
        bool operator()(pair<Int, Real> &p1, pair<Int, Real> &p2) {
            return p1.second > p2.second;
        }
};

long long cpu_sssp(Real* dist, Int src) {
    for(Int i = 0; i < n_nodes; i++) dist[i] = std::numeric_limits<Real>::max();
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

long long cuda_sssp(Real* dist, Int src, Int n_threads) {
    
    // --
    // Copy graph from host to device
    
    Int* d_cindices;
    Int* d_rindices;
    Real* d_data;

    cudaMalloc(&d_cindices,  n_edges       * sizeof(Int));
    cudaMalloc(&d_rindices,  n_edges       * sizeof(Int));
    cudaMalloc(&d_data,      n_edges       * sizeof(Real));

    cudaMemcpy(d_cindices, cindices,  n_edges       * sizeof(Int),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_rindices, rindices,  n_edges       * sizeof(Int),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_data,     data,      n_edges       * sizeof(Real), cudaMemcpyHostToDevice);
    
    // --
    // Setup problem on host
    
    Int* frontier_in  = (Int*)malloc(n_nodes * sizeof(Int));
    Int* frontier_out = (Int*)malloc(n_nodes * sizeof(Int));
    
    for(Int i = 0; i < n_nodes; i++) dist[i]          = std::numeric_limits<Real>::max();
    for(Int i = 0; i < n_nodes; i++) frontier_in[i]   = -1;
    for(Int i = 0; i < n_nodes; i++) frontier_out[i]  = -1;
    
    dist[src]        = 0;
    frontier_in[src] = 0;
    
    int iteration = 0;
    
    // --
    // Copy data to device
    
    Int* d_frontier_in;
    Int* d_frontier_out;
    Real* d_dist;
    
    cudaMalloc(&d_frontier_in,  n_nodes * sizeof(Int));
    cudaMalloc(&d_frontier_out, n_nodes * sizeof(Int));
    cudaMalloc(&d_dist,         n_nodes * sizeof(Real));

    cudaMemcpy(d_frontier_in,  frontier_in,  n_nodes * sizeof(Int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frontier_out, frontier_out, n_nodes * sizeof(Int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist,         dist,         n_nodes * sizeof(Real), cudaMemcpyHostToDevice);

    // --
    // Run
    
    cudaDeviceSynchronize();
    auto t = high_resolution_clock::now();
    
    // while(true) {
    while(iteration <= 7) {
        printf("iteration %d\n", iteration);
        
        Int iteration1 = iteration + 1;
        
        // Advance        
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
            thrust::device,
            thrust::make_counting_iterator<int>(0),
            thrust::make_counting_iterator<int>(n_edges),
            edge_op
        );

        // Swap input and output
        Int* tmp       = d_frontier_in;
        d_frontier_in  = d_frontier_out;
        d_frontier_out = tmp;
                
        // // Convergence criterion
        // auto keep_going = thrust::reduce(
        //     thrust::device, d_frontier_in + 0, d_frontier_in + n_nodes
        // );
        // if(keep_going == 0) break; 

        // Reset output frontier
        // thrust::fill_n(thrust::device, 
        //     d_frontier_out, n_nodes, false
        // );
        
        iteration++;
        
    }
    
    cudaMemcpy(dist, d_dist, n_nodes * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    auto elapsed = high_resolution_clock::now() - t;
    return duration_cast<microseconds>(elapsed).count();
}


int main(int n_args, char** argument_array) {
    
    // ---------------- INPUT ------------------------------

    load_data(argument_array[1]);
    
    // ---------------- CPU BASELINE -----------------------
    
    Real* cpu_dist = (Real*)malloc(n_nodes * sizeof(Real));
    auto cpu_time  = cpu_sssp(cpu_dist, src);
    
    // ---------------- GPU IMPLEMENTATION -----------------
    
    Real* gpu_dist = (Real*)malloc(n_nodes * sizeof(Real));
    auto gpu_time  = cuda_sssp(gpu_dist, src, 1);

    // ---------------- VALIDATION -------------------------
    
    for(Int i = 0; i < 40; i++) std::cout << cpu_dist[i] << " ";
    std::cout << std::endl;
    
    for(Int i = 0; i < 40; i++) std::cout << gpu_dist[i] << " ";
    std::cout << std::endl;

    int n_errors = 0;
    for(Int i = 0; i < n_nodes; i++) {
        if(cpu_dist[i] != gpu_dist[i]) n_errors++;
    }
    
    std::cout << "cpu_time=" << cpu_time << " | gpu_time=" << gpu_time << " | n_errors=" << n_errors << std::endl;
    
    return 0;
}
