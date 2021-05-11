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

void dijkstra_sssp(Real* dist, Int src) {
    for(Int i = 0; i < n_nodes; i++) dist[i] = 999.0;
    dist[src] = 0;

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
}

void advance(Real* dist, bool* frontier_in, bool* frontier_out, Int start, Int end) {
    for(Int src = start; src < end; src++) {
        if(!frontier_in[src]) continue;
        frontier_in[src] = false;
        
        for(int offset = indptr[src]; offset < indptr[src + 1]; offset++) {
            Int dst       = indices[offset];
            Real new_dist = dist[src] + data[offset];
            
            if(new_dist < dist[dst]) {
                dist[dst]         = new_dist; // false sharing? bad atomics?           
                frontier_out[dst] = true;     // false sharing?
            }
        }
    }
}

long long frontier_sssp(Real* dist, Int src, Int n_threads) {
    bool* frontier_in  = (bool*)malloc(n_nodes * sizeof(bool));
    bool* frontier_out = (bool*)malloc(n_nodes * sizeof(bool));
    
    for(Int i = 0; i < n_nodes; i++) dist[i]          = 999.0;
    for(Int i = 0; i < n_nodes; i++) frontier_in[i]   = false;
    for(Int i = 0; i < n_nodes; i++) frontier_out[i]  = false;
    
    dist[src]        = 0;
    frontier_in[src] = true;
    
    int iteration = 0;
    
    Real* ldists = (Real*)malloc(n_nodes * n_threads * sizeof(Real));
    for(Int i = 0; i < n_nodes * n_threads; i++) {
        ldists[i] = dist[i % n_nodes];
    }
    
    Int* starts    = (Int*)malloc(n_threads * sizeof(Int));
    Int* ends      = (Int*)malloc(n_threads * sizeof(Int));
    Int chunk_size = (n_nodes + n_threads - 1) / n_threads;
    for(Int i = 0; i < n_threads; i++) {
        starts[i] = i * chunk_size;
        ends[i]   = (i + 1) * chunk_size;
    }
    ends[n_threads - 1] = n_nodes;
    
    // Data
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

    auto t = high_resolution_clock::now();
    for(int it = 0; it < 5; it++) {
        
        // <<
        // auto node_op = [=] __device__(int const& src) -> bool {
        //     if(!d_frontier_in[src]) return false;
            
        //     for(int offset = d_indptr[src]; offset < d_indptr[src + 1]; offset++) {
        //         Int dst       = d_indices[offset];
        //         Real new_dist = d_dist[src] + d_data[offset];
                
        //         if(new_dist < d_dist[dst]) {
        //             d_dist[dst]         = new_dist; // false sharing? bad atomics?           
        //             d_frontier_out[dst] = true;     // false sharing?
        //         }
        //     }
        //     return false;
        // };
        
        // thrust::transform(
        //     thrust::device,
        //     thrust::make_counting_iterator<int>(0),
        //     thrust::make_counting_iterator<int>(n_nodes),
        //     thrust::make_discard_iterator(),
        //     node_op
        // );
        
        // --
        
        auto edge_op = [=] __device__(int const& offset) -> bool {
            Int src = d_rindices[offset];
            Int dst = d_indices[offset];
            
            if(!d_frontier_in[src]) return false;
            
            Real new_dist = d_dist[src] + d_data[offset];
            Real old_dist = atomicMin(d_dist + dst, new_dist);
            if(new_dist < old_dist) {
                d_frontier_out[dst] = true;
            }
            
            return false;
        };

        thrust::transform(
            thrust::device,
            thrust::make_counting_iterator<int>(0),
            thrust::make_counting_iterator<int>(n_edges),
            thrust::make_discard_iterator(),
            edge_op
        );
        
        // >>

        thrust::fill_n(
            thrust::device,
            d_frontier_in,
            n_nodes,
            false
        );

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
    
    // ---------------- INPUT ----------------

    load_data(argument_array[1]);

    int src = 0;
    // ---------------- DIJKSTRA ----------------
    
    Real* dijkstra_dist = (Real*)malloc(n_nodes * sizeof(Real));
    auto t1       = high_resolution_clock::now();
    dijkstra_sssp(dijkstra_dist, src);
    auto elapsed1 = high_resolution_clock::now() - t1;
    long long ms1 = duration_cast<microseconds>(elapsed1).count();
    
    // ---------------- FRONTIER ----------------
    
    Real* frontier_dist = (Real*)malloc(n_nodes * sizeof(Real));
    auto ms2 = frontier_sssp(frontier_dist, src, 1);

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
