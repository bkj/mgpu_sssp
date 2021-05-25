// sssp_1gpu.hxx

#pragma once

#include "thrust/device_vector.h"

#include "helpers.hxx"

template <typename Int, typename Real>
long long sssp_1gpu(Real* h_dist, Int n_seeds, Int* seeds, Int n_nodes, Int n_edges, Int* rindices, Int* cindices, Real* data) {
    
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
    
    char* h_frontier_in  = (char*)malloc(n_nodes * sizeof(char));
    char* h_frontier_out = (char*)malloc(n_nodes * sizeof(char));
    
    for(Int i = 0; i < n_nodes; i++) h_dist[i]          = std::numeric_limits<Real>::max();
    for(Int i = 0; i < n_nodes; i++) h_frontier_in[i]   = -1;
    for(Int i = 0; i < n_nodes; i++) h_frontier_out[i]  = -1;
    
    for(Int seed = 0; seed < n_seeds; seed++) {
        h_dist[seed]         = 0;
        h_frontier_in[seed]  = 0;
    }
    
    // --
    // Copy data to device
    
    char* d_frontier_in;
    char* d_frontier_out;
    char* d_keep_going;
    Real* d_dist;
    
    cudaMalloc(&d_frontier_in,  n_nodes * sizeof(char));
    cudaMalloc(&d_frontier_out, n_nodes * sizeof(char));
    cudaMalloc(&d_keep_going,   1       * sizeof(char));
    cudaMalloc(&d_dist,         n_nodes * sizeof(Real));
    
    cudaMemcpy(d_frontier_in,  h_frontier_in,  n_nodes * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frontier_out, h_frontier_out, n_nodes * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemset(d_keep_going,   0,              1       * sizeof(char));
    cudaMemcpy(d_dist,         h_dist,         n_nodes * sizeof(Real), cudaMemcpyHostToDevice);
    
    // --
    // Run
    
    cudaDeviceSynchronize();
    cuda_timer_t timer;
    timer.start();
    
    char iter       = 0;
    char keep_going = -1;
    while(true) {
        char next_iter = iter + 1;
        
        // Advance        
        auto edge_op = [=] __device__(int const& offset) -> void {
            Int src = d_rindices[offset];
            if(d_frontier_in[src] != iter) return;
            d_keep_going[0] = next_iter;
            
            Int dst = d_cindices[offset];
            
            Real new_dist = d_dist[src] + d_data[offset];
            Real old_dist = atomicMin(d_dist + dst, new_dist);
            if(new_dist < old_dist)
                d_frontier_out[dst] = next_iter;
        };
        
        thrust::for_each(
            thrust::device,
            thrust::make_counting_iterator<int>(0),
            thrust::make_counting_iterator<int>(n_edges),
            edge_op
        );

        // Swap input and output
        char* tmp      = d_frontier_in;
        d_frontier_in  = d_frontier_out;
        d_frontier_out = tmp;
                
        // Convergence criterion
        cudaMemcpy(&keep_going, d_keep_going, 1 * sizeof(char), cudaMemcpyDeviceToHost);
        if(keep_going != next_iter) break;
        
        iter++;
    }
    
    cudaMemcpy(h_dist, d_dist, n_nodes * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    return timer.stop();
}