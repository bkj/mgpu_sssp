// sssp_1gpu.hxx

#pragma once

#include "thrust/device_vector.h"

#include "helpers.hxx"

template <typename Int, typename Real>
long long sssp_1gpu(Real* dist, Int src, Int n_nodes, Int n_edges, Int* rindices, Int* cindices, Real* data) {
    
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
    
    bool* frontier_in  = (bool*)malloc(n_nodes * sizeof(bool));
    bool* frontier_out = (bool*)malloc(n_nodes * sizeof(bool));
    
    for(Int i = 0; i < n_nodes; i++) dist[i]          = std::numeric_limits<Real>::max();
    for(Int i = 0; i < n_nodes; i++) frontier_in[i]   = false;
    for(Int i = 0; i < n_nodes; i++) frontier_out[i]  = false;
    
    dist[src]        = 0;
    frontier_in[src] = true;
    
    int iter = 0;
    
    // --
    // Copy data to device
    
    bool* d_frontier_in;
    bool* d_frontier_out;
    Real* d_dist;
    
    cudaMalloc(&d_frontier_in,  n_nodes * sizeof(bool));
    cudaMalloc(&d_frontier_out, n_nodes * sizeof(bool));
    cudaMalloc(&d_dist,         n_nodes * sizeof(Real));

    cudaMemcpy(d_frontier_in,  frontier_in,  n_nodes * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frontier_out, frontier_out, n_nodes * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist,         dist,         n_nodes * sizeof(Real), cudaMemcpyHostToDevice);

    // --
    // Run
    
    cudaDeviceSynchronize();
    cuda_timer_t timer;
    timer.start();
    
    while(true) {
        printf("iter %d\n", iter);
        
        // Advance        
        auto edge_op = [=] __device__(int const& offset) -> void {
            Int src = d_rindices[offset];
            if(!d_frontier_in[src]) return;
            
            Int dst = d_cindices[offset];
            
            Real new_dist = d_dist[src] + d_data[offset];
            Real old_dist = atomicMin(d_dist + dst, new_dist);
            if(new_dist < old_dist)
                d_frontier_out[dst] = true;
        };
        
        thrust::for_each(
            thrust::device,
            thrust::make_counting_iterator<int>(0),
            thrust::make_counting_iterator<int>(n_edges),
            edge_op
        );

        // Swap input and output
        bool* tmp      = d_frontier_in;
        d_frontier_in  = d_frontier_out;
        d_frontier_out = tmp;
                
        // Convergence criterion
        auto keep_going = thrust::reduce(
            thrust::device, d_frontier_in + 0, d_frontier_in + n_nodes
        );
        if(keep_going == 0) break; 

        // Reset output frontier
        thrust::fill_n(thrust::device, 
            d_frontier_out, n_nodes, false
        );
        
        iter++;
    }
    
    cudaMemcpy(dist, d_dist, n_nodes * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    return timer.stop();
}