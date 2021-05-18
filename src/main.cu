#pragma GCC diagnostic ignored "-Wunused-result"

#include "sssp_cpu.hxx"
#include "sssp_1gpu.hxx"
#include "sssp_mgpu.hxx"

#define RUN_CPU

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
}

int main(int n_args, char** argument_array) {
    int n_gpus = 1;
    cudaGetDeviceCount(&n_gpus);
    
    // ---------------- INPUT ----------------

    load_data(argument_array[1]);

    int src = 0;
    
    // ---------------- CPU ----------------
    
    Real* cpu_dist = (Real*)malloc(n_nodes * sizeof(Real));
    long long cpu_time = 0;
#ifdef RUN_CPU
    cpu_time = sssp_cpu(cpu_dist, src, n_nodes, n_edges, indptr, cindices, data);
#endif
    
    // ---------------- GPU ----------------
    
    Real* gpu_dist = (Real*)malloc(n_nodes * sizeof(Real));
    long long gpu_time = 0;
    if(n_gpus == 1) {
        gpu_time = sssp_1gpu(gpu_dist, src, n_nodes, n_edges, rindices, cindices, data);
    } else {
        gpu_time = sssp_mgpu(gpu_dist, src, n_nodes, n_edges, rindices, cindices, data, n_gpus);
    }

    for(Int i = 0; i < min(n_nodes, 40); i++) std::cout << cpu_dist[i] << " ";
    std::cout << std::endl;
    for(Int i = 0; i < min(n_nodes, 40); i++) std::cout << gpu_dist[i] << " ";
    std::cout << std::endl;

    // ---------------- VALIDATE ----------------
    
    int n_errors = 0;
#ifdef RUN_CPU
    for(Int i = 0; i < n_nodes; i++) {
        if(cpu_dist[i] != gpu_dist[i]) n_errors++;
    }
#endif
    
    std::cout << "cpu_time=" << cpu_time << " | gpu_time=" << gpu_time << " | n_errors=" << n_errors << std::endl;
    
    return 0;
}
