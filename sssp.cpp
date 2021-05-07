#pragma GCC diagnostic ignored "-Wunused-result"

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
Int* indices;
Real* data;

Int n_nodes;
Int n_edges;

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
    
    for(Int i = 0; i < n_nodes; i++)   dist[i]          = 999.0;
    for(Int i = 0; i < n_nodes; i++)   frontier_in[i]   = false;
    for(Int i = 0; i < n_nodes; i++)   frontier_out[i]  = false;
    
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

    auto t = high_resolution_clock::now();
    for(int it = 0; it < 5; it++) {

        // Advance
        #pragma omp parallel for num_threads(n_threads)
        for(int tid = 0; tid < n_threads; tid++) {
            advance(
                // ldists + (tid * n_nodes),
                dist,
                frontier_in,
                frontier_out,
                starts[tid],
                ends[tid]
            );
        }
        
        // #pragma omp parallel for num_threads(n_threads)
        // for(Int i = 0; i < n_nodes; i++) {
        //     if(!frontier_out[i]) continue;
            
        //     // Reduce
        //     Real tmp = dist[i];
        //     for(Int tid = 0; tid < n_threads; tid++) {
        //         if(ldists[tid * n_nodes + i] < tmp)
        //             tmp = ldists[tid * n_nodes + i];
        //     }
            
        //     // Broadcast
        //     for(Int tid = 0; tid < n_threads; tid++) {
        //         ldists[tid * n_nodes + i] = tmp;
        //     }
            
        //     dist[i] = tmp;
        // }

        bool* tmp    = frontier_in;
        frontier_in  = frontier_out;
        frontier_out = tmp;
                
        iteration++;
    }
    auto elapsed = high_resolution_clock::now() - t;
    return duration_cast<microseconds>(elapsed).count();
}


int main(int n_args, char** argument_array) {
    int n_threads = 0;
    #pragma omp parallel
    {
        n_threads = omp_get_num_threads();
    }
    
    // ---------------- INPUT ----------------

    load_data(argument_array[1]);

    n_nodes = n_rows;
    n_edges = n_nnz;
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

    Real* frontier_dist2 = (Real*)malloc(n_nodes * sizeof(Real));
    auto ms3 = frontier_sssp(frontier_dist2, src, n_threads);

    for(Int i = 0; i < 40; i++) std::cout << dijkstra_dist[i] << " ";
    std::cout << std::endl;
    for(Int i = 0; i < 40; i++) std::cout << frontier_dist[i] << " ";
    std::cout << std::endl;
    for(Int i = 0; i < 40; i++) std::cout << frontier_dist2[i] << " ";
    std::cout << std::endl;

    int n_errors = 0;
    for(Int i = 0; i < n_nodes; i++) {
        if(dijkstra_dist[i] != frontier_dist[i]) n_errors++;
        if(dijkstra_dist[i] != frontier_dist2[i]) n_errors++;
    }
    
    std::cout << "ms1=" << ms1 << " | ms2=" << ms2 << " | ms3=" << ms3 << " | n_errors=" << n_errors << std::endl;
    
    return 0;
}
