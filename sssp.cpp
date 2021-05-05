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

#define VERBOSE

// --
// Global defs

typedef int Int;
typedef float Real;

// params
Real alpha   = 0.85;
Real tol     = 1e-6;
Int max_iter = 1000;

// graph
Int n_rows, n_cols, n_nnz;
Int* indptr;
Int* indices;
Real* data;

Int n_nodes;
Int n_edges;

// output
Real* x;

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
    fread(data,    sizeof(Real),     n_nnz      , ptr);

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

void frontier_sssp(Real* dist, Int src) {
    
    Int* visited      = (Int*)malloc(n_nodes * sizeof(Int));
    Int* frontier_in  = (Int*)malloc(n_nodes * sizeof(Int));
    Int* frontier_out = (Int*)malloc(n_nodes * sizeof(Int));
    
    for(Int i = 0; i < n_nodes; i++) dist[i]         = 999.0;
    for(Int i = 0; i < n_nodes; i++) visited[i]      = -1;
    for(Int i = 0; i < n_nodes; i++) frontier_in[i]  = -1;
    for(Int i = 0; i < n_nodes; i++) frontier_out[i] = -1;
    
    dist[src]      = 0;
    frontier_in[0] = src;
    
    unsigned int counter_in  = 1;
    unsigned int counter_out = 0;
    int iteration = 0;
    
    while(true) {
        
        for(unsigned int i = 0; i < counter_in; i++) {
            Int src = frontier_in[i];
            if(src == -1) continue;
            
            for(int offset = indptr[src]; offset < indptr[src + 1]; offset++) {
                Int dst       = indices[offset];
                Real new_dist = dist[src] + data[offset];
                if(new_dist < dist[dst]) {
                    dist[dst] = new_dist;
                    frontier_out[counter_out] = dst;
                    counter_out++;
                }
            }
        }   
        
        if(counter_out == 0) break;
        
        for(unsigned int i = 0; i < counter_out; i++) {
            Int v = frontier_out[i];
            if(visited[v] == iteration) {
                visited[v] = -1;
            }
            visited[v] = iteration;
        }
        
        Int* tmp     = frontier_in;
        frontier_in  = frontier_out;
        frontier_out = tmp;
        
        counter_in  = counter_out;
        counter_out = 0;
        
        iteration++;
    }
}


int main(int n_args, char** argument_array) {
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
    
    auto t2       = high_resolution_clock::now();
    
    frontier_sssp(frontier_dist, src);

    auto elapsed2 = high_resolution_clock::now() - t2;
    long long ms2 = duration_cast<microseconds>(elapsed2).count();

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
