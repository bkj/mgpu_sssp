// sssp_cpu.hxx
#pragma once

#include <queue>
#include <vector>
#include <chrono>

using namespace std;
using namespace std::chrono;

template <typename Int, typename Real>
class prioritize {
    public:
        bool operator()(pair<Int, Real> &p1, pair<Int, Real> &p2) {
            return p1.second > p2.second;
        }
};

template <typename Int, typename Real>
long long sssp_cpu(Real* dist, Int n_srcs, Int* srcs, Int n_nodes, Int n_edges, Int* indptr, Int* cindices, Real* data) {
    for(Int i = 0; i < n_nodes; i++) dist[i] = std::numeric_limits<Real>::max();
    
    auto t = high_resolution_clock::now();
    priority_queue<pair<Int,Real>, vector<pair<Int,Real>>, prioritize<Int, Real>> pq;
    
    for(Int src = 0; src < n_srcs; src++) {
        pq.push(make_pair(src, 0));
        dist[src] = 0;
    }
    
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