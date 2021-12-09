#include <cmath>
#include <set>
#include <iostream>

#include "dbscan.h"

inline float distance(const point_t& a, const point_t& b) {
    return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y)); 
}

namespace {
    shared_ptr<vector<int>> getNeighbors(const point_t* in, int* out, int idx, int num_points, float eps) {
        auto result = make_shared<vector<int>>();
        for(int i=0; i<num_points; i++) {
            if (i == idx && out[i] != UNCLASSIFIED) continue;
            if (distance(in[idx], in[i]) <= eps) {
                result->push_back(i);
            }
        }
        return result;
    }
}



void serialdbscan(const point_t* in, int* out, int num_points, float eps, int minPoints) {
    bool* visited = (bool *)calloc(num_points, sizeof(bool));
    int cluster_cnt = 0;
    for(int i=0; i<num_points; i++) {
        if(visited[i]) continue;
        visited[i] = true;
        auto neighbors = getNeighbors(in, out, i, num_points, eps);
        if (neighbors->size() < minPoints) {
            out[i] = NOISE;
        } else {
            out[i] = cluster_cnt;
            int total_count = neighbors->size();
            int count = 0;
            for(int j=0; j<neighbors->size(); j++) {
                int p_idx = neighbors->at(j);
                if (!visited[p_idx]) {
                    visited[p_idx] = true;
                    auto new_neighbors = getNeighbors(in, out, p_idx, num_points, eps);
                    if(new_neighbors->size() >= minPoints) {
                        for(auto new_p_idx: *new_neighbors) {
                            neighbors->push_back(new_p_idx);
                            out[new_p_idx] = cluster_cnt;
                        }
                    }
                    total_count = neighbors->size();
                }
                if (out[p_idx] == UNCLASSIFIED || out[p_idx] == NOISE) {
                    out[p_idx] = cluster_cnt;
                }
                count++;
            }
            cluster_cnt++;
            cout << total_count << " " << count << endl;
        }
    
    }
}