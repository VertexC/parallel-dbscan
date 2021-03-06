#include <cmath>
#include <set>
#include <iostream>
#include <atomic>
#include <omp.h>

#include "dbscan.h"


#define DEBUG
#undef DEBUG

inline float distance(const point_t& a, const point_t& b) {
    return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y)); 
}

struct node;
typedef struct node {
    int idx;
    node* parent;
    int size;
    int cluster;
} node_t;

node_t* find(node_t* a) {
    node_t* ptr = a;
    while (ptr->parent != ptr) {
        ptr = ptr->parent;
    }
    return ptr;
}

void unionOp(node_t* a, node_t* b) {
    node_t* x = a;
    node_t* y = b;
    while(x->parent != y->parent) {
        if(x->parent < y->parent) {
            if(x == x->parent) { 
                x->parent = y->parent;
                node_t* y_root = find(y);
                y_root->size += y->size; 
            }
            x = x->parent;
        } else {
            if(y == y->parent) { 
                y->parent = x->parent;
                node_t* x_root = find(x);
                x_root->size += x->size; 
            }
            y = y->parent;
        }
    }
    return;
}

namespace {
   shared_ptr<vector<int>> getNeighbors(const point_t* in, int* out, int idx, int num_points, float eps) {
        auto result = make_shared<vector<int>>();
        for(int i=0; i<num_points; i++) {
            if (i == idx) continue;
            if (distance(in[idx], in[i]) <= eps) {
                result->push_back(i);
            }
        }
        return result;
    }
}

void pdsdbscan(const point_t* in, int* out, int num_points, float eps, int minPoints) {
    // TODO: free the memory
    vector<node_t*> nodes;
    // convert points to nodes
    for(int i=0; i<num_points; i++) {
        node_t* node = (node_t*)malloc(sizeof(node_t));
        node->idx = i;
        node->parent = node;
        node->size = 1;
        node->cluster = -1;
        nodes.push_back(node);
    }
    int type[num_points]; // 0: border, 1: core
    bool clustered[num_points];
    for(int i=0; i<num_points; i++) {
        type[i] = 0;
        clustered[i] = false;
    }
    // union pionts
    for(int i=0; i<num_points; i++) {
        auto neighbors = getNeighbors(in, out, i, num_points, eps);
        if (neighbors->size() < minPoints) {
            continue;
        }
        type[i] = 1; // mark as core
        for(auto& n_id:*neighbors) {
            if(type[n_id]== 1) {
                unionOp(nodes[i], nodes[n_id]);
            } else {
                if(!clustered[n_id]) {
                    clustered[n_id] = true;
                    unionOp(nodes[i], nodes[n_id]);
                }
            }
        }
    }
#ifdef DEBUG
for(int i=0; i<num_points; i++) {
    node_t* node = nodes[i];
    node_t* root = find(node);
    printf("node %d -> %d\n", node->idx, root->idx);
}

#endif
    // labeling
    int label = 0;
    // mark root node
    for(int i=0; i<num_points; i++) {
        node_t* node = nodes[i];
        if(node->parent == node) {
            if (node->size > 1) {
                node->cluster = label++;
            } else {
                node->cluster = NOISE;
            }
            out[i] = node->cluster;
        }
    }
    for(int i=0; i<num_points; i++) {
        node_t* node = nodes[i];
        if(node->parent != node) {
            node_t* root = find(node);
            node->cluster = root->cluster;
            out[i] = node->cluster;
        }
    }
}


void unionOpWithLock(node_t* a, node_t* b, omp_lock_t* locks) {
    node_t* x = a;
    node_t* y = b;
    while(x->parent != y->parent) {
        if(x->parent < y->parent) {
            if (x == x->parent) {
                // printf("try to acquire lock at %d\n", x->idx);
                omp_set_lock(&locks[x->idx]);
                if(x == x->parent) { 
                    x->parent = y->parent;
                    node_t* y_root = find(y);
                    y_root->size += x->size;
                }
                omp_unset_lock(&locks[x->idx]);
                // printf("release lock at %d\n", x->idx);
            }
            x = x->parent;
        } else {
            if (y == y->parent) {
                // printf("try to acquire lock at %d\n", y->idx);
                omp_set_lock(&locks[y->idx]);
                if(y == y->parent) { 
                    y->parent = x->parent;
                    node_t* x_root = find(x);
                    x_root->size += x->size;
                }
                omp_unset_lock(&locks[y->idx]);
                // printf("release lock at %d\n", y->idx);
            }
            y = y->parent;
        }
    }
    return;
}

void pdsdbscan_omp(const point_t* in, int* out, int num_points, float eps, int minPoints, int numThreads) {
    vector<node_t*> nodes(num_points);
    // convert points to nodes
    #pragma omp parallel for
    for(int i=0; i<num_points; i++) {
        node_t* node = (node_t*)malloc(sizeof(node_t));
        node->idx = i;
        node->parent = node;
        node->size = 1;
        node->cluster = -1;
        nodes[i] = node;
    }
    int type[num_points]; // 0: border, 1: core
    std::atomic_flag clustered[num_points];
    for(int i=0; i<num_points; i++) {
        type[i] = 0;
        clustered[i].clear();
    }
    
    // int numThreads = 2;
    int block_size = (num_points + numThreads - 1) / numThreads;
    vector<vector<pair<int, int>>> crossThreadUnionSet(numThreads);
    omp_lock_t locks[num_points];
    for(int i=0; i<num_points; i++) {
        omp_init_lock(&locks[i]);
    }
    
    auto ckp1 = Clock::now();
    vector<double> gn_time(numThreads);
    int tid;
    for(tid=0; tid<numThreads; tid++) {
        gn_time[tid] = 0;
    }
    omp_set_num_threads(numThreads);
    #pragma omp parallel for default(shared) private(tid) schedule(dynamic)
    for(tid=0; tid<numThreads; tid++) {
        double wtime = omp_get_wtime();
        int start_idx = tid*block_size;
        int end_idx = (tid+1)*block_size;
        printf("using %d threads", omp_get_thread_num());
        for (int i=start_idx; i<end_idx; i++) {
            if (i >= num_points) continue;
            auto gn_ckp = Clock::now();
            auto neighbors = getNeighbors(in, out, i, num_points, eps);
            if (neighbors->size() < minPoints) {
                continue;
            }
            gn_time[tid] += duration_cast<dsec>(Clock::now() - gn_ckp).count();
            type[i] = 1; // mark as core
            for(auto& n_id:*neighbors) {
                if (n_id >= start_idx && n_id < end_idx) {
                    if(type[n_id]== 1) {
                        unionOp(nodes[i], nodes[n_id]);
                    } else {
                        if(!clustered[n_id].test_and_set()) {
                            unionOp(nodes[i], nodes[n_id]);
                        }
                    }
                } else {
                    pair<int, int> p = {i, n_id};
                    crossThreadUnionSet[tid].push_back(p);
                }
            }
        }
        printf("Time taken by thread %d is %lf.\n", omp_get_thread_num(), omp_get_wtime() - wtime);
    }
    auto ckp2 = Clock::now();
    for(tid = 0; tid < numThreads; tid++) {
        printf("tid %d getNeighbors: %lf.\n", tid, gn_time[tid]);
    }
    printf("local compute time: %lf.\n", duration_cast<dsec>(ckp2 - ckp1).count());
    #pragma omp parallel for
    for(int tid=0; tid<numThreads; tid++) {
        auto unionSet = crossThreadUnionSet[tid];
        for(auto p:unionSet) {
            int x = p.first;
            int y = p.second;
            // printf("%d: %d, %d\n", tid, x, y);
            if(type[y]== 1) {
                unionOpWithLock(nodes[x], nodes[y], locks);
            } else {
                if(!clustered[y].test_and_set()) {
                    unionOpWithLock(nodes[x], nodes[y], locks);
                }
            }
        }
    }
    auto ckp3 = Clock::now();
    printf("merge time: %lf.\n", duration_cast<dsec>(ckp3 - ckp2).count());
#ifdef DEBUG
for(int i=0; i<num_points; i++) {
    node_t* node = nodes[i];
    node_t* root = find(node);
    printf("node %d -> %d\n", node->idx, root->idx);
}
#endif
    // labeling
    int label = 0;
    // mark root node
    #pragma omp parallel for
    for(int i=0; i<num_points; i++) {
        node_t* node = nodes[i];
        if(node->parent == node) {
            if (node->size > 1) {
                node->cluster = label++;
            } else {
                node->cluster = NOISE;
            }
            out[i] = node->cluster;
        }
    }
    #pragma omp parallel for
    for(int i=0; i<num_points; i++) {
        node_t* node = nodes[i];
        if(node->parent != node) {
            node_t* root = find(node);
            node->cluster = root->cluster;
            out[i] = node->cluster;
        }
    }
}