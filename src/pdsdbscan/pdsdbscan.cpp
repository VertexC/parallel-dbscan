#include <cmath>
#include <set>
#include <iostream>

#include "dbscan.h"


#define DEBUG

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
    node_t* a_r = find(a);
    node_t* b_r = find(b);
    if (a_r == b_r) {
        return;
    }
    if (a_r->idx < b_r->idx) {
        a_r->parent = b_r;
        b_r->size += a_r->size;
    } else {
        b_r->parent = a_r;
        a_r->size += b_r->size; 
    }
    return;
}


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