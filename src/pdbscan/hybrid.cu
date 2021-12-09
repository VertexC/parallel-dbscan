#include <cmath>
#include <set>
#include <iostream>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <omp.h>

#include "dbscan.h"

#define BLOCK_X 32
#define BLOCK_Y 32
#define BLOCKSIZE BLOCK_X*BLOCK_Y 
#define SCAN_BLOCK_DIM  BLOCKSIZE // needed by sharedMemExclusiveScan implementation



// For debug
#define DEBUG
#define VERBOSE
#undef DEBUG
#undef VERBOSE

#ifdef DEBUG
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",
            cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#else 
#define cudaCheckError(ans) ans
#endif


namespace {

__host__ __device__ float distance(const float2 a, const float2 b) {
    return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y)); 
}


__global__ void collectDegree(const float2* points, const int num_points, const float eps, int* degree) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    float2 p_a = points[idx];
    int d = 0;
    for (int i=0; i<num_points; i++) {
        if (idx == i) continue;
        float2 p_b = points[i];
        if (distance(p_a, p_b) <= eps) {
            d++;
        }
    }
    degree[idx] = d;
    // printf("%d:%d\n", idx, degree[idx]);
}

__global__ void constructEaKernel(const float2* points, const int* Va_degree, const int* Va_idx, 
  int num_points, const float eps, int* Ea) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    int start_idx = Va_idx[idx];
    float2 p_a = points[idx];
    for (int i=0; i<num_points; i++) {
        if (idx == i) continue;
        float2 p_b = points[i];
        float dis = distance(p_a, p_b);
        if (dis <= eps) {
            Ea[start_idx] = i;
            start_idx++;
            // if (idx == 841) {
            //     printf("nb(871)->%d (%f vs %f) \n", i, dis, eps);
            // }
        }
    }
}

template<typename T1, typename T2>
struct bool2int
{
  __host__ __device__ T2 operator()(const T1 &x) const
  {
    return static_cast<T2>(x);
  }
};

template<typename T1, typename T2>
struct int2int
{
  __host__ __device__ T2 operator()(const T1 &x) const
  {
    return static_cast<T2>(x);
  }
};

shared_ptr<vector<int>> getNeighbors(const int i, const int* Va_idx, const int* Va_degree, const int* Ea) {
    auto result = make_shared<vector<int>>();
    int start_idx = Va_idx[i];
    for(int j=0; j<Va_degree[i]; j++) {
        int nb_idx = Ea[start_idx+j];
        result->push_back(nb_idx);
    }
    return result;
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

} // namespace




void hybrid(const point_t* in, int* out, int num_points, float eps, int minPoints, int numThreads) {
    // convert points data to float2 data
    float2* host_points = (float2 *)calloc(num_points, sizeof(float2));
    for(int i=0; i<num_points; i++) {
        host_points[i].x = in[i].x;
        host_points[i].y = in[i].y;
    }

    // set up cuda memory
    float2* cuda_points;
    int* Va_degree;
    int* Va_idx;

    cudaMalloc((void **)&cuda_points, num_points* sizeof(float2));

    cudaMalloc((void **)&Va_degree, num_points * sizeof(int));
    cudaMalloc((void **)&Va_idx, num_points * sizeof(int));
    cudaMemset(Va_degree, 0, num_points * sizeof(int));
    cudaMemset(Va_idx, 0, num_points * sizeof(int));
    
    cudaMemcpy(cuda_points, host_points, num_points * sizeof(float2), cudaMemcpyHostToDevice);
    // construct graphs
    const int threadsPerBlock = 256;
    const int blocks = (num_points + threadsPerBlock - 1) / threadsPerBlock;
    // 1. compute degree
    collectDegree<<<blocks, threadsPerBlock>>>(cuda_points, num_points, eps, Va_degree);
    // // 2. compute start idx of each vertex with exclusive scan
    int edge_num = thrust::transform_reduce(thrust::device,
                                        Va_degree, Va_degree + num_points,
                                        int2int<int,int>(),
                                        0,
                                        thrust::plus<int>());
    int* Ea;
    cudaMalloc((void **)&Ea, edge_num * sizeof(int));
    cudaMemset(Ea, 0, edge_num * sizeof(int));
    thrust::exclusive_scan(thrust::device, 
                           Va_degree, Va_degree + num_points, 
                           Va_idx, 0);
    // // 3. constructVaKernel
    constructEaKernel<<<blocks, threadsPerBlock>>>(cuda_points, Va_degree, Va_idx, num_points, eps, Ea);

    int* Va_degree_host = (int *)calloc(num_points, sizeof(int));
    int* Va_idx_host = (int *)calloc(num_points, sizeof(int));
    int* Ea_host = (int *)calloc(edge_num, sizeof(int));

    cudaCheckError(cudaMemcpy(Va_degree_host, Va_degree, num_points * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(Va_idx_host, Va_idx, num_points * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(Ea_host, Ea, edge_num * sizeof(int), cudaMemcpyDeviceToHost));

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
            auto neighbors = getNeighbors(i, Va_idx_host, Va_degree_host, Ea_host);
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

    // free resources
    cudaFree(Va_degree);
    cudaFree(Va_idx);
    cudaFree(Ea);
    cudaFree(cuda_points);

    free(Va_degree_host);
    free(Va_idx_host);
    free(Ea_host);
}
