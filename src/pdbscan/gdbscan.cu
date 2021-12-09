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


static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__host__ __device__ float distance(const float2 a, const float2 b) {
    return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y)); 
}

// float host_distance(const float2 a, const float2 b) {
//     return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y)); 
// }

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


__global__ void validateVaKernel(const int num_points, const int* Va_degree, const int* Va_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    int pre_sum = 0;
    for (int i=0; i<idx; i++) {
        pre_sum += Va_degree[i];
    }
    if (pre_sum != Va_idx[idx]) {
        printf("degree, idx doesn't match at %d with %d(degree), %d vs %d\n", idx, Va_degree[idx], pre_sum, Va_idx[idx]);
    }
}

__global__ void validateEaKernel(const float2* points, const int num_points, const int* Va_degree, const int* Va_idx, const int* Ea, const float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    float2 p_a = points[idx];
    int start_idx = Va_idx[idx];
    for (int i=0; i<Va_degree[idx]; i++) {
        int nb_idx = start_idx+i;
        float2 p_b = points[Ea[nb_idx]];
        if (distance(p_a, p_b) > eps) {
            printf("ValidateEaKernel failed\n");
        }
    }
}

__global__ void bfsKernel(bool* Fa, bool* Xa, const int* Va_degree,  const int* Va_idx, const int* Ea, int* type, int num_points, int minPoints) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    if (!Fa[idx]) return;
    Fa[idx] = 0;
    Xa[idx] = 1;
    if (Va_degree[idx] < minPoints) return;
    int base_idx = Va_idx[idx];
    for(int i=0; i< Va_degree[idx]; i++) {
        if (Va_degree[i] <= minPoints) continue;
        // if (i+base_idx > num_points) {
        //     printf("get %d base %d\n", i, base_idx);
        // } 
        int nb_idx = Ea[i+base_idx];
        if (!Xa[nb_idx]) {
            Fa[nb_idx] = 1;
        }
    }
}

// __global__ void anyOneKernel(bool* arr, int length, int* result) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     printf("In anyOne idx %d\n", idx);
//     if (idx == 0) {
//         *result = false;
//         for(int i=0; i<length; i++) {
//             printf("got %d\n", arr[i]);
//             *result = (*result) || arr[i]; 
//         }
//     }
// }

__global__ void classifyKernel(int num_points, int minPoints, int* Va_degree, int* type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    if (Va_degree[idx] >= minPoints) {
        type[idx] = 1; // core
    } else {
        type[idx] = 2; // noise
    }
    // printf("classifyKernel %d:%d %d\n", idx, Va_degree[idx], type[idx]);
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


void bfs(int idx, int label, int num_points, bool* visited, int* labels, int* type, int* Va_degree, int* Va_idx, int* Ea, int minPoints) {
    // printf("start bfs on %d", idx);
    // graph related data structure
    bool* Fa_host = (bool*)calloc(num_points, sizeof(bool));
    bool* Xa_host = (bool*)calloc(num_points, sizeof(bool));
    Fa_host[idx] = true;
    bool* Xa_cu;
    bool* Fa_cu;
    cudaCheckError(cudaMalloc((void **)&Fa_cu, num_points * sizeof(bool)));
    cudaCheckError(cudaMalloc((void **)&Xa_cu, num_points * sizeof(bool)));
    cudaCheckError(cudaMemcpy(Fa_cu, Fa_host, num_points * sizeof(bool), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(Xa_cu, Xa_host, num_points * sizeof(bool), cudaMemcpyHostToDevice));

    const int threadsPerBlock = 256;
    const int blocks = (num_points + threadsPerBlock - 1) / threadsPerBlock;
    // int flag = false;
    while (true) {
        // flag = false;
        // std::cout << "before anyone" << std::endl;
        // anyOneKernel<<<1, 1>>>(Fa_cu, num_points, &flag);
        int fa_sum = thrust::transform_reduce(thrust::device,
                                        Fa_cu, Fa_cu + num_points,
                                        bool2int<bool,int>(),
                                        0,
                                        thrust::plus<int>());
        if (fa_sum == 0) {
            break;
        }
        // printf("idx %d: fa_sum %d\n", idx, fa_sum);
        bfsKernel<<<blocks, threadsPerBlock>>>(Fa_cu, Xa_cu, Va_degree, Va_idx, Ea, type, num_points, minPoints);
        
        cudaCheckError(cudaThreadSynchronize());
    }
    cudaCheckError(cudaThreadSynchronize());
    cudaCheckError(cudaMemcpy(Xa_host, Xa_cu, num_points * sizeof(bool), cudaMemcpyDeviceToHost));
    for (int i=0; i<num_points; i++) {
        if(Xa_host[i]) {
            labels[i] = label;
            visited[i] = true;
            if (type[i] != 1 /*core*/) {
                type[i] = 0; /*border*/
            }
        }
    }
    cudaFree(Xa_cu);
    cudaFree(Fa_cu);
    free(Fa_host);
    free(Xa_host);
}


void cluster(int num_points, bool* visited, int* labels, int* type, int* Va_degree, 
  int* Va_idx, int* Ea, int minPionts) {
    int label = 0;
    for(int i=0; i<num_points; i++) {
        if (!visited[i] && (type[i] == 1)) {
            visited[i]= true;
            labels[i] = label;
            bfs(i, label, num_points, visited, labels, type, Va_degree, Va_idx, Ea, minPionts);
            label++;
        }
    }
}




void gdbscan(const point_t* in, int* out, int num_points, float eps, int minPoints) {
    std::cout << eps << " " << minPoints << std::endl;   
    // convert points data to float2 data
    float2* host_points = (float2 *)calloc(num_points, sizeof(float2));
    for(int i=0; i<num_points; i++) {
        host_points[i].x = in[i].x;
        host_points[i].y = in[i].y;
    }
    // graph related data structure
    bool* visited = (bool *)calloc(num_points, sizeof(bool));
    int* type = (int *)calloc(num_points, sizeof(int)); // 0 as , 1 as core, 2 as border


    // set up cuda memory
    float2* cuda_points;
    int* Va_degree;
    // int* Va_degree_temp;
    int* Va_idx;
    
    int* type_cu;

    cudaMalloc((void **)&cuda_points, num_points* sizeof(float2));

    // int rounded_length = nextPow2(num_points); // round up to nextPow2 for exclusive scan
    cudaMalloc((void **)&Va_degree, num_points * sizeof(int));
    // cudaMalloc((void **)&Va_degree_temp, num_points * sizeof(int));
    cudaMalloc((void **)&Va_idx, num_points * sizeof(int));
    
    cudaMalloc((void **)&type_cu, num_points * sizeof(int));
    cudaMemset(type_cu, 0, num_points * sizeof(int));
    cudaMemset(Va_degree, 0, num_points * sizeof(int));
    cudaMemset(Va_idx, 0, num_points * sizeof(int));
    
    cudaMemcpy(cuda_points, host_points, num_points * sizeof(float2), cudaMemcpyHostToDevice);
    // construct graphs
    const int threadsPerBlock = 256;
    const int blocks = (num_points + threadsPerBlock - 1) / threadsPerBlock;
    // 1. compute degree
    collectDegree<<<blocks, threadsPerBlock>>>(cuda_points, num_points, eps, Va_degree);
    cudaCheckError(cudaThreadSynchronize());
    // cudaMemcpy(Va_degree_temp, Va_degree, num_points * sizeof(int), cudaMemcpyDeviceToDevice);
    // cudaMemcpy(Va_degree, Va_idx, num_points * sizeof(int), cudaMemcpyDeviceToDevice);
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
    cudaCheckError(cudaThreadSynchronize());
    validateVaKernel<<<blocks, threadsPerBlock>>>(num_points, Va_degree, Va_idx);
    // // 3. constructVaKernel
    constructEaKernel<<<blocks, threadsPerBlock>>>(cuda_points, Va_degree, Va_idx, num_points, eps, Ea);
    cudaCheckError(cudaThreadSynchronize());
    validateEaKernel<<<blocks, threadsPerBlock>>>(cuda_points, num_points, Va_degree, Va_idx, Ea, eps);
    cudaCheckError(cudaThreadSynchronize());
    // // 4. classify type
    classifyKernel<<<blocks, threadsPerBlock>>>(num_points, minPoints, Va_degree, type_cu);
    cudaCheckError(cudaMemcpy(type, type_cu, num_points * sizeof(int), cudaMemcpyDeviceToHost));

    // check graph status
#ifdef VERBOSE
    int* Va_degree_host = (int *)calloc(num_points, sizeof(int));
    int* Va_idx_host = (int *)calloc(num_points, sizeof(int));
    int* Ea_host = (int *)calloc(edge_num, sizeof(int));

    cudaCheckError(cudaMemcpy(Va_degree_host, Va_degree, num_points * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(Va_idx_host, Va_idx, num_points * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(Ea_host, Ea, edge_num * sizeof(int), cudaMemcpyDeviceToHost));
    printf("================VERBOSE==============\n");
    for(int i=0; i<num_points; i++) {
        printf("%d (type: %d, degree: %d):", i, type[i], Va_degree_host[i]);

        int start_idx = Va_idx_host[i];
        for(int j=0; j<Va_degree_host[i]; j++) {
            int nb_idx = start_idx + j;
            printf(" %d (%f)", Ea_host[nb_idx], distance(host_points[Ea_host[nb_idx]], host_points[i]));
        }
        printf("\n");
    }
    printf("======================================\n");
#endif
    // bfs scan
    cluster(num_points, visited, out, type, Va_degree, Va_idx, Ea, minPoints);

    // free resources
    cudaFree(Va_degree);
    cudaFree(Va_idx);
    cudaFree(Ea);
    cudaFree(cuda_points);
    cudaFree(type_cu);

    free(visited);
    free(type);
}
