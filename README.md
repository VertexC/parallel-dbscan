


# cuda: gdbscan
1. multiple kernel version
2. single kernel, shared memory optimization
# 


bowenc@bowenc-MS-7A59 ~/d/c/p/s/pdbscan> nvprof  ./main -f ../../data/benchmark/twitter_100000.txt  -b 1 -t 10
filename: ../../data/benchmark/twitter_100000.txt, method 1, num_threads 0
==6883== NVPROF is profiling process 6883, command: ./main -f ../../data/benchmark/twitter_100000.txt -b 1 -t 10
graph construction time: 0.618310.
bfs scan time: 2.515870.
Computation Time: 3.134545.
basename: ../../data/benchmark/twitter_100000
==6883== Profiling application: ./main -f ../../data/benchmark/twitter_100000.txt -b 1 -t 10
==6883== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   60.99%  867.01ms     25660  33.788us  4.1930us  331.51us  bfsKernel(bool*, bool*, int const *, int const *, int const *, int*, int, int)
                   11.99%  170.45ms         1  170.45ms  170.45ms  170.45ms  collectDegree(float2 const *, int, float, int*)
                   11.62%  165.14ms         1  165.14ms  165.14ms  165.14ms  constructEaKernel(float2 const *, int const *, int const *, int, float, int*)
                    4.49%  63.858ms     28731  2.2220us  2.1120us  4.8000us  void cub::DeviceReduceKernel<cub::DeviceReducePolicy<int, int, int, thrust::plus<int>>::Policy600, thrust::cuda_cub::transform_input_iterator_t<int, bool*, bool2int<bool, int>>, int*, int, thrust::plus<int>>(int, int, int, cub::GridEvenShare<int>, thrust::plus<int>)
                    4.24%  60.326ms      6143  9.8200us  8.4480us  92.069us  [CUDA memcpy HtoD]
                    3.56%  50.541ms     28732  1.7590us  1.6310us  12.224us  void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<int, int, int, thrust::plus<int>>::Policy600, int*, int*, int, thrust::plus<int>, int>(int, int, int, thrust::plus<int>, cub::DeviceReducePolicy<int, int, int, thrust::plus<int>>::Policy600)
                    3.11%  44.222ms     31804  1.3900us     480ns  31.010us  [CUDA memcpy DtoH]
                    0.00%  71.012us         4  17.753us  3.2330us  60.419us  [CUDA memset]
                    0.00%  11.457us         1  11.457us  11.457us  11.457us  void cub::DeviceScanKernel<cub::AgentScanPolicy<int=128, int=15, int, cub::BlockLoadAlgorithm, cub::CacheLoadModifier, cub::BlockStoreAlgorithm, cub::BlockScanAlgorithm, cub::MemBoundScaling<int=128, int=15, int>>, int*, int*, cub::ScanTileState<int, bool=1>, thrust::plus<void>, int, int>(int=15, int, cub::BlockLoadAlgorithm, int, cub::CacheLoadModifier, cub::BlockStoreAlgorithm, cub::BlockScanAlgorithm)
                    0.00%  6.8170us         1  6.8170us  6.8170us  6.8170us  classifyKernel(int, int, int*, int*)
                    0.00%  5.5360us         1  5.5360us  5.5360us  5.5360us  void cub::DeviceReduceKernel<cub::DeviceReducePolicy<int, int, int, thrust::plus<int>>::Policy600, thrust::cuda_cub::transform_input_iterator_t<int, int*, int2int<int, int>>, int*, int, thrust::plus<int>>(int, int, int, cub::GridEvenShare<int>, thrust::plus<int>)
                    0.00%  2.0160us         1  2.0160us  2.0160us  2.0160us  void cub::DeviceScanInitKernel<cub::ScanTileState<int, bool=1>>(int, int)
      API calls:   26.31%  685.12ms     57465  11.922us     739ns  1.2219ms  cudaStreamSynchronize
                   15.06%  392.22ms     34880  11.244us  1.9880us  122.75ms  cudaMalloc
                   13.48%  351.09ms      9215  38.099us  11.173us  165.35ms  cudaMemcpy
                   11.00%  286.39ms     83129  3.4450us  2.3250us  305.57us  cudaLaunchKernel
                   10.76%  280.36ms     28732  9.7570us  6.3660us  2.0654ms  cudaMemcpyAsync
                    9.97%  259.56ms     34880  7.4410us  2.0460us  5.9036ms  cudaFree
                    6.55%  170.52ms         1  170.52ms  170.52ms  170.52ms  cudaThreadSynchronize
                    3.41%  88.732ms    747060     118ns      92ns  482.26us  cudaGetLastError
                    1.19%  30.941ms    114933     269ns     190ns  293.89us  cudaGetDevice
                    1.09%  28.481ms     57465     495ns     366ns  39.176us  cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
                    0.62%  16.220ms    114932     141ns      93ns  296.56us  cudaPeekAtLastError
                    0.56%  14.489ms     57465     252ns     189ns  297.89us  cudaDeviceGetAttribute
                    0.01%  237.12us       101  2.3470us     134ns  157.38us  cuDeviceGetAttribute
                    0.00%  78.313us         1  78.313us  78.313us  78.313us  cuDeviceTotalMem
                    0.00%  34.261us         4  8.5650us  2.9150us  16.220us  cudaMemset
                    0.00%  22.485us         1  22.485us  22.485us  22.485us  cuDeviceGetName
                    0.00%  6.8700us         1  6.8700us  6.8700us  6.8700us  cuDeviceGetPCIBusId
                    0.00%  5.0230us         1  5.0230us  5.0230us  5.0230us  cudaFuncGetAttributes
                    0.00%  1.0790us         3     359ns     172ns     710ns  cuDeviceGetCount
                    0.00%     612ns         2     306ns     130ns     482ns  cuDeviceGet
                    0.00%     249ns         1     249ns     249ns     249ns  cuDeviceGetUuid
                    0.00%     193ns         1     193ns     193ns     193ns  cudaGetDeviceCount