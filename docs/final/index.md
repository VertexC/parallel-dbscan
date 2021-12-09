
# SUMMARY
We accelerated a common clustring model **dbscan** which different parallelism implementation and data structure, on device (), we achiive: () with CUDA, () with 8 core, () threads with CPU, and () with a hybrid approach, compared to sequential CPU implementation.



# BACKGROUND

Describe the algorithm, application, or system you parallelized in computer science terms. Figures would be really useful here.
• What are the key data structures?
• What are the key operations on these data structures?
• What are the algorithm’s inputs and outputs?
• What is the part that computationally expensive and could benefit from parallelization?
• Break down the workload. Where are the dependencies in the program? How much
parallelism is there? Is it data-parallel? Where is the locality? Is it amenable to
SIMD execution?

# APPROACH & RESULT: Tell us how your implementation works. Your description should be sufficiently
## Dataset
### Validation
### Benchmark

## Target Device

## Related Works

In the course, we have learned potential approaches to make things parallel:
- under shared memory space
+ multi-process(threads)
- under distribued memory space
+ message passing
- utilize data parallel
+ SIMD/CUDA

As DBSCAN is an common used and efficient clustering approach, there are a lot existed research on this, in cluding:
- [FIXME:] add papers that use 

In this project, we implement GDBSCAN(cuda) and (GDB). We also propose a hybrid approach to utlize both cuda and multi-thread to achive better speed up on our target device.
## Overview
As we did in hw3, hw4, we wrote a C++ program which takes command line arguments:
```bash
Usage: ./main [options]
Program Options:
        -f <input_filename> (required)
        -b <backend_method> (default:0) (0:seq, 1:gdbscan, 2:ds-seq, 3:ds-shm 4:hybrid)
        -t <num_threads_omp> (default:1)
        -?  --help             This message
```
and it will load the input data, call the corresponding backend method (record time), and write labels to output file.

### Serial Implementatin (baseline)
We implement baseline as a serial implementation following the pseudocode of DBSCAN on wiki. Basically for each point, we start by gathering its neighbords. And we continually growing that neighborhoods by adding neighbor's neighbor and cluster them with the same label.
```golang
// pseudocode from https://en.wikipedia.org/wiki/DBSCAN
DBSCAN(DB, distFunc, eps, minPts) {
    C := 0                                                  /* Cluster counter */
    for each point P in database DB {
        if label(P) ≠ undefined then continue               /* Previously processed in inner loop */
        Neighbors N := RangeQuery(DB, distFunc, P, eps)     /* Find neighbors */
        if |N| < minPts then {                              /* Density check */
            label(P) := Noise                               /* Label as Noise */
            continue
        }
        C := C + 1                                          /* next cluster label */
        label(P) := C                                       /* Label initial point */
        SeedSet S := N \ {P}                                /* Neighbors to expand */
        for each point Q in S {                             /* Process every seed point Q */
            if label(Q) = Noise then label(Q) := C          /* Change Noise to border point */
            if label(Q) ≠ undefined then continue           /* Previously processed (e.g., border point) */
            label(Q) := C                                   /* Label neighbor */
            Neighbors N := RangeQuery(DB, distFunc, Q, eps) /* Find neighbors */
            if |N| ≥ minPts then {                          /* Density check (if Q is a core point) */
                S := S ∪ N                                  /* Add new neighbors to seed set */
            }
        }
    }
}

RangeQuery(DB, distFunc, Q, eps) {
    Neighbors N := empty list
    for each point P in database DB {                      /* Scan all points in the database */
        if distFunc(Q, P) ≤ eps then {                     /* Compute distance and check epsilon */
            N := N ∪ {P}                                   /* Add to result */
        }
    }
    return N
}
```
### GDBSCAN
We followed the paper G-DBSCAN to explore the cuda based parallelism of DBSCAN. Compared to the sequential implementation we decribed above, it further classify each point with different type as core, border, or noise. 

The idea behind that is to decoupling the neighborhood calculation and labeling into different stages to further utlize parallelism.




### Disjoint Tree

### Disjoint Tree (shared-memory parallelism)

### Hybrid

### GDBSCAN
detailed to provide the course staff a basic understanding of your approach. Again, it might
be very useful to include a figure here illustrating components of the system and/or their
mapping to parallel hardware.
5
• Describe the technologies used. What language/APIs? What machines did you target?
• Describe how you mapped the problem to your target parallel machine(s). IMPORTANT: How do the data structures and operations you described in part 2 map to
machine concepts like cores and threads. (or warps, thread blocks, gangs, etc.)
• Did you change the original serial algorithm to enable better mapping to a parallel
machine?
• If your project involved many iterations of optimization, please describe this process
as well. What did you try that did not work? How did you arrive at your solution? The notes you’ve been writing throughout your project should be helpful here.
Convince us you worked hard to arrive at a good solution.
• If you started with an existing piece of code, please mention it (and where it came
from) here.


If your project was optimizing an algorithm, please define how you measured performance. Is it wall-clock time? Speedup? An application specific rate? (e.g., moves
per second, images/sec)
• Please also describe your experimental setup. What were the size of the inputs? How
were requests generated?
• Provide graphs of speedup or execute time. Please precisely define the configurations
being compared. Is your baseline single-threaded CPU code? It is an optimized
parallel implementation for a single CPU?
• Recall the importance of problem size. Is it important to report results for different
problem sizes for your project? Do different workloads exhibit different execution
behavior?
• IMPORTANT: What limited your speedup? Is it a lack of parallelism? (dependencies) Communication or synchronization overhead? Data transfer (memory-bound or
bus transfer bound). Poor SIMD utilization due to divergence? As you try and answer these questions, we strongly prefer that you provide data and measurements to
support your conclusions. If you are merely speculating, please state this explicitly.
Performing a solid analysis of your implementation is a good way to pick up credit
even if your optimization efforts did not yield the performance you were hoping for.
• Deeper analysis: Can you break execution time of your algorithm into a number
of distinct components. What percentage of time is spent in each region? Where is
there room to improve?
• Was your choice of machine target sound? (If you chose a GPU, would a CPU have
been a better choice? Or vice versa.)

# REFERENCES
- Andrade G, Ramos G, Madeira D, et al. G-dbscan: A gpu accelerated algorithm for density-based clustering[J]. Procedia Computer Science, 2013, 18: 369-378.



