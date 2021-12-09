#include <vector>
#include <memory>
#include <assert.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>


using namespace std;
using namespace std::chrono;
typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double> dsec;

typedef struct {
    float x;
    float y;
} point_t;

#define UNCLASSIFIED -2
#define NOISE -1

void serialdbscan(const point_t* in, int* out, int num_points, float eps, int minPoints);
void pdsdbscan(const point_t* in, int* out, int num_points, float eps, int minPoints);
void pdsdbscan_omp(const point_t* in, int* out, int num_points, float eps, int minPoints, int num_threads);
void gdbscan(const point_t* in, int* out, int num_points, float eps, int minPoints);
void hybrid(const point_t* in, int* out, int num_points, float eps, int minPoints, int num_threads);