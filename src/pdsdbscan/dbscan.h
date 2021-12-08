#include <vector>
#include <memory>


using namespace std;

typedef struct {
    float x;
    float y;
} point_t;

#define UNCLASSIFIED -2
#define NOISE -1

void pdsdbscan(const point_t* in, int* out, int num_points, float eps, int minPoints);
void pdsdbscan_omp(const point_t* in, int* out, int num_points, float eps, int minPoints);
