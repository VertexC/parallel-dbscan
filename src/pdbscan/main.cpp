#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <cstring>
#include <iostream>
#include <vector>
#include <memory>

#include <assert.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "dbscan.h"


static int _argc;
static const char **_argv;


void show_help(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("\t-f <input_filename> (required)\n");
    printf("\t-b <backend_method> (default:0) (0:seq, 1:gdbscan, 2:ds-seq, 3:ds-shm)\n");
    printf("\t-t <num_threads_omp> (default:1)\n");
    printf("\t-?  --help             This message\n");
}

const char *get_option_string(const char *option_name,
                              const char *default_value) {
  for (int i = _argc - 2; i >= 0; i -= 2)
    if (strcmp(_argv[i], option_name) == 0)
      return _argv[i + 1];
  return default_value;
}

using namespace std;
using namespace std::chrono;
typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double> dsec;

int main(int argc, const char *argv[])
{
    _argc = argc - 1;
    _argv = argv + 1;
    const char *input_filename = get_option_string("-f", NULL);
    const char *backend_method = get_option_string("-b", "0");
    int method = atoi(backend_method);
    int num_threads = 0;
    if (method == 3) {
        const char *num_thread_str = get_option_string("-t", "1");
        num_threads = atoi(num_thread_str);
        if(num_threads <= 0) {
            printf("Error: Threads shoud larger than 1.\n");
            return 1;
        }
    }
    printf("filename: %s, method %d, num_threads %d\n", input_filename, method, num_threads);
    FILE *input = fopen(input_filename, "r");
    int error = 0;

    if (input_filename == NULL) {
        printf("Error: You need to specify -f.\n");
        error = 1;
    }

    if (error) {
        show_help(argv[0]);
        return 1;
    }

    int num_of_points;
    int min_points;
    float eps;
    fscanf(input, "%d\n", &num_of_points);
    fscanf(input, "%f %d\n", &eps, &min_points);

    point_t* points = (point_t *)calloc(num_of_points, sizeof(point_t));
    int* cluster = (int *)calloc(num_of_points, sizeof(int));
    /* Read the grid dimension and wire information from file */
    int i;
    for (i=0; i<num_of_points; i++) {
        fscanf(input, "%f %f\n", &(points[i].x), &(points[i].y));
        cluster[i] = UNCLASSIFIED;
    }
    fclose(input);

    auto compute_start = Clock::now();
    double compute_time = 0;
    char* method_str = "serial";
    switch (method)
    {
    case 0:
        serialdbscan(points, cluster, num_of_points, eps, min_points);
        method_str = "serial";
        break;
    case 1:
        gdbscan(points, cluster, num_of_points, eps, min_points);
        method_str = "gdbscan";
        break;
    case 2:
        pdsdbscan(points, cluster, num_of_points, eps, min_points);
        method_str = "ds-seq";
        break;
    case 3:
        pdsdbscan_omp(points, cluster, num_of_points, eps, min_points, num_threads);
        method_str = "ds-shm";
        break;
    default:
        break;
    }
    compute_time += duration_cast<dsec>(Clock::now() - compute_start).count();
    printf("Computation Time: %lf.\n", compute_time);

    // write wire
    char *out_file_name = (char*)malloc(100 * sizeof(char));

    // const char* filename = basename(input_filename);
    const char* filename = input_filename;
    char* base_name = (char*)malloc(sizeof(char) * (strlen(filename)-4));
    memcpy(base_name, filename, sizeof(char) * (strlen(filename)-4));
    base_name[strlen(filename)-4] = '\0';
    printf("basename: %s\n", base_name);
    sprintf(out_file_name, "%s_%s.txt", base_name, method_str);
    FILE *out = fopen(out_file_name, "w");
    if (out == NULL) {
        printf("Unable to open file: %s.\n", out_file_name);
        return 1;
    }
    fprintf(out, "%d\n", num_of_points);
    for(int i=0;i<num_of_points;i++) {
        fprintf(out, "%d\n", cluster[i]);
    }
    fclose(out);

    free(points);
    free(cluster);
    return 0;
}
