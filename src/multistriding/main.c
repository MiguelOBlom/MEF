#include "util.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <errno.h>
#include <sys/mman.h>
#include <string.h>
#include <time.h>

#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)
#define HUGEPAGE_SIZE (1UL << 30)  // 1GB hugepage size

#ifndef N
extern const size_t N;
#endif

void check_data_initialize(float * D, char * input_filename) {   
    FILE *fptr;

    #ifdef testN
    size_t N_expected = testN;
    #else
    size_t N_expected = N;
    #endif

    if ((fptr = fopen(input_filename, "r")) == NULL){
        printf("Cannot open file %s.", input_filename);
        exit(1);
    }

    fseek(fptr, 0, SEEK_END);
    size_t num_elements = ftell(fptr) / sizeof(float);
    fseek(fptr, 0, SEEK_SET);

    if (num_elements != N_expected) {
        printf("Wrong number of values in input testfile %s: expected %lu, got %lu.\n", input_filename, N_expected, num_elements);
        fclose(fptr); 
        exit(1);
    }

    size_t read_elements = fread(D, sizeof(float), num_elements, fptr);

    if (read_elements != num_elements) {
        perror("Something went wrong reading file.");
        fclose(fptr); 
        exit(1);
    }

    fclose(fptr); 
}

int check_data_compare(float * D, char * output_filename) {   
    int valid = 1;
    FILE *fptr;

    #ifdef testN
    size_t N_expected = testN;
    #else
    size_t N_expected = N;
    #endif

    if ((fptr = fopen(output_filename, "r")) == NULL){
        printf("Cannot open file %s.", output_filename);
        exit(1);
    }

    fseek(fptr, 0, SEEK_END);
    size_t num_elements = ftell(fptr) / sizeof(float);
    fseek(fptr, 0, SEEK_SET);

    if (num_elements != N_expected) {
        printf("Wrong number of values in output testfile %s: expected %lu, got %lu.\n", output_filename, N_expected, num_elements);
        fclose(fptr); 
        exit(1);
    }

    float f;
    for (size_t i = 0; i < N_expected; ++i) {
        if (fread(&f, sizeof(float), 1, fptr) != 1) {
            perror("Error reading file");
            fclose(fptr);
            exit(1);
        } 

        // Avoid signed zeroes
        if (f < 0.00001 && f > -0.00001) {
            f = 0;
        }

        if (D[i] < 0.00001 && D[i] > -0.00001) {
            D[i] = 0;
        }

        // Margin
        float abs_f = (f > 0)? f : -f;

        // Large and small numbers must fall in specified margins
        if ((D[i] + 0.01 * abs_f < f || D[i] - 0.01 * abs_f > f) && (D[i] + 1 < f || D[i] - 1 > f)) {
            valid = 0;
            printf("%lu Expected %f is not equal to actual %f.\n", i, f, D[i]);
        }
    }

    fclose(fptr);
    return valid;
}

int main (int argc, char ** argv) {
    #ifdef TIME
    struct timespec start_clock, end_clock;
    double start;
    double end;
    double cpu_time_used = 0;
    #endif

    size_t data_size = N * sizeof(float);
    size_t aligned_size = ((data_size + HUGEPAGE_SIZE - 1) / HUGEPAGE_SIZE) * HUGEPAGE_SIZE;
    #ifdef MMAP_FLAG_HUGE
    float *D = (float*) mmap(NULL, aligned_size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB | MAP_HUGE_1GB, -1, 0);
    #else
    float *D = (float*) mmap(NULL, aligned_size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    #endif

    if (D == MAP_FAILED) {
        perror("mmap failed");
        exit(-1);
    }

    #ifdef TESTING
    if (argc > 2) {
        printf("Initializing data...\n");
        check_data_initialize(D, argv[1]);
        printf("Done initializing data!\n");
    } else {
        printf("Expected input file and output file for check.\n");
        exit(-2);
    }
    #else
    (void) argc;
    (void) argv;

    // Initialize
    for (unsigned int i = 0; i < aligned_size / sizeof(float); ++i) {
        D[i] = i;
    }

    // Warmup
    for (unsigned int i = 0; i < WARMUP; ++i){
        experiment(D);
    }
    #endif

    for (unsigned int i = 0; i < REPETITIONS; ++i) {
        #ifdef TIME
        clock_gettime(CLOCK_MONOTONIC_RAW, &start_clock);
        #endif
        experiment(D);
        #ifdef TIME
        clock_gettime(CLOCK_MONOTONIC_RAW, &end_clock);
        start = start_clock.tv_sec + start_clock.tv_nsec / 1e9;
        end = end_clock.tv_sec + end_clock.tv_nsec / 1e9;
        cpu_time_used += ((double) (end - start));
        #endif
    }

    int retval = 0;

    #ifdef TESTING
    printf("Comparing result...\n");
    if (check_data_compare(D, argv[2])) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
        retval = -3;
    }
    #endif

    if (munmap(D, aligned_size) == -1) {
        perror("munmap failed\n");
        exit(-1);
    }
    
    #ifdef TIME
    printf("%f\n", ((N * sizeof(float) * REPETITIONS)/ cpu_time_used) / 1073741824 );
    #endif

    return retval;
}
