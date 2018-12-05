#ifndef UTILS_H
#define UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h> // for printf
#include <stdlib.h> // for posix_memalign
#include <string.h> // for memset
#include <math.h> // for sqrt
#include <float.h> //for FLT_MAX
#include <time.h> // for clock_gettime
#include <cblas.h>

#define STACK_SIZE_TOTAL 0x8000000 // 128Mbytes

typedef struct
    {
    int index;
    float score;
    }score_index_struct;

typedef struct
    {
    float x;
    float y;
    float w;
    float h;
    }box_struct;

typedef struct
    {
    void *stack_starting_address;
    char *stack_current_address;
    unsigned int stack_current_alloc_size;
    }stack_struct;


void init_stack
    (
    void
    );

void free_stack
    (
    void
    );

void *alloc_from_stack
    (
    unsigned int len
    );

void partial_free_from_stack
    (
    unsigned int len
    );

unsigned int get_stack_current_alloc_size
    (
    void
    );

void reset_stack_ptr_to_assigned_position
    (
    unsigned int assigned_size
    );

double what_time_is_it_now
    (
    void
    );

#ifdef __cplusplus
}
#endif

#endif // UTILS_H
