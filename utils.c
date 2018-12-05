#include "utils.h"

static stack_struct objdet_stack = { NULL, NULL, 0 };

void init_stack
    (
    void
    )
{
#ifdef WIN32
    objdet_stack.stack_starting_address = _aligned_malloc(STACK_SIZE_TOTAL, 0x20);
#else
    posix_memalign(&(objdet_stack.stack_starting_address), 0x20, STACK_SIZE_TOTAL);
#endif
    if (objdet_stack.stack_starting_address == NULL)
        {
        printf("failed to alloc whole memory stack\n");
        exit(EXIT_FAILURE);
        }
    objdet_stack.stack_current_address = (char *)objdet_stack.stack_starting_address;
    objdet_stack.stack_current_alloc_size = 0;
}

void free_stack
    (
    void
    )
{
#if WIN32
    _aligned_free(objdet_stack.stack_starting_address);
#else
    free(objdet_stack.stack_starting_address);
#endif
}

void *alloc_from_stack
    (
    unsigned int len
    )
{
    void *ptr = NULL;
    if (len <= 0)
        {
        len = 0x20;
        }
    unsigned int aligned_len = (len + 0xF) & (~0xF);
    objdet_stack.stack_current_alloc_size += aligned_len;
    if (objdet_stack.stack_current_alloc_size >= STACK_SIZE_TOTAL)
        {
        printf("failed to allocate memory from stack anymore\n");
#ifdef WIN32
        _aligned_free(objdet_stack.stack_starting_address);
#else
        free(objdet_stack.stack_starting_address);
#endif
        exit(EXIT_FAILURE);
        }
    ptr = objdet_stack.stack_current_address;
    objdet_stack.stack_current_address += aligned_len;
    // C99: all zero bits means 0 for fixed points, 0.0 for floating points
    memset(ptr, 0, len);
    return ptr;
}

void partial_free_from_stack
    (
    unsigned int len
    )
{
    unsigned int aligned_len = (len + 0xF) & (~0xF);
    objdet_stack.stack_current_alloc_size -= aligned_len;
    objdet_stack.stack_current_address -= aligned_len;
}

unsigned int get_stack_current_alloc_size
    (
    void
    )
{
    return objdet_stack.stack_current_alloc_size;
}

void reset_stack_ptr_to_assigned_position
    (
    unsigned int assigned_size
    )
{
    objdet_stack.stack_current_address = (char *)objdet_stack.stack_starting_address + assigned_size;
    objdet_stack.stack_current_alloc_size = assigned_size;
}

double what_time_is_it_now
    (
    void
    )
{
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    return now.tv_sec + now.tv_nsec*1e-9;
}
