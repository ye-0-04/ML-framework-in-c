#include <stdio.h>
#include <stdint.h>
#include <time.h>

#include <immintrin.h>

#include "../include/neural.h"
#include "../include/tensors.h"
#include "../include/models.h"

int main()
{
    struct timespec start1, end1;
    struct timespec start2, end2;

    int shape[]=  {1  ,1000, 1000};
    tensor* a = create_tensor(shape, 3);
    tensor* b = create_tensor(shape, 3);
    tensor* c = create_tensor(shape, 3);
    rand_fill(a);
    rand_fill(b);
    clock_gettime(CLOCK_MONOTONIC, &start1);
    tensor_matmul_V2(c, a , b);
    clock_gettime(CLOCK_MONOTONIC, &end1);
    double time1 = (end1.tv_sec - start1.tv_sec) + 
                   (end1.tv_nsec - start1.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start2);
    tensor* d = tensor_matmul(a , b);
    clock_gettime(CLOCK_MONOTONIC, &end2);
    double time2 = (end2.tv_sec - start2.tv_sec) + 
                   (end2.tv_nsec - start2.tv_nsec) / 1e9;

    printf("matmul V2 time: %.6f seconds\n", time1);
    printf("matmul time:    %.6f seconds\n", time2);
    printf("Difference:         %.6f seconds\n", time2 - time1);
    printf("Speedup ratio:       %.2fx\n", time2 / time1);

    return 0;
}