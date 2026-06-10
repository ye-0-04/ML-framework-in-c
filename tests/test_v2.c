#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>

#include <immintrin.h>

#include "../include/neural.h"
#include "../include/tensors.h"
#include "../include/models.h"

int main()
{
    struct timespec start1, end1;

    int hhh = 501;
    int shape[]=  {1  ,hhh, hhh};
    tensor* a = create_tensor(shape, 3);
    tensor* b = create_tensor(shape, 3);
    tensor* c = create_tensor(shape, 3);
    rand_fill(a);
    rand_fill(b);
    // clock_gettime(CLOCK_MONOTONIC, &start1);
    // tensor_matmul_V2(c, a , b);
    // clock_gettime(CLOCK_MONOTONIC, &end1);
    // double time1 = (end1.tv_sec - start1.tv_sec) + 
    //                (end1.tv_nsec - start1.tv_nsec) / 1e9;

    // clock_gettime(CLOCK_MONOTONIC, &start1);
    // tensor* d = tensor_matmul(a , b);
    // clock_gettime(CLOCK_MONOTONIC, &end1);
    // double time2 = (end1.tv_sec - start1.tv_sec) + 
    //                (end1.tv_nsec - start1.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start1);
    tensor_matmul_V2_simd_packed(c, a , b);
    clock_gettime(CLOCK_MONOTONIC, &end1);
    double time3 = (end1.tv_sec - start1.tv_sec) + 
                   (end1.tv_nsec - start1.tv_nsec) / 1e9;

                   
    clock_gettime(CLOCK_MONOTONIC, &start1);
    tensor_matmul_V2_simd_packed_v2(c, a, b);
    clock_gettime(CLOCK_MONOTONIC, &end1);
    double time4 = (end1.tv_sec - start1.tv_sec) + 
                   (end1.tv_nsec - start1.tv_nsec) / 1e9;

    // printf("matmul V2 time:                 %.6f seconds\n", time1);
    // printf("matmul time:                    %.6f seconds\n", time2);
    printf("tensor_matmul_V2_simd_packed    %.6f seconds\n", time3);
    printf("tensor_matmul_V2_simd_packed_v2 %.6f seconds\n", time4);
    printf("Difference:         %.6f seconds\n", time4 - time3);
    // printf("Speedup ratio:       %.2fx\n", time1 / time2);
    // printf("Speedup ratio2:       %.2fx\n", time1 / time3);

    return 0;
}