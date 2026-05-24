#include <stdio.h>
#include <stdint.h>
#include <time.h>

#include <immintrin.h>

#include "../include/neural.h"
#include "../include/tensors.h"
#include "../include/models.h"

tensor* add_tensor_2(tensor* a, tensor* b, bool con);
 // Assuming this exists

int main()
{
    printf("hello\n");
    int shape[] = {1, 10000, 10000};
    
    // First test: add_tensor_2
    tensor* a1 = create_tensor(shape, 3);
    tensor* b1 = create_tensor(shape, 3);
    
    struct timespec start1, end1;
    clock_gettime(CLOCK_MONOTONIC, &start1);
    
    tensor* c1 = add_tensor_2(a1, b1, true);
    
    clock_gettime(CLOCK_MONOTONIC, &end1);
    double time1 = (end1.tv_sec - start1.tv_sec) + 
                   (end1.tv_nsec - start1.tv_nsec) / 1e9;
    
    free_tensor(c1);

    
    // Second test: add_tensor
    tensor* a2 = create_tensor(shape, 3);
    tensor* b2 = create_tensor(shape, 3);
    
    struct timespec start2, end2;
    clock_gettime(CLOCK_MONOTONIC, &start2);
    
    tensor* c2 = add_tensor(a2, b2, true);
    
    clock_gettime(CLOCK_MONOTONIC, &end2);
    double time2 = (end2.tv_sec - start2.tv_sec) + 
                   (end2.tv_nsec - start2.tv_nsec) / 1e9;
    
    free_tensor(c2);

    
    // Print results
    printf("add_tensor_2 time: %.6f seconds\n", time1);
    printf("add_tensor time:    %.6f seconds\n", time2);
    printf("Difference:         %.6f seconds\n", time1 - time2);
    printf("Speedup ratio:       %.2fx\n", time2 / time1);
    
    return 0;
}
tensor* add_tensor_2(tensor* a,tensor* b, bool con )
{
    if (comp_tensor_size(a , b)==1)
    {
        printf("tensors aren't equal in size");
        
    }
    
    tensor* result = create_tensor(a->shape, a->dims);
    for (int i = 0; i <= a->size; i++)
    {
 
        result->data[i] = a->data[i] + b->data[i];
    }
    if (con) 
    {
        free_tensor(a);
        free_tensor(b);
    }
    return result;

}
