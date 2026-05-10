#ifndef tensors_h
#define tensors_h
#endif

#include <stdio.h>
#include <stdint.h>
#include <include/tensors.h>


int main()
{
    printf("test\n");
    fflush(stdout);

    int shape[] = {5, 1, 3 ,2};
    int shape_2[] = {5 , 1 , 2, 3};

    tensor* a = create_tensor(shape, 4);
    tensor* b = create_tensor(shape_2 , 4);

    rand_fill(a);
    rand_fill(b);


    tensor* c = tensor_matmul(a, b);
    
    print_tensor_rec(c, 0, 0);
    
    return 0;
}
