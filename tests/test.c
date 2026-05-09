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

    int shape[] = {2, 3, 1};

    tensor* a = create_tensor(shape, 3);

    if (a == NULL)
    {
        printf("allocation failed\n");
        return 1;
    }

    printf("%f\n", get_tensor_val(a, 0));
    print_tensor_rec(a, 0 , 0);
    return 0;
}
