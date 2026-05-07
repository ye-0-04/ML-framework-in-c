#include <stdlib.h>
#include <stdint.h>

#ifndef tensors_h
#define tensors_h
#endif

typedef struct{
    int rows, cols,dims;
    int32_t *data;
}tensor;
tensor* create_tensor(int rows, int cols , int dims)
{
    if (rows < 1 || cols < 1 || dims < 1) 
    {
        printf("Invalid tensor dimensions\n");
        return 0;
    }
    
    tensor* t = (tensor*)malloc(sizeof(tensor));
    if (t == NULL)
    {
        printf("Memory allocation failed\n");
        
        return NULL;
    }
    t->rows = rows;
    t->cols = cols;
    t->dims = dims;
    
    int size = rows*cols*dims;
    t->data = (int32_t*)malloc(size*sizeof(int32_t));
    if (t->data == NULL)
    {
        printf("Memory allocation failed\n");
        
        return NULL;
    }
    for (int i = 0;i < size;i++)
    {
        t->data[i] = 0;
    }
    return t;

}
tensor* add_tensor(tensor* a,tensor*b)
{
    if (a->rows != b->cols ||b->rows != b-> cols|| a->dims != b->dims)
    {
        printf("Tensors arent equal in size\n");
        return NULL;
    }
    int cols = a->rows;
    int rows = a->cols;
    int dims = a->dims;
    tensor* result = create_tensor(rows,cols,dims);
    for (int i = 0;i < rows;i++)
    {
        for (int j = 0; j < cols; j++)
        {
            for (int k = 0; k < dims; k++)
            {
                int index = i * (cols * dims) + j * dims + k;
                result->data[index] = a->data[index] + b->data[index];
            }
        }
    }
    return result;
}