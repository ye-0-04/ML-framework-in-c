#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#ifndef tensors_h
#define tensors_h
#endif

typedef struct{
    int* shape; //cols,  rows , etc
    int* stride; //cols , rows , etc
    double *data;
    int dims;
    int size;
}tensor;
tensor* create_tensor(int* shape, int dims)
{
    int size = 1;
    if (dims <= 0)
    {
        printf("invalid number of dimensions\n");
        return NULL;
    }
    for (int i = 0; i < dims; i++)
    {
        if (shape[i] == NULL)
        {
            printf("shape doesnt match dimensions\n");
            return NULL;
        }
        if (shape[i] <= 0)
        {
            printf("shape contain an invalid number for a dimension\n");
            return NULL;
        }
        size = shape[i] * size;
    }
    
    tensor* t = (tensor*)malloc(sizeof(tensor));
    if (t == NULL)
    {
        printf("Memory allocation failed\n");
        
        return NULL;
    }
    t->size = size;
    
    t->dims = dims;
    
    t->stride = (int)malloc(3*sizeof(int));
    if (t->stride == NULL)
    {
        printf("stride memory allocation failed\n");
        return NULL;
    }

    
    
    t->data = (double*)malloc(size*sizeof(double));
    if (t->data == NULL)
    {
        printf("Memory allocation failed\n");
        
        return NULL;
    }
    fill(t , 0.0);
    t->stride = (int)malloc(sizeof(shape));
    if (t->stride == NULL)
    {
        printf("stride memory allocation failed\n");
        return NULL;
    }
    t->stride[0] = 1;
    for (int i = 1; i < dims; i++)
    {
        t->stride[i] = shape[i-1] * t->stride[i-1];
    }


    return t;


}
int comp_tensor_size(tensor* a, tensor *b)
{
    if (a->dims != b->dims)
    {
        printf("tensor dimensions arent equal\n");
        return 1;
    }
    for (int i = 0; i < a->dims; i++)
    {
        if (a->shape[i] != b->shape[i])
        {
            printf("tensors arent equal in size\n");
            return 1;
        }
    }
    return 0;
}
tensor* add_tensor(tensor* a,tensor*b)
{
    if (comp_tensor_size(a , b)==1)
    {
        printf("tensors arent equal in size");
        return NULL;
    }
    
    tensor* result = create_tensor(a->shape, a->dims);
    for (int i = 0; i < a->size; i++ )
    {
        result->data[i] = a->data[i] + b->data[i];
    }
    return result;
}
tensor* sub(tensor* a,tensor *b)
{
    if (comp_tensor_size(a , b) == 1)
    {
        printf("tensors arent equal in size\n");
        return NULL;
    }
    tensor* result = create_tensor(a->shape, a->dims);
    if (result == NULL)
    {
        printf("memory allocation failed\n");
        return NULL;
    }
    for (int i = 0; i < a->size; i++)
    {
        result->data[i] = a->data[i] - b->data[i];
    }
    return result;
}
tensor* mul(tensor* a,tensor* b)
{
    if (comp_tensor_size(a, b) == 1)
    {
        printf("tensors size arent equal in size\n");
        return NULL;
    }

    tensor* result = create_tensor(a->shape, a->dims);
    if (result== NULL)
    {
        printf("memory allocation failed\n");
        return NULL;
    }
    for (int i = 0; i < a->size; i++)
    {
        result->data[i] = a->data[i] * b->data[i];
    }
    return result;
}

tensor* div(tensor* a , tensor*b)
{
    if (comp_tensor_size(a, b) == 1)
    {
        printf("tensors size arent equal in size\n");
        return NULL;
    }

    tensor* result = create_tensor(a->shape, a->dims);
    if (result== NULL)
    {
        printf("memory allocation failed\n");
        return NULL;
    }
    for (int i = 0; i < a->size; i++)
    {
        result->data[i] = a->data[i] / b->data[i];
    }
    return result;
}

void scal_add(tensor* a , double b)
{
    for (int i = 0; i < a->size; i++)
    {
        a->data[i] = a->data[i] + b;
    }
}

void scal_mul(tensor* a , double b)
{
    for (int i = 0; i < a->size; i++)
    {
        a->data[i] = a->data[i] + b;
    }
}

void fill(tensor* a , double b)
{
    
    for (int i = 0; i < a->size; i++)
    {
        a->data[i] = b;
    }
}
void rand_fill(tensor* a)
{
    srand(time(NULL));
    for (int i = 0; i < a->size; i++)
    {
        a->data[i] = (double)rand();
    }
}

void copy(tensor* a, tensor* b)
{
    if (comp_tensor_sizes(a, b))
    {
        printf("Two tensors arent equal in size\n");
        return NULL;
    }
    for (int i = 0; i < a->size; i++)
    {
        a->data[i] = b->data[i];
    }
    
}

void reshape(tensor* a,int* shape, int dims)
{
    int size = 1;
    for (int i = 0; i < dims; i++)
    {
        if (shape[i] <= 0)
        {
            printf("invalid number in shape\n");
            return NULL;
        }
        int size = shape[i] * size;
    }
    if (size != a->size)
    {
        printf("tensor cant be reshaped into that size\n");
        return NULL;
    }
    a->size = size;
    a->shape = shape;
    a->dims = dims;
    a->stride[0] = 1;
    for (int i = 1; i < a->size; i++)
    {
        a->stride = a->stride[i-1] * a->shape[i-1];
    }
}