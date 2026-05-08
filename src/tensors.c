#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#ifndef tensors_h
#define tensors_h
#endif

/**
 * @brief Tensor structure representing a multi-dimensional array
 * 
 * Members:
 * - shape: Array storing dimensions (cols, rows, etc.)
 * - stride: Array storing stride values for each dimension for efficient indexing
 * - data: Pointer to the actual data stored as a flat array
 * - dims: Number of dimensions
 * - size: Total number of elements
 */
typedef struct{
    int* shape;
    int* stride;
    double *data;
    int dims;
    int size;
}tensor;

/**
 * @brief Creates a new tensor with the specified shape and dimensions
 * 
 * Allocates memory for the tensor structure, data array, shape array, and stride array.
 * Initializes all data to zero and calculates strides for multi-dimensional indexing.
 * 
 * @param shape Array specifying the size of each dimension
 * @param dims Number of dimensions (must be > 0)
 * @return Pointer to the created tensor, or NULL on failure
 */
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
    
    t->stride = (int*)malloc(3*sizeof(int));
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
    t->stride = (int*)malloc(dims * sizeof(int));
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

/**
 * @brief Compares the size and dimensions of two tensors
 * 
 * @param a First tensor
 * @param b Second tensor
 * @return 0 if tensors have the same size, 1 otherwise
 */
int comp_tensor_size(tensor* a, tensor *b)
{
    if (a->dims != b->dims)
    {
        printf("tensor dimensions aren't equal\n");
        return 1;
    }
    for (int i = 0; i < a->dims; i++)
    {
        if (a->shape[i] != b->shape[i])
        {
            printf("tensors aren't equal in size\n");
            return 1;
        }
    }
    return 0;
}

void free_tensor(tensor* a)
{
    free(a->shape);
    free(a->stride);
    free(a->data);
    free(a->dims);
    free(a->size);
}

void print_tensor(tensor* a , int dims, int offset)
{
    if (dims == a->dims)
    {
        print("%d", a->data[offset]);
        return;
    }
    printf("[");
    for (int i =0; i < a->shape[dims]; i++)
    {
        print_tensor(a, dims + 1, offset + i * a->stride[dims]);

        if (i != a->shape[dims])
        {
            print(",");
        }
    }
    print("]");
    
    
}

/**
 * @brief Performs element-wise addition of two tensors
 * 
 * Both tensors must have the same shape. Creates a new tensor for the result.
 * 
 * @param a First tensor
 * @param b Second tensor
 * @return New tensor with the result, or NULL on failure
 */
tensor* add_tensor(tensor* a,tensor*b)
{
    if (comp_tensor_size(a , b)==1)
    {
        printf("tensors aren't equal in size");
        return NULL;
    }
    
    tensor* result = create_tensor(a->shape, a->dims);
    for (int i = 0; i < a->size; i++ )
    {
        result->data[i] = a->data[i] + b->data[i];
    }
    return result;
}

/**
 * @brief Performs element-wise subtraction of two tensors
 * 
 * Both tensors must have the same shape. Creates a new tensor for the result.
 * 
 * @param a First tensor
 * @param b Second tensor
 * @return New tensor with the result, or NULL on failure
 */
tensor* sub(tensor* a,tensor *b)
{
    if (comp_tensor_size(a , b) == 1)
    {
        printf("tensors aren't equal in size\n");
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

/**
 * @brief Performs element-wise multiplication of two tensors
 * 
 * Both tensors must have the same shape. Creates a new tensor for the result.
 * 
 * @param a First tensor
 * @param b Second tensor
 * @return New tensor with the result, or NULL on failure
 */
tensor* mul(tensor* a,tensor* b)
{
    if (comp_tensor_size(a, b) == 1)
    {
        printf("tensors size aren't equal in size\n");
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

/**
 * @brief Performs element-wise division of two tensors
 * 
 * Both tensors must have the same shape. Creates a new tensor for the result.
 * Note: Does not check for division by zero.
 * 
 * @param a First tensor (numerator)
 * @param b Second tensor (denominator)
 * @return New tensor with the result, or NULL on failure
 */
tensor* div(tensor* a , tensor*b)
{
    if (comp_tensor_size(a, b) == 1)
    {
        printf("tensors size aren't equal in size\n");
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

/**
 * @brief Adds a scalar value to all elements of a tensor (in-place)
 * 
 * @param a Tensor to modify
 * @param b Scalar value to add
 */
void scal_add(tensor* a , double b)
{
    for (int i = 0; i < a->size; i++)
    {
        a->data[i] = a->data[i] + b;
    }
}

/**
 * @brief Multiplies all elements of a tensor by a scalar value (in-place)
 * 
 * @param a Tensor to modify
 * @param b Scalar value to multiply by
 */
void scal_mul(tensor* a , double b)
{
    for (int i = 0; i < a->size; i++)
    {
        a->data[i] = a->data[i] * b;
    }
}

/**
 * @brief Fills all elements of a tensor with a specified value
 * 
 * @param a Tensor to fill
 * @param b Value to fill with
 */
void fill(tensor* a , double b)
{
    
    for (int i = 0; i < a->size; i++)
    {
        a->data[i] = b;
    }
}

/**
 * @brief Fills all elements of a tensor with random double values
 * 
 * Seeds the random number generator with current time and fills
 * each element with a random double value.
 * 
 * @param a Tensor to fill
 */
void rand_fill(tensor* a)
{
    srand(time(NULL));
    for (int i = 0; i < a->size; i++)
    {
        a->data[i] = (double)rand();
    }
}

/**
 * @brief Copies data from one tensor to another
 * 
 * Both tensors must have the same size. The destination tensor
 * is modified in place.
 * 
 * @param a Destination tensor (modified in place)
 * @param b Source tensor
 */
void copy(tensor* a, tensor* b)
{
    if (comp_tensor_size(a, b))
    {
        printf("Two tensors aren't equal in size\n");
        return;
    }
    for (int i = 0; i < a->size; i++)
    {
        a->data[i] = b->data[i];
    }
    
}

/**
 * @brief Reshapes a tensor to new dimensions
 * 
 * Changes the shape of a tensor while keeping the total number of elements the same.
 * The stride array is recalculated for the new shape.
 * 
 * @param a Tensor to reshape
 * @param shape New shape array
 * @param dims New number of dimensions
 */
void reshape(tensor* a,int* shape, int dims)
{
    int size = 1;
    for (int i = 0; i < dims; i++)
    {
        if (shape[i] <= 0)
        {
            printf("invalid number in shape\n");
            return;
        }
        size = shape[i] * size;
    }
    if (size != a->size)
    {
        printf("tensor cant be reshaped into that size\n");
        return;
    }
    a->size = size;
    a->shape = shape;
    a->dims = dims;
    a->stride[0] = 1;
    for (int i = 1; i < dims; i++)
    {
        a->stride[i] = a->stride[i-1] * a->shape[i-1];
    }
}

void transpose(tensor* a, int* shape ,int dims)
{
    if (a==NULL)
    {
        printf("tensor doesnt exist\n");
        return NULL;
    }
    int size = 0;
    for (int i = 0; i < dims; i++)
    {
        size += shape[i];
    }
    
    

}