#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#ifndef tensors_h
#define tensors_h
#endif

#include <include/tensors.h>

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
typedef struct tensor{
    int* shape;  //[col, row, etc]
    int* stride; //[col, row , etc]
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
tensor* create_tensor(int shape[], int dims)
{
    int size = 1;
    if (dims <= 0)
    {
        printf("invalid number of dimensions\n");
        return NULL;
    }
    for (int i = 0; i < dims; i++)
    {

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
    
    t->stride = (int*)malloc(dims *sizeof(int));
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
    t->shape = (int*)malloc(dims * sizeof(int));
    if(t->shape == NULL)
    {
        printf("memory allocation failed\n");
        return NULL;
    }
    for (int i = 0; i < dims; i++)
    {
        t->shape[i] = shape[i];
    }
    t->stride[0] = 1;
    for (int i = 1; i < dims; i++)
    {
        t->stride[i] = shape[i-1] * t->stride[i-1];
    }

    printf("tensor created sucessfully\n");
    return t;


}

/**
 * @brief Compares the size and dimensions of two tensors
 *
 * Checks if two tensors have the same number of dimensions and the same
 * size for each dimension.
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

/**
 * @brief Frees all memory associated with a tensor
 *
 * Frees the shape, stride, and data arrays, then frees the tensor structure itself.
 *
 * @param a Tensor to free
 */
void free_tensor(tensor* a)
{
    free(a->shape);
    free(a->stride);
    free(a->data);
}

/**
 * @brief Gets the value at a specific index in the tensor's flat data array
 *
 * @param a Tensor to get value from
 * @param index Position in the flat data array
 * @return Value at the specified index
 */
double get_tensor_val(tensor* a, int index)
{
    if (index > a->size)
    {
        printf("index out of bounds\n");
    }
    return a->data[index];
}

/**
 * @brief Sets the value at a specific index in the tensor's flat data array
 *
 * @param a Tensor to set value in
 * @param val Value to set
 * @param index Position in the flat data array
 */
void set_tensor_val(tensor* a, double val, int index)
{
    if (index > a->size)
    {
        printf("index out of bounds\n");
        return;
    }
    a->data[index] = val;
}

void print_tensor_rec(tensor* a, int dim, int offset)
{
    if (a == NULL)
    {
        printf("NULL pointer\n");
        return;
    }

    if (dim == a->dims - 1)
    {
        printf("[");
        for (int i = 0; i < a->shape[dim]; i++)
        {
            printf("%f", a->data[offset + i * a->stride[dim]]);
            if (i != a->shape[dim] - 1)
                printf(", ");
        }
        printf("]");
        return;
    }

    printf("[");

    for (int i = 0; i < a->shape[dim]; i++)
    {
        print_tensor_rec(
            a,
            dim + 1,
            offset + i * a->stride[dim]
        );

        if (i != a->shape[dim] - 1)
            printf(", ");
    }

    printf("]");
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
tensor* tensor_div(tensor* a , tensor*b)
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
int scal_add(tensor* a , double b)
{
    for (int i = 0; i < a->size; i++)
    {
        a->data[i] = a->data[i] + b;
    }
    return 0;
}

/**
 * @brief Multiplies all elements of a tensor by a scalar value (in-place)
 * 
 * @param a Tensor to modify
 * @param b Scalar value to multiply by
 */
int scal_mul(tensor* a , double b)
{
    for (int i = 0; i < a->size; i++)
    {
        a->data[i] = a->data[i] * b;
    }
    return 0;
}

/**
 * @brief Fills all elements of a tensor with a specified value
 * 
 * @param a Tensor to fill
 * @param b Value to fill with
 */
int fill(tensor* a , double b)
{
    
    for (int i = 0; i < a->size; i++)
    {
        a->data[i] = b;
    }
    return 0;
}

/**
 * @brief Fills all elements of a tensor with random double values
 * 
 * Seeds the random number generator with current time and fills
 * each element with a random double value.
 * 
 * @param a Tensor to fill
 */
int rand_fill(tensor* a)
{
    srand(time(NULL));
    for (int i = 0; i < a->size; i++)
    {
        a->data[i] = (double)rand();
    }
    return 0;
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
int copy(tensor* a, tensor* b)
{
    if (comp_tensor_size(a, b))
    {
        printf("Two tensors aren't equal in size\n");
        return 1;
    }
    for (int i = 0; i < a->size; i++)
    {
        a->data[i] = b->data[i];
    }
    return 0;
    
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
int reshape(tensor* a,int* shape, int dims)
{
    int size = 1;
    for (int i = 0; i < dims; i++)
    {
        if (shape[i] <= 0)
        {
            printf("invalid number in shape\n");
            return 1;
        }
        size = shape[i] * size;
    }
    if (size != a->size)
    {
        printf("tensor cant be reshaped into that size\n");
        return 1;
    }
    a->size = size;
    a->shape = shape;
    a->dims = dims;
    a->stride[0] = 1;
    for (int i = 1; i < dims; i++)
    {
        a->stride[i] = a->stride[i-1] * a->shape[i-1];
    }
    return 0;
}

int transpose(tensor* a, int swap_axes[2])
{
    if (a==NULL)
    {
        printf("tensor doesnt exist\n");
        return 1;
    }
    if (swap_axes[0] >= a->dims || swap_axes[1] >= (int)a->dims)
    {
        printf("cant do operation , check axes numbers\n");
        return 1;
    }
    int x = swap_axes[0];
    int y = swap_axes[1];
    int z = a->shape[x];
    a->shape[x] = a->shape[y];
    a->shape[y] = z;

    z = a->stride[x];
    a->stride[x] = a->stride[y];
    a->stride[y] = z;
    return 0;
}

double tensor_sum(tensor* a)
{
    if (a == NULL)
    {
        printf("NULL tensor\n");
        return 1;
    }
    #pragma omp parallel for reduction(+:sum)
    double sum = 0;
    for (int i = 0; i < a->size; i++)
    {
        if (isnan(a->data[i]) || isinf(a->data[i]))
        {
            continue;
        }
        sum += a->data[i];
    }
    return sum;
}

double tensor_get_max(tensor *a)
{
    if (a == NULL)
    {
        printf("NULL tensor\n");
        return 1;
    }
    #pragma omp parallel for reduction(max:maxval)
    double max = a->data[0];
    for (int i = 0; i < a->size; i++)
    {
        if (isnan(a->data[i]) || isinf(a->data[i]))
        {
            continue;
        }
        if (a->data[i] > max) max = a->data[i];
    }
    return max;
}

double tensor_get_min(tensor* a)
{
    if (a == NULL)
    {
        printf("NULL tensor\n");
        return 1;
    }
    #pragma omp parallel for reduction(min:minval)
    double min = a->data[0];
    for (int i = 0; i < a->size; i++)
    {
        if (isnan(a->data[i]) || isinf(a->data[i]))
        {
            continue;
        }
        if (a->data[i] < min) min = a->data[i];
    }
    return min;
}

double tensor_get_mean(tensor* a)
{
    if (a == NULL)
    {
        printf("NULL tensor\n");
        return 1;
    }
    double sum = tensor_sum(a);
    return (sum / a->size);

}
double tensor_argmax(tensor* a)
{
    if (a == NULL)
    {
        printf("NULL tensor\n");
        return 1;
    }
    #pragma omp parallel for reduction(max:maxval)
    double max = a->data[0];
    int argmax = 0;
    for (int i = 0; i < a->size; i++)
    {
        if (isnan(a->data[i]) || isinf(a->data[i]))
        {
            continue;
        }
        if (a->data[i] > max) argmax = i;
    }
    return argmax;
}
double tensor_argmin(tensor* a)
{
    if (a == NULL)
    {
        printf("NULL tensor\n");
        return 1;
    }
    #pragma omp parallel for reduction(max:maxval)
    double min = a->data[0];
    int argmin = 0;
    for (int i = 0; i < a->size; i++)
    {
        if (isnan(a->data[i]) || isinf(a->data[i]))
        {
            continue;
        }
        if (a->data[i] < min) argmin = i;
    }
    return argmin;
}

tensor* tensor_matmul(tensor* a, tensor* b)
{
    if (a->dims != b->dims)
    {
        printf("tensors arent equal in dims\n");
        return NULL;
    }
    if (a->shape[0] != b->shape[1])
    {
        printf("tensors arent compatible in size,cant perform matmul\n");
        return NULL;
    }
    int shape[a->dims];
    shape[0] = a->shape[0];
    shape[1] = b->shape[1];
    for (int i = 1; i < a->dims; i++)
    {
        if (i == 1) continue;
        if (a->shape[i] != b->shape[i])
        {
            printf("tensors dont have equal batch sizes\n");
            return NULL;
        }
        shape[i] = a->shape[i];
    }
    int size = a->shape[1] * b->shape[0];
    tensor* result = create_tensor(shape, a->dims);
    if (result == NULL)
    {
        printf("Memory allocation failed\n");
    }
    int number_of_batches = 1;
    for (int i = 2; i < a->dims; i++)
    {
        number_of_batches *= a->shape[i];
    }
    

    int tensor_1_jump = a->shape[0] * a->shape[1];
    int tensor_2_jump = b->shape[0] * b->shape[1];
    

    int result_index = 0;

    for (int i = 0; i < number_of_batches; i++)
    {

        for (int row = 0; row < a->shape[1]; row++)  // row a
        {
            for (int col = 0; col < b->shape[0]; col++)  //col b
            {
                for (int k = 0; k < a->shape[0]; k++ ) //col a
                {
                    int a_col_index = i * tensor_1_jump + (((row + 1) * (k + 1)) * a->stride[0]);
                    double a_val = a->data[a_col_index]; 

                    int b_index = i * tensor_2_jump + (((k + 1) * (col + 1)) * b->stride[1]);
                    double b_val = b->data[b_index];
                    result->data[result_index] += a_val * b_val;
                }

                result_index ++;
            }
        }
    }
    return result;
}

