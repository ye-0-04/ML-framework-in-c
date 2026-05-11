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

    t->stride[dims - 1] = 1;
    for (int i = dims - 2; i >= 0; i--)
    {
        t->stride[i] = shape[i + 1] * t->stride[i + 1];
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
const double get_tensor_val(tensor* a, int index)
{
    if (index >= a->size)
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
const int get_tensor_size(tensor* a)
{
    
    if (!a) return NULL;
    return a->size;

}
const int get_tensor_dims(tensor* a)
{
    if (!a) return NULL;
    return a->dims;
}
const int* get_tensor_shape(tensor* a)
{
    if (!a) return NULL;
    return a->shape;
}
const int* get_tensor_stride(tensor* a)
{
    if(!a) return -1;
    return a->stride;
}

void print_tensor_rec(tensor* a, int dim, int offset)
{
    if (a == NULL)
    {
        printf("NULL pointer\n");
        return;
    }

    // last dimension → print values
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
    printf("\n");
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
tensor* tensor_sub(tensor* a,tensor *b)
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
tensor* tensor_mul(tensor* a,tensor* b)
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
int reshape(tensor* a,int shape[], int dims)
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
    a->stride[dims - 1] = 1;
    for (int i = dims - 2; i >= 0; i--)
    {
        a->stride[i] = a->stride[i+1] * a->shape[i+1];
    }
    return 0;
}

int transpose(tensor* a, int swap_axes[])
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
        printf("dims mismatch\n");
        return NULL;
    }

    int d = a->dims;

    int M = a->shape[d - 2];
    int K = a->shape[d - 1];
    int N = b->shape[d - 1];

    if (b->shape[d - 2] != K)
    {
        printf("inner dims mismatch\n");
        return NULL;
    }

    
    int shape[d];
    for (int i = 0; i < d - 2; i++)
        shape[i] = a->shape[i];

    shape[d - 2] = M;
    shape[d - 1] = N;

    tensor* c = create_tensor(shape, d);
    if (!c) return NULL;

    
    
        

    int batch_size = 1;
    for (int i = 0; i < d - 2; i++)batch_size *= a->shape[i];

    for (int bidx = 0; bidx < batch_size; bidx++)
    {
        int batch_offset_a = bidx * a->stride[d - 3];
        int batch_offset_b = bidx * b->stride[d - 3];
        int batch_offset_c = bidx * c->stride[d - 3];

        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                double sum = 0.0;

                for (int k = 0; k < K; k++)
                {
                    int a_idx =
                        batch_offset_a +
                        i * a->stride[d - 2] +
                        k * a->stride[d - 1];

                    int b_idx =
                        batch_offset_b +
                        k * b->stride[d - 2] +
                        j * b->stride[d - 1];

                    sum += a->data[a_idx] * b->data[b_idx];
                }

                int c_idx =
                    batch_offset_c +
                    i * c->stride[d - 2] +
                    j * c->stride[d - 1];

                c->data[c_idx] = sum;
            }
        }
    }

    return c;
}


double add(double x, double y) {return x + y;}
double sub(double x, double y) {return x - y;}
double mul(double x, double y) {return x * y;}
double div(double x, double y) {return x / y;}
int broadcast_compatible(tensor* a, tensor* b)
{
    if (a->dims != b->dims)
        return 0;

    for (int i = 0; i < a->dims; i++)
    {
        if (a->shape[i] != b->shape[i] && a->shape[i] != 1 && b->shape[i] != 1)
        {
            return 0;
        }
    }

    return 1;
}
tensor* tensor_apply_broadcast(tensor* a, tensor* b, double (*op)(double, double))
{
    if (!broadcast_compatible(a, b))
    {
        printf("not broadcast compatible\n");
        return NULL;
    }

    tensor* c = create_tensor(a->shape, a->dims);

    for (int i = 0; i < c->size; i++)
    {
        int coord = i;

        int a_idx = 0;
        int b_idx = 0;

        int tmp_a = i;
        int tmp_b = i;

        for (int d = a->dims - 1; d >= 0; d--)
        {
            int dim_a = a->shape[d];
            int dim_b = b->shape[d];

            int coord_d = tmp_a % dim_a;

            if (dim_a != 1) a_idx += coord_d * a->stride[d];

            if (dim_b != 1) b_idx += coord_d * b->stride[d];

            tmp_a /= dim_a;
        }

        c->data[i] = op(a->data[a_idx], b->data[b_idx]);
    }

    return c;
}

double reLu(double x)
{
    return (x < 0.0) ? 0.0 : x;
}

double sigmoid(double x)
{
    return (1 / (1 + exp(-x)));
}

void softmax(double x[])
{
    int size = sizeof(x) / sizeof(x[0]);
    double sum = 0.0;
    for (int i = 0; i < size; i++)
    {
        x[i] = exp(x[i]);
        sum += x[i];
    }
    for (int i = 0; i < size; i++)
    {
        x[i] /= sum;
    }
}

double mean_squared_error(double true_val[], double predictions[])
{
    int true_val_size = sizeof(true_val) / sizeof(true_val[0]);
    int predictions_size = sizeof(predictions) / sizeof(predictions[0]);

    double squared_diff = 0.0;
    double sum = 0.0;
    double error = 0.0;
    for (int i = 0; i < true_val_size; i++)
    {
        error = true_val[i] - predictions[i];
        error = error * error;
        sum += error / true_val_size;
    }
    return sum;
}

double entropy(double pred[])
{
    int size = sizeof(pred) / sizeof(pred[0]);
    double sum = 0.0;
    double x = 0;
    for (int i = 0; i < size; i++)
    {
        sum += pred[i] * log(pred[i]);
    }
    if (sum < 0) sum *= -1;
    return (sum < 0) ? (sum * -1) : sum;
}
