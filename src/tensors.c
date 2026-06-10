#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <immintrin.h>
#include <malloc.h>

#ifndef tensors_h
#define tensors_h


#include "../include/tensors.h"

#define BS 32
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
    if (dims <= 0) {
        printf("invalid number of dimensions\n");
        return NULL;
    }
    
    int size = 1;
    for (int i = 0; i < dims; i++) {
        if (shape[i] <= 0) {
            printf("shape contain an invalid number for a dimension\n");
            return NULL;
        }
        size *= shape[i];
    }
    
    tensor* t = (tensor*)malloc(sizeof(tensor));
    if (!t) return NULL;
    t->size = size;
    t->dims = dims;
    
    t->stride = (int*)malloc(dims * sizeof(int));
    if (!t->stride) { free(t); return NULL; }
    
    t->data = (float*)malloc(size * sizeof(float));
    if (!t->data) { free(t->stride); free(t); return NULL; }
    
    t->shape = (int*)malloc(dims * sizeof(int));
    if (!t->shape) { free(t->data); free(t->stride); free(t); return NULL; }
    
    for (int i = 0; i < dims; i++) {
        t->shape[i] = shape[i];
    }
    
    // compute strides
    t->stride[dims - 1] = 1;
    for (int i = dims - 2; i >= 0; i--) {
        t->stride[i] = shape[i + 1] * t->stride[i + 1];
    }
    
    fill(t, 0.0);
    // printf("tensor created successfully (size=%d)\n", size);
    return t;
}

tensor* add_tensor(tensor* a,tensor* b, bool con )
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


tensor* tensor_sub(tensor* a,tensor *b)
{
    if (comp_tensor_size(a , b) == 1)
    {
        printf("tensors aren't equal in size\n");
        
    }
    tensor* result = create_tensor(a->shape, a->dims);
    if (&result == NULL)
    {
        printf("memory allocation failed\n");
        
    }
    for (int i = 0; i < a->size; i++)
    {
        result->data[i] = a->data[i] - b->data[i];
    }
    return result;
}

tensor* tensor_mul(tensor* a,tensor* b)
{
    if (comp_tensor_size(a, b) == 1)
    {
        printf("tensors size aren't equal in size\n");
        
    }

    tensor* result = create_tensor(a->shape, a->dims);
    if (&result== NULL)
    {
        printf("memory allocation failed\n");
        
    }
    for (int i = 0; i < a->size; i++)
    {
        result->data[i] = a->data[i] * b->data[i];
    }
    return result;
}

tensor* tensor_div(tensor* a , tensor*b)
{
    if (comp_tensor_size(a, b) == 1)
    {
        printf("tensors size aren't equal in size\n");
        
    }

    tensor* result = create_tensor(a->shape, a->dims);
    if (&result== NULL)
    {
        printf("memory allocation failed\n");
        
    }
    for (int i = 0; i < a->size; i++)
    {
        result->data[i] = a->data[i] / b->data[i];
    }
    return result;
}

tensor* tensor_matmul(tensor* a, tensor* b)
{
    if (a->dims != b->dims)
    {
        printf("dims mismatch\n");
        
    }

    int d = a->dims;

    int M = a->shape[d - 2];
    int K = a->shape[d - 1];
    int N = b->shape[d - 1];

    if (b->shape[d - 2] != K)
    {
        printf("inner dims mismatch\n");
        
    }

    
    int shape[d];
    for (int i = 0; i < d - 2; i++)
        shape[i] = a->shape[i];

    shape[d - 2] = M;
    shape[d - 1] = N;

    tensor* c = create_tensor(shape, d);
    

    
    
        

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
                float sum = 0.0;

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

tensor* tensor_apply_broadcast(tensor* a, tensor* b, float (*op)(float, float))
{
    if (!broadcast_compatible(a, b))
    {
        printf("not broadcast compatible\n");
        
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

tensor load_tensor(FILE *f)
{
    tensor t;

    fread(&t.dims, sizeof(int), 1, f);
    fread(&t.size, sizeof(int), 1, f);

    t.shape = malloc(sizeof(int) * t.dims);
    t.stride = malloc(sizeof(int) * t.dims);
    t.data = malloc(sizeof(float) * t.size);


    fread(t.shape, sizeof(int), t.dims, f);
    fread(t.stride, sizeof(int), t.dims, f);
    fread(t.data, sizeof(float), t.size, f);

    return t;
}

float tensor_sum(tensor* a)
{
    if (a == NULL)
    {
        printf("NULL tensor\n");
        return 1;
    }
    
    float sum = 0;
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

float tensor_get_max(tensor *a)
{
    if (a == NULL)
    {
        printf("NULL tensor\n");
        return 1;
    }
    
    float max = a->data[0];
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

float tensor_get_min(tensor* a)
{
    if (a == NULL)
    {
        printf("NULL tensor\n");
        return 1;
    }
    
    float min = a->data[0];
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

float tensor_get_mean(tensor* a)
{
    if (a == NULL)
    {
        printf("NULL tensor\n");
        return 1;
    }
    float sum = tensor_sum(a);
    return (sum / a->size);

}

float mean_squared_error(float true_val[],int true_val_size, float predictions[], int predictions_size)
{


    float squared_diff = 0.0;
    float sum = 0.0;
    float error = 0.0;
    for (int i = 0; i < true_val_size; i++)
    {
        error = true_val[i] - predictions[i];
        error = error * error;
        sum += error / true_val_size;
    }
    return sum;
}

float entropy(float pred[],int size)
{
    
    float sum = 0.0;
    float x = 0;
    for (int i = 0; i < size; i++)
    {
        sum += pred[i] * log(pred[i]);
    }
    if (sum < 0) sum *= -1;
    return (sum < 0) ? (sum * -1) : sum;
}

const float get_tensor_val(tensor* a, int index)
{
    if (index >= a->size)
    {
        printf("index out of bounds\n");
    }
    return a->data[index];
}

float add(float x, float y) {return x + y;}

float sub(float x, float y) {return x - y;}

float mul(float x, float y) {return x * y;}

float div_2(float x, float y) {return x / y;}

int tensor_argmax(tensor* a)
{
    if (a == NULL)
    {
        printf("NULL tensor\n");
        return 1;
    }
    
    float max = a->data[0];
    int argmax = 0;
    for (int i = 0; i < a->size; i++)
    {
        if (isnan(a->data[i]) || isinf(a->data[i]))
        {
            continue;
        }
        if (a->data[i] > max) 
        {
            argmax = i;
            max = a->data[i];

        }
    }
    return argmax;
}

int tensor_argmin(tensor* a)
{
    if (a == NULL)
    {
        printf("NULL tensor\n");
        return 1;
    }
    
    float min = a->data[0];
    int argmin = 0;
    for (int i = 0; i < a->size; i++)
    {
        if (isnan(a->data[i]) || isinf(a->data[i]))
        {
            continue;
        }
        if (a->data[i] < min)
        {
            argmin = i;
            min = a->data[i];

        }
    }
    return argmin;
}
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

int tensor_argmax_index(tensor *t)
{
    int idx = 0;
    float max = t->data[0];

    for (int i = 1; i < t->size; i++)
    {
        if (t->data[i] > max)
        {
            max = t->data[i];
            idx = i;
        }
    }

    return idx;
}

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

const int get_tensor_size(tensor* a)
{
    
    
    return a->size;

}

const int get_tensor_dims(tensor* a)
{
    
    return a->dims;
}

const int* get_tensor_shape(tensor* a)
{
    
    return a->shape;
}

const int* get_tensor_stride(tensor* a)
{
    
    return a->stride;
}


void init_empty_tensor(tensor* a)
{
    a->shape = NULL;
    a->data = NULL;
    a->stride = NULL;
    a->dims = 0;
    a->size = 0;
}

void free_tensor(tensor* a)
{
    free(a->shape);
    free(a->stride);
    free(a->data);
    free(a);
}

void set_tensor_val(tensor* a, float val, int index)
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

void scal_tensor_add(tensor* a, tensor* b)
{
    if (comp_tensor_size(a , b) == 1)
    {
        printf("tensors aren't equal in size\n");
        return ;
    }
    for (int i = 0; i < a->size; i++)
    {
        a->data[i] = a->data[i] + b->data[i];
    }
}

void scal_tensor_sub(tensor* a, tensor* b)
{
    if (comp_tensor_size(a , b) == 1)
    {
        printf("tensors aren't equal in size\n");
        return ;
    }
    for (int i = 0; i < a->size; i++)
    {
        a->data[i] = a->data[i] - b->data[i];
    }
}

void scal_tensor_mul(tensor* a, tensor* b)
{
    if (comp_tensor_size(a , b) == 1)
    {
        printf("tensors aren't equal in size\n");
        return ;
    }
    for (int i = 0; i < a->size; i++)
    {
        a->data[i] = a->data[i] * b->data[i];
    }
}

void scal_pow(tensor*a, int b)
{
    for (int i = 0; i < a->size; i++)
    {
        a->data[i] = pow( a->data[i] , b);
    }
}

 void scal_add(tensor* a , float b)
{
    for (int i = 0; i < a->size; i++)
    {
        a->data[i] = a->data[i] + b;
    }

}

void scal_mul(tensor* a , float b)
{
    for (int i = 0; i < a->size; i++)
    {
        a->data[i] = a->data[i] * b;
    }
    
}

void fill(tensor* a , float b)
{
    
    for (int i = 0; i < a->size; i++)
    {
        a->data[i] = b;
    }
    
}

void rand_fill(tensor* a)
{
    
    for (int i = 0; i < a->size; i++)
    {
        a->data[i] = ((float)rand() / RAND_MAX - 0.5) * 0.01;
    }
    
}

void transpose(tensor* a, int swap_axes[])
{
    if (a==NULL)
    {
        printf("tensor doesnt exist\n");
        
    }
    if (swap_axes[0] >= a->dims || swap_axes[1] >= (int)a->dims)
    {
        printf("cant do operation , check axes numbers\n");
        
    }
    int x = swap_axes[0];
    int y = swap_axes[1];
    int z = a->shape[x];
    a->shape[x] = a->shape[y];
    a->shape[y] = z;

    z = a->stride[x];
    a->stride[x] = a->stride[y];
    a->stride[y] = z;
    
}

void reLu(tensor* x)
{
    for (int i = 0;i < x->size;i++)
    {
        x->data[i] = (x->data[i] < 0.0) ? 0.0 : x->data[i];
    }
    
}

void sigmoid(tensor* x)
{
    
    for (int i = 0;i < x->size;i++)
    {
        x->data[i] = (1 / (1 + exp(x->data[i])));
    }
}

void softmax(tensor *x)
{
    int size = x->size;

    float max = x->data[0];
    for (int i = 1; i < size; i++)
    {
        if (x->data[i] > max)
            max = x->data[i];
    }

    float sum = 0.0;
    for (int i = 0; i < size; i++)
    {
        x->data[i] = exp(x->data[i] - max);
        sum += x->data[i];
    }

    for (int i = 0; i < size; i++)
    {
        x->data[i] /= sum;
    }
}

void clone(tensor* a, tensor* b)
{
    if (a->shape) free(a->shape);
    if (a->stride) free(a->stride);
    if (a->data) free(a->data);
    
    a->dims = b->dims;
    a->size = b->size;
    
    a->shape = (int*)malloc(a->dims * sizeof(int));
    a->stride = (int*)malloc(a->dims * sizeof(int));
    a->data = (float*)malloc(a->size * sizeof(float));
    
    for (int i = 0; i < a->dims; i++) {
        a->shape[i] = b->shape[i];
        a->stride[i] = b->stride[i];
    }
    
    for (int i = 0; i < a->size; i++) {
        a->data[i] = b->data[i];
    }
}

void print_tensor_values(tensor* a)
{
    printf("[");
    for (int i = 0; i < a->size; i++)
    {

        printf("%f,", get_tensor_val(a, i));
    }
    printf("]\n");
}


// Version 2 of tensors operations designed for optimizarions
// Less memory allocations inside the function
// Kernal fusion


void tensor_matmul_V2(tensor* output, tensor* a, tensor* b)
{
    // ... validation ...
    
    int d = a->dims;
    int M = a->shape[d-2], K = a->shape[d-1], N = b->shape[d-1];
    
    int batch_size = 1;
    for (int i = 0; i < d-2; i++) batch_size *= a->shape[i];
    
    // Initialize output to zero
    for (int i = 0; i < output->size; i++) output->data[i] = 0.0f;
    
    
    for (int bidx = 0; bidx < batch_size; bidx++)
    {
        int batch_offset_a = bidx * a->stride[d-3];
        int batch_offset_b = bidx * b->stride[d-3];
        int batch_offset_c = bidx * output->stride[d-3];
        
        // Now IKJ order within each batch
        for (int i = 0; i < M; i++)
        {
            for (int k = 0; k < K; k++)
            {
                int a_idx = batch_offset_a + i * a->stride[d-2] + k * a->stride[d-1];
                float a_val = a->data[a_idx];
                
                for (int j = 0; j < N; j++)
                {
                    int b_idx = batch_offset_b + k * b->stride[d-2] + j * b->stride[d-1];
                    int c_idx = batch_offset_c + i * output->stride[d-2] + j * output->stride[d-1];
                    
                    output->data[c_idx] += a_val * b->data[b_idx];
                }
            }
        }
    }
}


void tensor_matmul_V2_simd_packed(tensor* output, tensor* a, tensor* b)
{
    // Validation
    if (a->dims != b->dims)
    {
        printf("dims mismatch\n");
        return;
    }

    int d = a->dims;
    int M = a->shape[d-2];
    int K = a->shape[d-1];
    int N = b->shape[d-1];

    if (b->shape[d-2] != K)
    {
        printf("inner dims mismatch\n");
        return;
    }

    // Calculate batch size
    int batch_size = 1;
    for (int i = 0; i < d-2; i++)
        batch_size *= a->shape[i];

    // Initialize output to zero
    for (int i = 0; i < output->size; i++)
        output->data[i] = 0.0f;

    // SIMD parameters
    const int MR = 6;
    const int NR = 8;
    const int KR = 128;

    // Allocate packed buffer
    float* packed_b = (float*)_aligned_malloc(K * N * sizeof(float), 32);
    if (!packed_b) {
        printf("_aligned_malloc failed\n");
        return;
    }
    
    // Main batch loop
    for (int bidx = 0; bidx < batch_size; bidx++)
    {
        int batch_offset_a = bidx * a->stride[d-3];
        int batch_offset_b = bidx * b->stride[d-3];
        int batch_offset_c = bidx * output->stride[d-3];
        
        for (int k = 0; k < K; k += KR)
        {
            int k_end = (k + KR < K) ? k + KR : K;
            int k_size = k_end - k;
            
            for (int j = 0; j < N; j += NR)
            {
                int j_end = (j + NR < N) ? j + NR : N;
                int j_size = j_end - j;
                
                // Pack B block
                for (int kk = 0; kk < k_size; kk++)
                {
                    for (int jj = 0; jj < j_size; jj++)
                    {
                        int b_idx = batch_offset_b + (k+kk) * b->stride[d-2] + (j+jj) * b->stride[d-1];
                        packed_b[kk * j_size + jj] = b->data[b_idx];
                    }
                }
                
                // Main computation
                for (int i = 0; i < M; i += MR)
                {
                    int i_end = (i + MR < M) ? i + MR : M;
                    int i_size = i_end - i;
                    
                    __m256 acc[6];
                    for (int ii = 0; ii < i_size; ii++)
                        acc[ii] = _mm256_setzero_ps();
                    
                    for (int kk = 0; kk < k_size; kk++)
                    {
                        for (int ii = 0; ii < i_size; ii++)
                        {
                            int a_idx = batch_offset_a + (i+ii) * a->stride[d-2] + (k+kk) * a->stride[d-1];
                            __m256 a_broad = _mm256_broadcast_ss(&a->data[a_idx]);
                            
                            __m256 b_vec = _mm256_load_ps(&packed_b[kk * j_size]);
                            
                            acc[ii] = _mm256_fmadd_ps(a_broad, b_vec, acc[ii]);
                        }
                    }
                    
                    // Store results
                    for (int ii = 0; ii < i_size; ii++)
                    {
                        int c_idx = batch_offset_c + (i+ii) * output->stride[d-2] + j * output->stride[d-1];
                        _mm256_store_ps(&output->data[c_idx], acc[ii]);
                    }
                }
            }
        }
    }
    
    _aligned_free(packed_b);
}
void tensor_matmul_V2_simd_packed_v2(tensor* output, tensor* a, tensor* b)
{
    // Validation
    if (a->dims != b->dims)
    {
        printf("dims mismatch\n");
        return;
    }

    int d = a->dims;
    int M = a->shape[d-2];
    int K = a->shape[d-1];
    int N = b->shape[d-1];

    if (b->shape[d-2] != K)
    {
        printf("inner dims mismatch\n");
        return;
    }

    // Calculate batch size
    int batch_size = 1;
    for (int i = 0; i < d-2; i++)
        batch_size *= a->shape[i];

    // Initialize output to zero
    for (int i = 0; i < output->size; i++)
        output->data[i] = 0.0f;

    // SIMD parameters
    const int MR = 6;
    const int NR = 8;
    const int KR = 128;

    // Get number of threads
    int num_threads = omp_get_max_threads();
    
    // Allocate buffer for each thread
    float** thread_buffers = (float**)malloc(num_threads * sizeof(float*));
    if (!thread_buffers) {
        printf("Failed to allocate thread buffer array\n");
        return;
    }
    
    for (int t = 0; t < num_threads; t++) {
        thread_buffers[t] = (float*)_aligned_malloc(K * N * sizeof(float), 32);
        if (!thread_buffers[t]) {
            printf("Failed to allocate buffer for thread %d\n", t);
            // Cleanup already allocated buffers
            for (int tt = 0; tt < t; tt++)
                _aligned_free(thread_buffers[tt]);
            free(thread_buffers);
            return;
        }
    }
    
    // Main batch loop with OpenMP
    #pragma omp parallel for
    for (int bidx = 0; bidx < batch_size; bidx++)
    {
        int tid = omp_get_thread_num();
        float* packed_b = thread_buffers[tid];
        
        int batch_offset_a = bidx * a->stride[d-3];
        int batch_offset_b = bidx * b->stride[d-3];
        int batch_offset_c = bidx * output->stride[d-3];
        
        for (int k = 0; k < K; k += KR)
        {
            int k_end = (k + KR < K) ? k + KR : K;
            int k_size = k_end - k;
            
            for (int j = 0; j < N; j += NR)
            {
                int j_end = (j + NR < N) ? j + NR : N;
                int j_size = j_end - j;
                
                // Pack B block into thread-local buffer
                for (int kk = 0; kk < k_size; kk++)
                {
                    for (int jj = 0; jj < j_size; jj++)
                    {
                        int b_idx = batch_offset_b + (k+kk) * b->stride[d-2] + (j+jj) * b->stride[d-1];
                        packed_b[kk * j_size + jj] = b->data[b_idx];
                    }
                }
                
                // Main computation with SIMD
                for (int i = 0; i < M; i += MR)
                {
                    int i_end = (i + MR < M) ? i + MR : M;
                    int i_size = i_end - i;
                    
                    __m256 acc[6];
                    for (int ii = 0; ii < i_size; ii++)
                        acc[ii] = _mm256_setzero_ps();
                    
                    for (int kk = 0; kk < k_size; kk++)
                    {
                        for (int ii = 0; ii < i_size; ii++)
                        {
                            int a_idx = batch_offset_a + (i+ii) * a->stride[d-2] + (k+kk) * a->stride[d-1];
                            __m256 a_broad = _mm256_broadcast_ss(&a->data[a_idx]);
                            __m256 b_vec = _mm256_load_ps(&packed_b[kk * j_size]);
                            acc[ii] = _mm256_fmadd_ps(a_broad, b_vec, acc[ii]);
                        }
                    }
                    
                    // Store results
                    for (int ii = 0; ii < i_size; ii++)
                    {
                        int c_idx = batch_offset_c + (i+ii) * output->stride[d-2] + j * output->stride[d-1];
                        _mm256_store_ps(&output->data[c_idx], acc[ii]);
                    }
                }
            }
        }
    }
    
    // Free all thread buffers
    for (int t = 0; t < num_threads; t++)
        _aligned_free(thread_buffers[t]);
    free(thread_buffers);
}
void matmul_bare_hands(tensor* output, tensor* a, tensor* b)
{

    int batches = 1;
    for (int i = 0; i < a->dims - 2; i++)
    {
        batches *= a->shape[i];

    }
    int a_batch_stride = a->shape[a->dims - 2] * a->shape[a->dims - 1];
    int b_batch_stride = b->shape[b->dims - 2] * b->shape[b->dims - 1];
    int output_batch_stride = output->shape[output->dims - 2] * output->shape[output->dims - 1];
    float sum = 0.0;

    int bb_a_idx = 0;
    int bb_b_idx = 0;
    int bb_output_idx = 0;
    for (int bb = 0; bb < batches; bb++)
    {  
        bb_a_idx = bb * a_batch_stride;
        bb_b_idx = bb * b_batch_stride;
        bb_output_idx = bb * output_batch_stride;
        for (int i = 0; i < a->shape[a->dims - 2]; i++)
        {
            for (int j = 0; j < b->shape[b->dims - 1]; j++)
            {
                sum = 0.0;
                for (int k = 0; k < a->shape[a->dims - 1]; k++ )
                {
                    sum += a->data[bb_a_idx + i * a->stride[a->dims - 2] + k * a->stride[a->dims - 1]] * b->data[bb_b_idx + k * b->stride[b->dims - 2] + j * b->stride[b->dims - 1]];
                }
                output->data[bb_output_idx + i * output->stride[a->dims - 2] + j * output->stride[b->dims - 1]] = sum;
            }
        }
    }


}