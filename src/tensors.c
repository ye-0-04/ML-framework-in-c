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

    
    int shape[] = {1 , 1, 1};
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



void matmul_bare_hands(tensor* output, tensor* a, tensor* b)
{
    memset(output->data, 0.0, output->size * sizeof(float));
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

void matmul_bare_hands_v2(tensor* output, tensor* a, tensor* b)
{   
    memset(output->data, 0.0, output->size * sizeof(float));
    int batches = 1;

    for (int i = 0; i < a->dims - 2; i++)
        batches *= a->shape[i];

    int M = a->shape[a->dims - 2];
    int K = a->shape[a->dims - 1];
    int N = b->shape[b->dims - 1];

    int a_batch_stride = M * K;
    int b_batch_stride = K * N;
    int out_batch_stride = M * N;

    for (int bb = 0; bb < batches; bb++)
    {
        int a_base = bb * a_batch_stride;
        int b_base = bb * b_batch_stride;
        int out_base = bb * out_batch_stride;

        for (int i = 0; i < M; i++)
        {
            float* c_row =
                &output->data[
                    out_base +
                    i * output->stride[output->dims - 2]
                ];

            float* a_row =
                &a->data[
                    a_base +
                    i * a->stride[a->dims - 2]
                ];

            for (int k = 0; k < K; k++)
            {
                float a_val = a_row[k];

                float* b_row =
                    &b->data[
                        b_base +
                        k * b->stride[b->dims - 2]
                    ];

                for (int j = 0; j < N; j++)
                {
                    c_row[j] += a_val * b_row[j];
                }
            }
        }
    }
}


void matmul_simd_avx2(tensor* output, tensor* a, tensor* b)
{   
    memset(output->data, 0.0, output->size * sizeof(float));
    int batches = 1;

    for (int d = 0; d < a->dims - 2; d++)
        batches *= a->shape[d];

    const int M = a->shape[a->dims - 2];
    const int K = a->shape[a->dims - 1];
    const int N = b->shape[b->dims - 1];

    const int a_batch_stride = M * K;
    const int b_batch_stride = K * N;
    const int out_batch_stride = M * N;

    for (int bb = 0; bb < batches; bb++)
    {
        const int a_base = bb * a_batch_stride;
        const int b_base = bb * b_batch_stride;
        const int out_base = bb * out_batch_stride;

        for (int i = 0; i < M; i++)
        {
            float* c_row =
                output->data +
                out_base +
                i * output->stride[output->dims - 2];

            float* a_row =
                a->data +
                a_base +
                i * a->stride[a->dims - 2];

            for (int k = 0; k < K; k++)
            {
                const float a_val = a_row[k];

                const __m256 a_vec = _mm256_set1_ps(a_val);

                float* b_row =
                    b->data +
                    b_base +
                    k * b->stride[b->dims - 2];

                int j = 0;

                for (; j <= N - 8; j += 8)
                {
                    __m256 b_vec = _mm256_loadu_ps(b_row + j);
                    __m256 c_vec = _mm256_loadu_ps(c_row + j);

                    c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);

                    _mm256_storeu_ps(c_row + j, c_vec);
                }

                for (; j < N; j++)
                {
                    c_row[j] += a_val * b_row[j];
                }
            }
        }
    }
}
void matmul_simd_omp_batch(tensor* output, tensor* a, tensor* b)
{
    memset(output->data, 0.0, output->size * sizeof(float));
    int batches = 1;

    for (int i = 0; i < a->dims - 2; i++)
        batches *= a->shape[i];

    int M = a->shape[a->dims - 2];
    int K = a->shape[a->dims - 1];
    int N = b->shape[b->dims - 1];

    int a_batch_stride = M * K;
    int b_batch_stride = K * N;
    int out_batch_stride = M * N;

    #pragma omp parallel for schedule(static)
    for (int bb = 0; bb < batches; bb++)
    {
        int a_base   = bb * a_batch_stride;
        int b_base   = bb * b_batch_stride;
        int out_base = bb * out_batch_stride;

        for (int i = 0; i < M; i++)
        {
            float* c_row =
                &output->data[
                    out_base +
                    i * output->stride[output->dims - 2]
                ];

            float* a_row =
                &a->data[
                    a_base +
                    i * a->stride[a->dims - 2]
                ];

            for (int k = 0; k < K; k++)
            {
                float a_val = a_row[k];

                __m256 a_vec = _mm256_set1_ps(a_val);

                float* b_row =
                    &b->data[
                        b_base +
                        k * b->stride[b->dims - 2]
                    ];

                int j;

                for (j = 0; j <= N - 8; j += 8)
                {
                    __m256 b_vec = _mm256_loadu_ps(b_row + j);
                    __m256 c_vec = _mm256_loadu_ps(c_row + j);

                    c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);

                    _mm256_storeu_ps(c_row + j, c_vec);
                }

                for (; j < N; j++)
                {
                    c_row[j] += a_val * b_row[j];
                }
            }
        }
    }
}