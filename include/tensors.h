#ifndef TENSORS_H
#define TENSORS_H

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>


typedef struct tensor{
    int* shape;  //[[.....,row,col]
    int* stride; 
    float *data;
    int dims;
    int size;
}tensor;


tensor* create_tensor(int shape[], int dims);


tensor* add_tensor(tensor* a, tensor* b, bool con);


tensor* tensor_sub(tensor* a, tensor* b);


tensor* tensor_mul(tensor* a, tensor* b);


tensor* tensor_div(tensor* a, tensor* b);

tensor* tensor_matmul(tensor* a, tensor* b);

tensor* tensor_apply_broadcast(tensor* a, tensor* b, float (*op)(float, float));

tensor load_tensor(FILE *f);

float tensor_sum(tensor* a);


float tensor_get_max(tensor* a);


float tensor_get_min(tensor* a);

float add(float x, float y);

float sub(float x, float y);

float mul(float x, float y);

float div_2(float x, float y);

float mean_squared_error(float true_val[],int true_val_size, float predictions[], int predictions_size);

float entropy(float pred[],int size);

const float get_tensor_val(tensor* a, int index);

int tensor_argmax(tensor* a);

int tensor_argmin(tensor* a);

int copy(tensor* a, tensor* b);


int reshape(tensor* a, int shape[], int dims);

int comp_tensor_size(tensor* a, tensor* b);

int broadcast_compatible(tensor* a, tensor* b);

int tensor_argmax_index(tensor *t);

const int get_tensor_size(tensor* a);

const int get_tensor_dims(tensor* a);

const int* get_tensor_shape(tensor* a);

const int* get_tensor_stride(tensor* a);


void fill(tensor* a, float b);


void rand_fill(tensor* a);

void scal_add(tensor* a,float b);

void scal_pow(tensor*a, int b);


void scal_mul(tensor* a, float b);


void transpose(tensor* a, int swap_axes[]);

void print_tensor_rec(tensor* a, int dim, int offset);

void set_tensor_val(tensor* a, float val, int index);

void free_tensor(tensor* a);

void reLu(tensor*  x);

void sigmoid(tensor* x);

void softmax(tensor *x);

void clone(tensor* a, tensor* b);

void scal_tensor_add(tensor* a, tensor* b);

void scal_tensor_sub(tensor* a, tensor* b);

void scal_tensor_mul(tensor* a, tensor* b);

void print_tensor_values(tensor* a);

void init_empty_tensor(tensor* a);

void tensor_matmul_V2(tensor* output, tensor* a, tensor* b);

void tensor_matmul_V2_simd_packed(tensor* output, tensor* a, tensor* b);

void tensor_matmul_V2_simd_packed_v2(tensor* output, tensor* a, tensor* b);
#endif