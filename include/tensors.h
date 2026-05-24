#ifndef TENSORS_H
#define TENSORS_H

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>


typedef struct tensor{
    int* shape;  //[[.....,row,col]
    int* stride; 
    double *data;
    int dims;
    int size;
}tensor;


tensor* create_tensor(int shape[], int dims);


tensor* add_tensor(tensor* a, tensor* b, bool con);


tensor* tensor_sub(tensor* a, tensor* b);


tensor* tensor_mul(tensor* a, tensor* b);


tensor* tensor_div(tensor* a, tensor* b);

tensor* tensor_matmul(tensor* a, tensor* b);

tensor* tensor_apply_broadcast(tensor* a, tensor* b, double (*op)(double, double));

tensor load_tensor(FILE *f);

double tensor_sum(tensor* a);


double tensor_get_max(tensor* a);


double tensor_get_min(tensor* a);


double tensor_argmax(tensor* a);


double tensor_argmin(tensor* a);

double add(double x, double y);

double sub(double x, double y);

double mul(double x, double y);

double div_2(double x, double y);

double mean_squared_error(double true_val[],int true_val_size, double predictions[], int predictions_size);

double entropy(double pred[],int size);

const double get_tensor_val(tensor* a, int index);


int copy(tensor* a, tensor* b);


int reshape(tensor* a, int shape[], int dims);

int comp_tensor_size(tensor* a, tensor* b);

int broadcast_compatible(tensor* a, tensor* b);

int tensor_argmax_index(tensor *t);

const int get_tensor_size(tensor* a);

const int get_tensor_dims(tensor* a);

const int* get_tensor_shape(tensor* a);

const int* get_tensor_stride(tensor* a);


void fill(tensor* a, double b);


void rand_fill(tensor* a);

void scal_add(tensor* a, double b);

void scal_pow(tensor*a, int b);


void scal_mul(tensor* a, double b);


void transpose(tensor* a, int swap_axes[]);

void print_tensor_rec(tensor* a, int dim, int offset);

void set_tensor_val(tensor* a, double val, int index);

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

#endif