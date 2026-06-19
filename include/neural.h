#ifndef NEURAL_H
#define NEURAL_H


#include "../include/tensors.h"



typedef struct layer
{
    tensor weights;
    tensor bias;
    tensor output;
    tensor input;
    tensor gradient;
    
}layer;
typedef struct neural_network
{
    layer *layers;
    int size;
}neural_network;

layer create_layer(int inputs, int outputs);

neural_network create_net(int neural_shape[], int input, int size);

void forward_pass(neural_network *nt, tensor* input);

void backprop(neural_network *nt, tensor* correct_output, double l_r);

void backprop_v2(neural_network *nt, tensor* correct_output, double l_r);
void forward_pass_v2(neural_network *nt, tensor* input);
#endif