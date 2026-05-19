#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "../include/tensors.h"


typedef struct layer
{
    tensor* weights;
    tensor* bias;
    
}layer;
typedef struct neural_network
{
    layer* layers;
}neural_network;


layer create_layer(int inputs, int outputs)
{
    layer l;
    int weights_shape[] = {1, outputs , inputs};
    int bias_shape[] = {1, outputs, 1};

    l.weights = create_tensor(weights_shape, 3);
    l.bias = create_tensor(bias_shape, 3);

    return l;
}

neural_network create_net(int neural_shape[], int input, int size)
{
    neural_network nt;

    nt.layers = malloc(size * sizeof(layer));
    nt.layers[0] = create_layer(input, neural_shape[1]);
    for (int i = 1; i < size; i++)
    {
        nt.layers[i] = create_layer(neural_shape[i-1], neural_shape[i]);
    }
    return nt;
}