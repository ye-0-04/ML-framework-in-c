#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "../include/tensors.h"

typedef struct neuron
{
    tensor* weights;
    double bias;
    
}neuron;
typedef struct layer
{
    tensor* weights;
    tensor* bias;
    
    
    

}layer;

typedef struct neural_network
{
    layer *layers;
}neural_network;

layer create_layer(int num_of_neurons, int num_of_inputs)
{
    
    if (num_of_neurons <=0) return;

    layer l;
    int shape[] = {1,num_of_neurons,num_of_inputs};
    int bias_shape[] = {1, num_of_inputs, 1};
    
    l.weights = create_tensor(shape, 3);
    l.bias = create_tensor(shape, 3);
    


}
neural_network create_network(int num_of_layers[])
{
    int size = sizeof(num_of_layers)/ sizeof(num_of_layers[0]);
    if (size <= 1) return;

    neural_network nt;
    for (int i = 0; i < size; i++)
    {
        nt.layers[i] = create_layer(num_of_layers[i], num_of_layers[i-1]);
    }
}