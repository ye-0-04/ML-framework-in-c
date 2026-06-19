#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "../include/tensors.h"

#include "../include/neural.h"



layer create_layer(int inputs, int outputs)
{
    layer l;
    int weights_shape[] = {1, outputs, inputs};
    int bias_shape[] = {1, outputs, 1};

    tensor* w = create_tensor(weights_shape, 3);
    l.weights = *w;
    free(w);      // free only the wrapper struct

    tensor* b = create_tensor(bias_shape, 3);
    l.bias = *b;
    free(b);

    rand_fill(&l.weights);
    rand_fill(&l.bias);

    init_empty_tensor(&l.gradient);
    init_empty_tensor(&l.input);
    init_empty_tensor(&l.output);
    
    return l;
}

neural_network create_net(int neural_shape[], int input, int size)
{
    neural_network nt;

    nt.layers = malloc(size * sizeof(layer));
    nt.layers[0] = create_layer(input, neural_shape[0]);
    for (int i = 1; i < size; i++)
    {
        // printf("create layer %i \n", i);
        nt.layers[i] = create_layer(neural_shape[i-1], neural_shape[i]);
        // printf("created layer %i \n", i);
    }
    nt.size = size;
    return nt;
}

void forward_pass(neural_network *nt, tensor* input)
{   
    // printf("hello forward\n");
    clone(&nt->layers[0].input, input);
    for (int i = 0; i < nt->size; i++)
    {
        // printf("Layer %d: weights shape [%d,%d,%d], input shape [%d,%d,%d]\n", i,
        // nt->layers[i].weights.shape[0],
        // nt->layers[i].weights.shape[1],
        // nt->layers[i].weights.shape[2],
        // nt->layers[i].input.shape[0],
        // nt->layers[i].input.shape[1],
        // nt->layers[i].input.shape[2]);
        tensor* x = tensor_matmul(&nt->layers[i].weights, &nt->layers[i].input);
        clone(&nt->layers[i].output, x);
        free_tensor(x);
        scal_tensor_add(&nt->layers[i].output, &nt->layers[i].bias);
        if (i == nt->size - 1)
        {
            softmax(&nt->layers[nt->size-1].output);
            // sigmoid(&nt->layers[nt->size-1].output);
        }
        else
        {
            reLu(&nt->layers[i].output);
            clone(&nt->layers[i+1].input , &nt->layers[i].output);
        }
        // printf("layer number %d\n", i);
        // print_tensor_values(&nt->layers[i].weights);
    }

}

void backprop(neural_network *nt, tensor* correct_output, double l_r)
{
    // printf("hello_backprop\n");
    //step one -> calculate the error between actual answer and our prediction
    tensor* delta = tensor_sub(correct_output, &nt->layers[nt->size-1].output);
    
    clone(&nt->layers[nt->size-1].gradient , delta);
    free_tensor(delta);
    for (int i = nt->size-1; i >= 0; i--)
    {
        tensor weighted_tensor_transposed;
        init_empty_tensor(&weighted_tensor_transposed);
        clone(&weighted_tensor_transposed, &nt->layers[i].weights);
        int swap[] = {weighted_tensor_transposed.dims -1,weighted_tensor_transposed.dims -2 };
        transpose(&weighted_tensor_transposed, swap );
        
        tensor* delta_sent_back = tensor_matmul(&weighted_tensor_transposed, &nt->layers[i].gradient);
        
       
        for (int j = 0; j < nt->layers[i].gradient.size; j++) {
            if (nt->layers[i].output.data[j] == 0.0)
                nt->layers[i].gradient.data[j] = 0.0;
        }

        //bias training
        tensor x;
        init_empty_tensor(&x);
        clone(&x , &nt->layers[i].gradient);
        scal_mul(&x, l_r);
        scal_tensor_sub(&nt->layers[i].bias, &x);
        //
        tensor r;
        init_empty_tensor(&r);
        clone(&r , &nt->layers[i].gradient);

        tensor clone_input;
        init_empty_tensor(&clone_input);
        clone(&clone_input,&nt->layers[i].input);
        int axes[] = {nt->layers[i].input.dims - 1 , nt->layers[i].input.dims - 2};
        transpose(&clone_input , axes);
        tensor* m = tensor_matmul(&r, &clone_input);  //delta x input
        scal_mul(m, l_r);  //delta x input x learning rate
        tensor_sub(&nt->layers[i].weights , m);
        free_tensor(m);
        free_tensor(delta_sent_back);
        free_tensor(&weighted_tensor_transposed);
    }
}

void backprop_v2(neural_network *nt, tensor* correct_output, double l_r)
{
    // printf("hello_backprop\n");
    //step one -> calculate the error between actual answer and our prediction
    tensor* delta = tensor_sub(&nt->layers[nt->size-1].output, correct_output);
    
    clone(&nt->layers[nt->size-1].gradient , delta);
    free_tensor(delta);
    
    for (int i = nt->size-1; i >= 0; i--)
    {
        // FIX 1: Apply ReLU derivative BEFORE matmul (moved up)
        // Apply ReLU derivative only to hidden layers, not the final softmax layer
        if (i != nt->size - 1)
        {
            for (int j = 0; j < nt->layers[i].gradient.size; j++)
            {
                if (nt->layers[i].output.data[j] == 0.0)
                    nt->layers[i].gradient.data[j] = 0.0;
            }
        }

        tensor weighted_tensor_transposed;
        init_empty_tensor(&weighted_tensor_transposed);
        clone(&weighted_tensor_transposed, &nt->layers[i].weights);
        int swap[] = {weighted_tensor_transposed.dims -1, weighted_tensor_transposed.dims -2 };
        transpose(&weighted_tensor_transposed, swap);
        
        tensor* delta_sent_back = tensor_matmul(&weighted_tensor_transposed, &nt->layers[i].gradient);
        
        // FIX 2: Propagate error to previous layer
        if (i != 0) {
            clone(&nt->layers[i-1].gradient, delta_sent_back);
        }

        //bias training
        tensor x;
        init_empty_tensor(&x);
        clone(&x , &nt->layers[i].gradient);
        scal_mul(&x, l_r);
        scal_tensor_sub(&nt->layers[i].bias, &x);
        
        tensor r;
        init_empty_tensor(&r);
        clone(&r , &nt->layers[i].gradient);

        tensor clone_input;
        init_empty_tensor(&clone_input);
        clone(&clone_input, &nt->layers[i].input);
        int axes[] = {nt->layers[i].input.dims - 1 , nt->layers[i].input.dims - 2};
        transpose(&clone_input, axes);
        
        tensor* m = tensor_matmul(&r, &clone_input);  //delta x input
        scal_mul(m, l_r);  //delta x input x learning rate
        scal_tensor_sub(&nt->layers[i].weights, m);
        
        // FIX 3: Free all local tensors to prevent memory leaks
        free_tensor(m);
        free_tensor(delta_sent_back);
        free_tensor(&weighted_tensor_transposed);
        free_tensor(&x);
        free_tensor(&r);
        free_tensor(&clone_input);
    }
}

//optimised version


void forward_pass_v2(neural_network *nt, tensor* input)
{
    
    clone(&nt->layers[0].input, input );
    for (int i = 0; i < nt->size; i++)
    {
        matmul_bare_hands_v2(&nt->layers[i].output, &nt->layers[i].weights, &nt->layers[i].input);
        scal_tensor_add(&nt->layers[i].output, &nt->layers[i].bias);

        if (i == nt->size - 1)
        {
            softmax(&nt->layers[nt->size-1].output);
        }
        else
        {
            reLu(&nt->layers[i].output);
            
            clone(&nt->layers[i+1].input, &nt->layers[i].output);
        }
    }


}