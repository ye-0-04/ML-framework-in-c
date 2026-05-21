

#include <stdio.h>
#include <stdint.h>

#include "../include/neural.h"


int main()
{
    printf("hello\n");
    int shape[] = {4,3};

    neural_network net = create_net(shape , 3 , 2);
    
    int s[] = {1,3,1};
    tensor* input = create_tensor(s, 3);
    input->data[0] = 1.5;
    input->data[1] = 9.6;
    input->data[2] = 7.8;

    
    tensor* target = create_tensor(s, 3);
    target->data[0] = 2.0;
    target->data[1] = 1.0;
    target->data[2] = 1.0;
    for (int epoch = 0; epoch < 10000; epoch++)
    {
        
        printf("hello\n");
        forward_pass(&net, input);
        backprop_v2(&net, target, 0.001);

        
            
        
            
        
    }
    int size = get_tensor_size(&net.layers[net.size-1].output);
    printf("%d", size);
    for (int i = 0; i < 3; i++)
    {
        printf("%d\n", get_tensor_val(&net.layers[net.size-1].output, i));
    }
    return 0;
}
