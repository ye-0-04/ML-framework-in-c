

#include <stdio.h>
#include <stdint.h>

#include "../include/neural.h"
#include "../include/tensors.h"
#include "../include/models.h"
int main()
{
    printf("hello\n");
    int shape[] = {4,3};

    neural_network net = create_net(shape , 3 , 2);
    
    int s[] = {1,3,1};
    tensor* input = create_tensor(s, 3);
    input->data[0] = 5.6;
    input->data[1] = 9.6;
    input->data[2] = 7.8;

    
    tensor* target = create_tensor(s, 3);
    target->data[0] = 1.0;
    target->data[1] = 1.0;
    target->data[2] = 1.0;
    FILE *f = fopen("model.bin", "wb");
    save_model(&net, "model.bin");
    neural_network net_v2;
    load_model(&net_v2, "model.bin");
    // for (int epoch = 0; epoch < 100000; epoch++)
    // {
        
        
    //     forward_pass(&net, input);
    //     backprop_v2(&net, target, 0.0001);
    //     for (int i = 0; i < 3; i++)
    //     {
    //         printf("%f,", get_tensor_val(&net.layers[net.size-1].output, i));
    //     }
    //     printf("\n");
    // }

    

    return 0;
}
