#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "include/models.h"
#include "include/neural.h"
#include "include/tensors.h"

void load_input_sample(tensor *t, float *dataset, int index);
void load_label(tensor *t, int label);

int main(void)
{
    const int train_size = 1;
    const int epochs = 1000;
    const double learning_rate = 0.1;

    FILE *fx = fopen("mnist_x.bin", "rb");
    if (fx == NULL)
    {
        printf("Failed to open mnist_x.bin\n");
        return 1;
    }

    float *x_train = malloc(train_size * 784 * sizeof(float));
    if (x_train == NULL)
    {
        printf("Failed to allocate x_train\n");
        fclose(fx);
        return 1;
    }

    fread(x_train, sizeof(float), train_size * 784, fx);
    fclose(fx);

    FILE *fy = fopen("mnist_y.bin", "rb");
    if (fy == NULL)
    {
        printf("Failed to open mnist_y.bin\n");
        free(x_train);
        return 1;
    }

    uint8_t *y_train = malloc(train_size * sizeof(uint8_t));
    if (y_train == NULL)
    {
        printf("Failed to allocate y_train\n");
        fclose(fy);
        free(x_train);
        return 1;
    }

    fread(y_train, sizeof(uint8_t), train_size, fy);
    fclose(fy);

    int shape[] = {10};
    neural_network net = create_net(shape, 784, 1);

    int input_shape[] = {1, 784, 1};
    tensor *input_ptr = create_tensor(input_shape, 3);
    tensor input_tensor = *input_ptr;
    free(input_ptr);

    int target_shape[] = {1, 10, 1};
    tensor *target_ptr = create_tensor(target_shape, 3);
    tensor target_tensor = *target_ptr;
    free(target_ptr);

    printf("Debug overfit test: one sample, label=%d\n", y_train[0]);

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        load_input_sample(&input_tensor, x_train, 0);
        load_label(&target_tensor, y_train[0]);

        forward_pass(&net, &input_tensor);

        int pred = tensor_argmax_index(&net.layers[net.size - 1].output);
        double confidence = net.layers[net.size - 1].output.data[y_train[0]];

        if (epoch % 100 == 0 || epoch == epochs - 1)
        {
            printf("epoch=%d pred=%d actual=%d confidence=%f\n",
                   epoch, pred, y_train[0], confidence);
        }

        backprop_v2(&net, &target_tensor, learning_rate);
    }

    save_model(&net, "debug_one_sample_net.bin");

    free_tensor(&input_tensor);
    free_tensor(&target_tensor);
    free(x_train);
    free(y_train);

    return 0;
}

void load_input_sample(tensor *t, float *dataset, int index)
{
    for (int i = 0; i < 784; i++)
    {
        t->data[i] = dataset[index * 784 + i] / 255.0;
    }
}

void load_label(tensor *t, int label)
{
    for (int i = 0; i < 10; i++)
    {
        t->data[i] = 0.0;
    }

    t->data[label] = 1.0;
}
