#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <windows.h>

#include "include/neural.h"
#include "include/tensors.h"
#include "include/models.h"

int TRAIN_SIZE =  60000;
int EPOCHS = 1;
void load_input_sample(tensor *t, float *dataset, int index);
void load_label(tensor *t, int label);
int main()
{
    // Train Model
    LARGE_INTEGER frequency, start, end;
    double elapsed;
    QueryPerformanceFrequency(&frequency);
    
    // Start timing
    QueryPerformanceCounter(&start);


    FILE *f = fopen("mnist_x.bin", "rb");

    float *X_train = malloc(TRAIN_SIZE * 784 * sizeof(float));

    fread(X_train, sizeof(float), TRAIN_SIZE * 784, f);

    fclose(f);

    FILE *f2 = fopen("mnist_y.bin", "rb");

    uint8_t *y_train = malloc(TRAIN_SIZE);

    fread(y_train, sizeof(uint8_t), TRAIN_SIZE, f2);

    fclose(f2);
 
    int shape[] = {128 , 64, 10};

    neural_network net = create_net(shape , 784 , 3);

    int input_shape[] = {1, 784, 1};
    tensor *input_ptr = create_tensor(input_shape, 3);
    tensor input_tensor = *input_ptr;
    free(input_ptr);

    int target_shape[] = {1, 10, 1};
    tensor *target_ptr = create_tensor(target_shape, 3);
    tensor target_tensor = *target_ptr;
    free(target_ptr);

    double learning_rate = 0.01;

    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        for (int i = 0; i < TRAIN_SIZE; i++)
        {

            printf("\033[2J\033[H");
            printf("%i", i);
            load_input_sample(&input_tensor, X_train, i);

            input_tensor.shape[0] = 1;
            input_tensor.shape[1] = 784;
            input_tensor.shape[2] = 1;

            load_label(&target_tensor, y_train[i]);

            forward_pass(&net, &input_tensor);

            backprop_v2(&net, &target_tensor, learning_rate);

            if (i % 1000 == 0)
            {
                printf("Sample %d, label %d\n", i, y_train[i]);
            }
            printf("Epoch %i done\n", epoch);
        }
        printf("Epoch %i done\n", epoch);
    }
    save_model(&net, "v7_net.bin");

    QueryPerformanceCounter(&end);
    
    // Calculate elapsed time in seconds
    elapsed = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;
    
    printf("\nExecution time: %.6f seconds\n", elapsed);
    printf("Execution time: %.2f milliseconds\n", elapsed * 1000);
    
    // EValuate
    int correct = 0;
    float *X_test = malloc(10000 * 784 * sizeof(float));
    FILE *fx = fopen("mnist_test_x_v3.bin", "rb");
    fread(X_test, sizeof(float), 10000*784, fx);
    fclose(fx);

    uint8_t *y_test = malloc(10000 * sizeof(uint8_t));
    FILE *fy = fopen("mnist_test_y_v3.bin", "rb");
    fread(y_test, sizeof(uint8_t), 10000, fy);
    fclose(fy);

    int input_shape_2[] = {1, 784, 1};
    tensor *input_ptr_2 = create_tensor(input_shape_2, 3);
    tensor input_tensor_2 = *input_ptr_2;
    free(input_ptr_2);

    neural_network net_2;
    load_model(&net_2, "v7_net.bin");
    for (int i = 0; i < 10000; i++)
    {
        for (int j = 0; j < 784; j++)
        {
            input_tensor_2.data[j] = X_test[i * 784 + j];
        }

        forward_pass(&net, &input_tensor_2);

        int predicted = 0;
        int max = tensor_argmax_index(&net.layers[net.size-1].output);
        int actual = y_test[i];

        if (max == actual) correct++;

    }
    double accuracy = ((double)correct / 10000.0) * 100.0;
    printf("Accuracy: %.2f%%\n", accuracy);
    return 0;
}
void load_input_sample(tensor *t, float *dataset, int index)
{
    for (int i = 0; i < 784; i++)
    {
        t->data[i] = dataset[index * 784 + i];
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