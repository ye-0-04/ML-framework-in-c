#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "include/models.h"
#include "include/neural.h"
#include "include/tensors.h"

int main(void)
{
    const int test_size = 10000;

    float *x_test = malloc(test_size * 784 * sizeof(float));
    if (x_test == NULL)
    {
        printf("Failed to allocate x_test\n");
        return 1;
    }

    FILE *fx = fopen("mnist_test_x_v3.bin", "rb");
    if (fx == NULL)
    {
        printf("Failed to open mnist_test_x_v3.bin\n");
        free(x_test);
        return 1;
    }
    fread(x_test, sizeof(float), test_size * 784, fx);
    fclose(fx);

    uint8_t *y_test = malloc(test_size * sizeof(uint8_t));
    if (y_test == NULL)
    {
        printf("Failed to allocate y_test\n");
        free(x_test);
        return 1;
    }

    FILE *fy = fopen("mnist_test_y_v3.bin", "rb");
    if (fy == NULL)
    {
        printf("Failed to open mnist_test_y_v3.bin\n");
        free(x_test);
        free(y_test);
        return 1;
    }
    fread(y_test, sizeof(uint8_t), test_size, fy);
    fclose(fy);

    neural_network net;
    load_model(&net, "v4_net.bin");

    int input_shape[] = {1, 784, 1};
    tensor *input_ptr = create_tensor(input_shape, 3);
    tensor input_tensor = *input_ptr;
    free(input_ptr);

    int correct = 0;
    int predicted_counts[10] = {0};
    int actual_counts[10] = {0};

    for (int i = 0; i < test_size; i++)
    {
        for (int j = 0; j < 784; j++)
        {
            input_tensor.data[j] = x_test[i * 784 + j];
        }

        forward_pass(&net, &input_tensor);

        int pred = tensor_argmax_index(&net.layers[net.size - 1].output);
        int actual = y_test[i];

        if (pred >= 0 && pred < 10)
            predicted_counts[pred]++;
        if (actual >= 0 && actual < 10)
            actual_counts[actual]++;

        if (pred == actual)
            correct++;

        if (i < 20)
        {
            printf("i=%d pred=%d actual=%d correct_prob=%f max_prob=%f\n",
                   i,
                   pred,
                   actual,
                   net.layers[net.size - 1].output.data[actual],
                   net.layers[net.size - 1].output.data[pred]);
        }
    }

    printf("correct=%d / %d\n", correct, test_size);
    printf("Accuracy: %.2f%%\n", ((double)correct / test_size) * 100.0);

    printf("Actual counts: ");
    for (int i = 0; i < 10; i++)
        printf("%d:%d ", i, actual_counts[i]);
    printf("\n");

    printf("Predicted counts: ");
    for (int i = 0; i < 10; i++)
        printf("%d:%d ", i, predicted_counts[i]);
    printf("\n");

    free_tensor(&input_tensor);
    free(x_test);
    free(y_test);

    return 0;
}
