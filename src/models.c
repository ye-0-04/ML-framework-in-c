#include "../include/neural.h"
#include "../include/tensors.h"

#include <stdio.h>
#include <stdlib.h>



void save_model(neural_network *nt, const char *filename)
{
    FILE *f = fopen(filename, "wb");
    fwrite(&nt->size, sizeof(int), 1, f);

    for (int i = 0; i < nt->size; i++)
    {
        layer *l = &nt->layers[i];

        fwrite(&l->weights.dims, sizeof(int), 1 , f);
        printf("Layer %i weights dimensions saved\n", i);
        fwrite(&l->weights.size, sizeof(int), 1 , f);
        printf("Layer %i weights size saved\n", i);
        fwrite(l->weights.shape, sizeof(int), l->weights.dims, f);
        printf("Layer %i weights shape saved\n", i);
        fwrite(l->weights.stride, sizeof(int), l->weights.dims, f);
        printf("Layer %i weights stride saved\n", i);
        fwrite(l->weights.data, sizeof(double), l->weights.size, f);
        printf("Layer %i weights data saved\n", i);


        fwrite(&l->bias.dims, sizeof(int), 1, f);
        printf("Layer %i bias dimension saved\n", i);
        fwrite(&l->bias.size, sizeof(int), 1 , f);
        printf("layer %i bias size saved\n", i);
        fwrite(l->bias.shape, sizeof(int), l->bias.dims, f);
        printf("Layer %i bias shape saved\n", i);
        fwrite(l->bias.stride, sizeof(int), l->bias.dims, f);
        printf("Layer %i bias stride saved\n", i);
        fwrite(l->bias.data, sizeof(double), l->bias.size, f);
        printf("Layer %i bias data saved\n", i);
    }
    printf("Model saved successfully");
    fclose(f);
}

void load_model(neural_network *nt, const char *filename)
{
    FILE *f = fopen(filename, "rb");
    fread (&nt->size, sizeof(int), 1, f);
    nt->layers = malloc(sizeof(layer) * nt->size);

    for (int i = 0; i < nt->size; i++)
    {
        nt->layers[i].weights = load_tensor(f);
        printf("Layer %i weights loaded\n", i);
        nt->layers[i].bias = load_tensor(f);
        printf("Layer %i bias loaded\n", i);

        init_empty_tensor(&nt->layers[i].input);
        init_empty_tensor(&nt->layers[i].output);
        init_empty_tensor(&nt->layers[i].gradient);
        
    }

    fclose(f);
}

