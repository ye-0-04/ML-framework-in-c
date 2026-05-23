#ifndef MODELS_H
#define MODELS_H


#include "../include/neural.h"


void save_model(neural_network *nt, const char *filename);
void load_model(neural_network *nt, const char *filename);
#endif