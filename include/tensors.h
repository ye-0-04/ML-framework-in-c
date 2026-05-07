#include <stdint.h>


typedef struct tensor tensor;

tensor* create_tensor(int rows , int cols , int dims);
tensor* add_tensor(tensor* a, tensor* b);