#include <stdint.h>


typedef struct tensor tensor;

tensor* create_tensor(int* shape, int dims);
tensor* add_tensor(tensor* a, tensor* b);
tensor* mul(tensor* a, tensor* b);
tensor* div(tensor* a, tensor* b);
tensor* fill(tensor* a, double b);
tensor* rand_fill(tensor* a);
int comp_tensor_size(tensor* a, tensor* b);
void copy(tensor* a, tensor* b);
void reshape(tensor* a, int rows, int cols, int dims);
void scal_add(tensor* a, double b);
void scal_mul(tensor* a, double b);