#include <stdint.h>

/**
 * @brief Forward declaration of tensor structure
 */
typedef struct tensor tensor;

/**
 * @brief Creates a new tensor with the specified shape and dimensions
 * @param shape Array specifying the size of each dimension
 * @param dims Number of dimensions
 * @return Pointer to the created tensor, or NULL on failure
 */
tensor* create_tensor(int* shape, int dims);

/**
 * @brief Performs element-wise addition of two tensors
 * @param a First tensor
 * @param b Second tensor
 * @return New tensor with the result, or NULL on failure
 */
tensor* add_tensor(tensor* a, tensor* b);

/**
 * @brief Performs element-wise subtraction of two tensors
 * @param a First tensor
 * @param b Second tensor
 * @return New tensor with the result, or NULL on failure
 */
tensor* sub(tensor* a, tensor* b);

/**
 * @brief Performs element-wise multiplication of two tensors
 * @param a First tensor
 * @param b Second tensor
 * @return New tensor with the result, or NULL on failure
 */
tensor* mul(tensor* a, tensor* b);

/**
 * @brief Performs element-wise division of two tensors
 * @param a First tensor (numerator)
 * @param b Second tensor (denominator)
 * @return New tensor with the result, or NULL on failure
 */
tensor* div(tensor* a, tensor* b);

/**
 * @brief Fills all elements of a tensor with a specified value
 * @param a Tensor to fill
 * @param b Value to fill with
 */
void fill(tensor* a, double b);

/**
 * @brief Fills all elements of a tensor with random double values
 * @param a Tensor to fill
 */
void rand_fill(tensor* a);

/**
 * @brief Copies data from one tensor to another (both must have same size)
 * @param a Destination tensor
 * @param b Source tensor
 */
void copy(tensor* a, tensor* b);

/**
 * @brief Reshapes a tensor to new dimensions (total size must remain the same)
 * @param a Tensor to reshape
 * @param shape New shape array
 * @param dims New number of dimensions
 */
void reshape(tensor* a, int* shape, int dims);

/**
 * @brief Adds a scalar value to all elements of a tensor (in-place)
 * @param a Tensor to modify
 * @param b Scalar value to add
 */
void scal_add(tensor* a, double b);

/**
 * @brief Multiplies all elements of a tensor by a scalar value (in-place)
 * @param a Tensor to modify
 * @param b Scalar value to multiply by
 */
void scal_mul(tensor* a, double b);

void transpose(tensor* a, const int* swap_axes[2]);

double tensor_sum(tensor* a);

double tensor_get_max(tensor* a);

double tensor_get_min(tensor* a);

double tensor_argmax(tensor* a);

double tensor_argmin(tensor* a);

/**
 * @brief Compares the size and dimensions of two tensors
 * @param a First tensor
 * @param b Second tensor
 * @return 0 if tensors have the same size, 1 otherwise
 */
int comp_tensor_size(tensor* a, tensor* b);
