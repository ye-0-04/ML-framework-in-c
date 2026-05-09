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
tensor* create_tensor(int shape[], int dims);

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
tensor* tensor_div(tensor* a, tensor* b);

tensor* tensor_matmul(tensor* a, tensor* b);

/**
 * @brief Fills all elements of a tensor with a specified value
 * @param a Tensor to fill
 * @param b Value to fill with
 */
int fill(tensor* a, double b);

/**
 * @brief Fills all elements of a tensor with random double values
 * @param a Tensor to fill
 */
int rand_fill(tensor* a);

/**
 * @brief Copies data from one tensor to another (both must have same size)
 * @param a Destination tensor
 * @param b Source tensor
 */
int copy(tensor* a, tensor* b);

/**
 * @brief Reshapes a tensor to new dimensions (total size must remain the same)
 * @param a Tensor to reshape
 * @param shape New shape array
 * @param dims New number of dimensions
 */
int reshape(tensor* a, int* shape, int dims);

/**
 * @brief Adds a scalar value to all elements of a tensor (in-place)
 * @param a Tensor to modify
 * @param b Scalar value to add
 */
int scal_add(tensor* a, double b);

/**
 * @brief Multiplies all elements of a tensor by a scalar value (in-place)
 * @param a Tensor to modify
 * @param b Scalar value to multiply by
 */
int scal_mul(tensor* a, double b);

/**
 * @brief Transposes a tensor by swapping two axes
 * @param a Tensor to transpose
 * @param swap_axes Array of 2 axis indices to swap
 * @return 0 on success, 1 on failure
 */
int transpose(tensor* a, int swap_axes[2]);

/**
 * @brief Computes the sum of all elements in a tensor
 * @param a Tensor to sum
 * @return Sum of all elements, or 1 on error
 */
double tensor_sum(tensor* a);

/**
 * @brief Finds the maximum value in a tensor
 * @param a Tensor to search
 * @return Maximum value, or 1 on error
 */
double tensor_get_max(tensor* a);

/**
 * @brief Finds the minimum value in a tensor
 * @param a Tensor to search
 * @return Minimum value, or 1 on error
 */
double tensor_get_min(tensor* a);

/**
 * @brief Finds the index of the maximum value in a tensor
 * @param a Tensor to search
 * @return Index of maximum value, or 1 on error
 */
double tensor_argmax(tensor* a);

/**
 * @brief Finds the index of the minimum value in a tensor
 * @param a Tensor to search
 * @return Index of minimum value, or 1 on error
 */
double tensor_argmin(tensor* a);

/**
 * @brief Compares the size and dimensions of two tensors
 * @param a First tensor
 * @param b Second tensor
 * @return 0 if tensors have the same size, 1 otherwise
 */
int comp_tensor_size(tensor* a, tensor* b);

void print_tensor_rec(tensor* a, int dim, int offset);

/**
 * @brief Gets the value at a specific index in the tensor's flat data array
 * @param a Tensor to get value from
 * @param index Position in the flat data array
 * @return Value at the specified index
 */
double get_tensor_val(tensor* a, int index);

/**
 * @brief Sets the value at a specific index in the tensor's flat data array
 * @param a Tensor to set value in
 * @param val Value to set
 * @param index Position in the flat data array
 */
void set_tensor_val(tensor* a, double val, int index);