// High-level CUDA tensor operations for AiCraft
// This file provides the interface between tensor.c and CUDA kernels

#include <cuda_runtime.h>
#include <stdio.h>

// === EXTERNAL KERNEL DECLARATIONS ===

extern "C" {
    // From cuda_kernels.cu
    void cuda_tensor_add(const float* a, const float* b, float* c, int size);
    void cuda_tensor_mul(const float* a, const float* b, float* c, int size);
    void cuda_tensor_scalar_mul(const float* a, float scalar, float* c, int size);
    void cuda_fill(float* data, float value, int size);
    void cuda_relu(const float* input, float* output, int size);
    void cuda_gelu(const float* input, float* output, int size);
    void cuda_sigmoid(const float* input, float* output, int size);
    void cuda_softmax(const float* input, float* output, int rows, int cols);
    void cuda_gemm(const float* A, const float* B, float* C, int M, int N, int K);
    
    // From cuda_utils.cu
    bool cuda_init_device(void* device_info);
    void cuda_cleanup();
    float* cuda_malloc(size_t size);
    void cuda_free(float* ptr);
    void cuda_memcpy_to_device(float* dst, const float* src, size_t size);
    void cuda_memcpy_to_host(float* dst, const float* src, size_t size);
    void cuda_memcpy_device_to_device(float* dst, const float* src, size_t size);
    void cuda_gemm_blas(const float* A, const float* B, float* C, 
                        int M, int N, int K, float alpha, float beta);
    bool cuda_is_available();
    void cuda_synchronize();
    void cuda_check_error(const char* operation);
    size_t cuda_get_memory_usage();
    void cuda_print_memory_stats();
    void cuda_device_info();
}

// === SMART KERNEL SELECTION ===

// Auto-select between custom kernels and cuBLAS based on problem size
void cuda_gemm_auto(const float* A, const float* B, float* C, int M, int N, int K) {
    // Use cuBLAS for large matrices (better optimized)
    if (M >= 512 && N >= 512 && K >= 512) {
        cuda_gemm_blas(A, B, C, M, N, K, 1.0f, 0.0f);
    } else {
        // Use our custom ultra-optimized kernel for smaller matrices
        cuda_gemm(A, B, C, M, N, K);
    }
    
    cuda_check_error("gemm_auto");
}

// === TENSOR OPERATION WRAPPERS ===

extern "C" {

// Memory management wrappers
float* aicraft_cuda_malloc(size_t size) {
    return cuda_malloc(size);
}

void aicraft_cuda_free(float* ptr) {
    cuda_free(ptr);
}

void aicraft_cuda_memcpy_h2d(float* dst, const float* src, size_t size) {
    cuda_memcpy_to_device(dst, src, size);
}

void aicraft_cuda_memcpy_d2h(float* dst, const float* src, size_t size) {
    cuda_memcpy_to_host(dst, src, size);
}

// Basic tensor operations
void aicraft_cuda_add(const float* a, const float* b, float* c, int size) {
    cuda_tensor_add(a, b, c, size);
    cuda_check_error("tensor_add");
}

void aicraft_cuda_mul(const float* a, const float* b, float* c, int size) {
    cuda_tensor_mul(a, b, c, size);
    cuda_check_error("tensor_mul");
}

void aicraft_cuda_scalar_mul(const float* a, float scalar, float* c, int size) {
    cuda_tensor_scalar_mul(a, scalar, c, size);
    cuda_check_error("tensor_scalar_mul");
}

void aicraft_cuda_fill(float* data, float value, int size) {
    cuda_fill(data, value, size);
    cuda_check_error("tensor_fill");
}

// Matrix operations
void aicraft_cuda_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    cuda_gemm_auto(A, B, C, M, N, K);
}

// Activation functions
void aicraft_cuda_relu(const float* input, float* output, int size) {
    cuda_relu(input, output, size);
    cuda_check_error("relu");
}

void aicraft_cuda_gelu(const float* input, float* output, int size) {
    cuda_gelu(input, output, size);
    cuda_check_error("gelu");
}

void aicraft_cuda_sigmoid(const float* input, float* output, int size) {
    cuda_sigmoid(input, output, size);
    cuda_check_error("sigmoid");
}

void aicraft_cuda_softmax(const float* input, float* output, int rows, int cols) {
    cuda_softmax(input, output, rows, cols);
    cuda_check_error("softmax");
}

// System information
bool aicraft_cuda_available() {
    return cuda_is_available();
}

void aicraft_cuda_sync() {
    cuda_synchronize();
}

size_t aicraft_cuda_memory_usage() {
    return cuda_get_memory_usage();
}

void aicraft_cuda_print_stats() {
    cuda_print_memory_stats();
}

void aicraft_cuda_device_info() {
    cuda_device_info();
}

// High-level tensor operations with automatic optimization
void aicraft_cuda_dense_forward(const float* input, const float* weights, const float* bias,
                                float* output, int batch_size, int input_size, int output_size) {
    // Compute: output = input * weights^T + bias
    // input: [batch_size, input_size]
    // weights: [output_size, input_size] (stored row-major)
    // bias: [output_size]
    // output: [batch_size, output_size]
    
    // Matrix multiplication: output = input * weights^T
    cuda_gemm_blas(weights, input, output, 
                   output_size, batch_size, input_size, 
                   1.0f, 0.0f);
    
    // Add bias (broadcast)
    for (int b = 0; b < batch_size; b++) {
        cuda_tensor_add(output + b * output_size, bias, 
                       output + b * output_size, output_size);
    }
    
    cuda_check_error("dense_forward");
}

// Optimized batch operations
void aicraft_cuda_batch_relu(const float* input, float* output, int batch_size, int size) {
    cuda_relu(input, output, batch_size * size);
    cuda_check_error("batch_relu");
}

void aicraft_cuda_batch_softmax(const float* input, float* output, int batch_size, int size) {
    cuda_softmax(input, output, batch_size, size);
    cuda_check_error("batch_softmax");
}

// Memory optimization utilities
void aicraft_cuda_prefetch(const float* data, size_t size) {
    if (cuda_is_available()) {
        cudaMemPrefetchAsync(data, size, 0, 0);
    }
}

void aicraft_cuda_zero_async(float* data, int size) {
    if (cuda_is_available()) {
        cudaMemsetAsync(data, 0, size * sizeof(float), 0);
    }
}

// Performance monitoring
void aicraft_cuda_start_timing() {
    // Could implement CUDA events here for precise timing
    cuda_synchronize();
}

float aicraft_cuda_end_timing() {
    cuda_synchronize();
    return 0.0f; // Placeholder - could return actual timing
}

// Advanced operations for neural networks
void aicraft_cuda_layer_norm(const float* input, const float* gamma, const float* beta,
                             float* output, int batch_size, int features, float eps) {
    // Placeholder for layer normalization
    // For now, just copy input to output
    cuda_memcpy_device_to_device(output, input, batch_size * features * sizeof(float));
    cuda_check_error("layer_norm");
}

void aicraft_cuda_dropout(const float* input, float* output, const float* mask,
                          int size, float keep_prob) {
    // Apply dropout mask
    cuda_tensor_mul(input, mask, output, size);
    
    // Scale by 1/keep_prob during training
    if (keep_prob > 0.0f && keep_prob < 1.0f) {
        cuda_tensor_scalar_mul(output, 1.0f / keep_prob, output, size);
    }
    
    cuda_check_error("dropout");
}

// Gradient operations for backpropagation
void aicraft_cuda_compute_gradients(const float* predictions, const float* targets,
                                    float* gradients, int batch_size, int num_classes) {
    // Compute gradients for cross-entropy loss
    // grad = (predictions - targets) / batch_size
    
    // First subtract: grad = predictions - targets
    for (int i = 0; i < batch_size * num_classes; i += num_classes) {
        cuda_tensor_add(predictions + i, targets + i, gradients + i, num_classes);
    }
    
    // Then scale by 1/batch_size
    cuda_tensor_scalar_mul(gradients, 1.0f / batch_size, gradients, 
                          batch_size * num_classes);
    
    cuda_check_error("compute_gradients");
}

} // extern "C"
