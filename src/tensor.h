#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

// Backend detection and selection
typedef enum {
    BACKEND_CPU,
    BACKEND_CUDA,
    BACKEND_AUTO
} BackendType;

// Tensor structure with multi-backend support
typedef struct {
    float* data;
    float* cuda_data;  // GPU memory pointer (NULL if CPU-only)
    int rows;
    int cols;
    int stride;
    bool on_cuda;      // True if data is on GPU
    bool owns_data;    // True if tensor owns its memory
    BackendType backend;
    int ref_count;
    char name[64];     // For debugging
} Tensor;

// Device information
typedef struct {
    bool cuda_available;
    int cuda_device_count;
    int current_device;
    size_t total_memory;
    size_t free_memory;
    char device_name[256];
    int compute_capability_major;
    int compute_capability_minor;
} DeviceInfo;

// Global device state
extern DeviceInfo g_device_info;
extern BackendType g_default_backend;

// === CORE TENSOR FUNCTIONS ===

// Initialization and cleanup
void aicraft_init(void);
void aicraft_cleanup(void);
BackendType aicraft_detect_backend(void);
void aicraft_print_device_info(void);

// Tensor creation and management
Tensor tensor_create(int rows, int cols, BackendType backend);
Tensor tensor_create_from_data(float* data, int rows, int cols, bool copy_data);
Tensor tensor_zeros(int rows, int cols, BackendType backend);
Tensor tensor_ones(int rows, int cols, BackendType backend);
Tensor tensor_random(int rows, int cols, float min_val, float max_val, BackendType backend);
void tensor_free(Tensor tensor);
Tensor tensor_copy(Tensor src);
Tensor tensor_clone(Tensor src);

// Memory management
void tensor_to_cuda(Tensor* tensor);
void tensor_to_cpu(Tensor* tensor);
void tensor_sync_to_cpu(Tensor* tensor);
void tensor_sync_to_cuda(Tensor* tensor);

// Basic operations (automatically dispatch to CPU/CUDA)
Tensor tensor_add(Tensor a, Tensor b);
Tensor tensor_sub(Tensor a, Tensor b);
Tensor tensor_mul(Tensor a, Tensor b);
Tensor tensor_div(Tensor a, Tensor b);
Tensor tensor_scalar_add(Tensor a, float scalar);
Tensor tensor_scalar_mul(Tensor a, float scalar);

// Matrix operations
Tensor tensor_matmul(Tensor a, Tensor b);
Tensor tensor_transpose(Tensor a);
void tensor_fill(Tensor* tensor, float value);
void tensor_zero(Tensor* tensor);

// Activation functions
Tensor tensor_relu(Tensor input);
Tensor tensor_relu_derivative(Tensor input);
Tensor tensor_sigmoid(Tensor input);
Tensor tensor_tanh(Tensor input);
Tensor tensor_softmax(Tensor input);
Tensor tensor_gelu(Tensor input);

// Advanced operations for neural networks
Tensor dense_forward(Tensor input, Tensor weights, Tensor bias);
Tensor dense_backward_input(Tensor grad_output, Tensor weights);
void dense_backward_weights(Tensor input, Tensor grad_output, Tensor* grad_weights, Tensor* grad_bias);

// Utility functions
void tensor_set(Tensor tensor, int row, int col, float value);
float tensor_get(Tensor tensor, int row, int col);
int tensor_argmax(Tensor tensor);
Tensor tensor_sum_axis(Tensor tensor, int axis);
void tensor_elementwise_mul(Tensor* a, Tensor b);
void tensor_copy_row(Tensor* dst, int dst_row, Tensor* src, int src_row);
void tensor_print(Tensor tensor, const char* name);
void tensor_print_info(Tensor tensor);
bool tensor_equal(Tensor a, Tensor b, float tolerance);
float tensor_sum(Tensor tensor);
float tensor_mean(Tensor tensor);
float tensor_max(Tensor tensor);
float tensor_min(Tensor tensor);

// Memory info
size_t tensor_memory_usage(Tensor tensor);
size_t get_gpu_memory_usage(void);
void print_memory_usage(void);

// Error handling
typedef enum {
    TENSOR_SUCCESS = 0,
    TENSOR_ERROR_INVALID_SHAPE,
    TENSOR_ERROR_MEMORY_ALLOCATION,
    TENSOR_ERROR_CUDA_ERROR,
    TENSOR_ERROR_BACKEND_MISMATCH
} TensorError;

const char* tensor_error_string(TensorError error);

// Logging
typedef enum {
    LOG_DEBUG,
    LOG_INFO,
    LOG_WARNING,
    LOG_ERROR
} LogLevel;

void aicraft_log(LogLevel level, const char* format, ...);
void aicraft_set_log_level(LogLevel level);

// Performance profiling
void start_timer(void);
double get_elapsed_time(void);
void print_performance_stats(void);

// Mixed precision training support
typedef enum {
    DTYPE_FLOAT32,
    DTYPE_FLOAT16,
    DTYPE_BFLOAT16,
    DTYPE_INT8,
    DTYPE_INT32
} DataType;

// Precision configuration
typedef struct {
    bool enabled;
    DataType compute_type;
    DataType storage_type;
    float loss_scale;
    float loss_scale_growth_factor;
    int loss_scale_growth_interval;
    int consecutive_unskipped_steps;
    bool dynamic_loss_scaling;
} MixedPrecisionConfig;

// Advanced tensor with mixed precision
typedef struct {
    float* data;
    void* cuda_data;  // Can be float16*, float32*, etc.
    DataType dtype;
    int rows, cols;
    bool on_cuda;
    bool requires_grad;
    struct Tensor* grad;  // Gradient tensor
    
    // Memory optimization
    bool is_view;
    struct Tensor* base_tensor;
    int view_offset;
    
    // Graph node information
    int node_id;
    char operation[32];
    struct Tensor** inputs;
    int num_inputs;
} TensorV2;

// Gradient accumulation and checkpointing
typedef struct {
    bool enabled;
    int accumulation_steps;
    int current_step;
    bool checkpoint_activations;
    size_t memory_limit;
    bool auto_mixed_precision;
} GradientConfig;

#endif // TENSOR_H
