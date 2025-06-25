#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cudnn.h>
#include "cuda_kernels.h"

// Mixed precision kernels
extern "C" {

// FP16 GEMM with automatic mixed precision
void aicraft_cuda_gemm_mixed_precision(const void* A, const void* B, void* C,
                                      int M, int N, int K,
                                      float alpha, float beta,
                                      cudaDataType_t A_type,
                                      cudaDataType_t B_type,
                                      cudaDataType_t C_type,
                                      cudaDataType_t compute_type,
                                      cublasLtHandle_t ltHandle);

// Tensor Core optimized GEMM for modern GPUs
void aicraft_cuda_tensor_core_gemm(const half* A, const half* B, float* C,
                                  int M, int N, int K,
                                  float alpha, float beta);

// Dynamic loss scaling for mixed precision
__global__ void aicraft_cuda_check_finite_kernel(const float* input, bool* is_finite, int size);
__global__ void aicraft_cuda_scale_gradients_kernel(float* gradients, float scale, int size);
__global__ void aicraft_cuda_unscale_gradients_kernel(float* gradients, float scale, int size);

// Advanced activation functions with FP16 support
__global__ void aicraft_cuda_gelu_fp16_kernel(half* output, const half* input, int size);
__global__ void aicraft_cuda_swish_kernel(float* output, const float* input, int size);
__global__ void aicraft_cuda_mish_kernel(float* output, const float* input, int size);

// Fused operations for better performance
__global__ void aicraft_cuda_fused_linear_gelu_kernel(float* output, const float* input,
                                                     const float* weight, const float* bias,
                                                     int batch_size, int input_size, int output_size);

__global__ void aicraft_cuda_fused_layernorm_gelu_kernel(float* output, const float* input,
                                                        const float* gamma, const float* beta,
                                                        float eps, int batch_size, int hidden_size);

// Advanced optimizer kernels
__global__ void aicraft_cuda_adabound_kernel(float* weights, const float* gradients,
                                            float* m, float* v,
                                            float lr, float beta1, float beta2, float eps,
                                            float final_lr, float gamma, int t, int size);

__global__ void aicraft_cuda_radam_kernel(float* weights, const float* gradients,
                                         float* m, float* v,
                                         float lr, float beta1, float beta2, float eps,
                                         int t, int size);

__global__ void aicraft_cuda_lamb_kernel(float* weights, const float* gradients,
                                        float* m, float* v,
                                        float lr, float beta1, float beta2, float eps,
                                        float weight_decay, int t, int size);

// Memory-efficient operations
__global__ void aicraft_cuda_inplace_relu_kernel(float* data, int size);
__global__ void aicraft_cuda_inplace_dropout_kernel(float* data, const float* mask, 
                                                   float keep_prob, int size);

// Gradient accumulation and checkpointing
__global__ void aicraft_cuda_accumulate_gradients_kernel(float* accumulated_grad,
                                                        const float* new_grad,
                                                        float scale, int size);

// Quantization kernels for INT8 inference
__global__ void aicraft_cuda_quantize_fp32_to_int8_kernel(int8_t* output, const float* input,
                                                         float scale, int8_t zero_point, int size);

__global__ void aicraft_cuda_dequantize_int8_to_fp32_kernel(float* output, const int8_t* input,
                                                           float scale, int8_t zero_point, int size);

// Multi-head attention kernels (for Transformer support)
__global__ void aicraft_cuda_scaled_dot_product_attention_kernel(float* output,
                                                                const float* query,
                                                                const float* key,
                                                                const float* value,
                                                                int batch_size, int seq_len,
                                                                int num_heads, int head_dim);

// Batch operations
__global__ void aicraft_cuda_batch_norm_forward_kernel(float* output, const float* input,
                                                      const float* running_mean,
                                                      const float* running_var,
                                                      const float* gamma, const float* beta,
                                                      float eps, int batch_size, int channels);

__global__ void aicraft_cuda_group_norm_kernel(float* output, const float* input,
                                              const float* gamma, const float* beta,
                                              int batch_size, int channels, int groups,
                                              int spatial_size, float eps);

// Performance monitoring
__global__ void aicraft_cuda_compute_flops_kernel(const float* input, int size, 
                                                 unsigned long long* flop_count);

}

// Template kernels for different data types
template<typename T>
__global__ void aicraft_cuda_elementwise_add_kernel(T* output, const T* a, const T* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = a[idx] + b[idx];
    }
}

template<typename T>
__global__ void aicraft_cuda_elementwise_mul_kernel(T* output, const T* a, const T* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = a[idx] * b[idx];
    }
}

// Auto-tuning support
struct KernelConfig {
    dim3 block_size;
    dim3 grid_size;
    int shared_memory;
    cudaStream_t stream;
};

// Kernel auto-tuning database
class KernelAutoTuner {
private:
    std::unordered_map<std::string, KernelConfig> config_cache;
    
public:
    KernelConfig get_optimal_config(const std::string& kernel_name, 
                                   int problem_size, 
                                   int device_id);
    
    void benchmark_kernel(const std::string& kernel_name,
                         const std::vector<KernelConfig>& configs,
                         int problem_size);
    
    void save_cache(const std::string& filename);
    void load_cache(const std::string& filename);
};

// Graph optimization utilities
class GraphOptimizer {
public:
    static void fuse_consecutive_ops(std::vector<Operation>& ops);
    static void eliminate_dead_code(std::vector<Operation>& ops);
    static void optimize_memory_layout(std::vector<Tensor>& tensors);
    static void schedule_operations(std::vector<Operation>& ops, int num_streams);
};

// Performance profiler integration
class CudaProfiler {
private:
    cudaEvent_t start_event, stop_event;
    std::unordered_map<std::string, float> timing_data;
    
public:
    void start_timing(const std::string& operation_name);
    void end_timing(const std::string& operation_name);
    void print_summary();
    void export_to_json(const std::string& filename);
};

#endif // CUDA_AVAILABLE
