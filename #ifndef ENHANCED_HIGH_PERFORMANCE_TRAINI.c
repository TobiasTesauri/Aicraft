#ifndef ENHANCED_HIGH_PERFORMANCE_TRAINING_H
#define ENHANCED_HIGH_PERFORMANCE_TRAINING_H

#include <immintrin.h>  // AVX/AVX2/AVX-512
#include <omp.h>        // OpenMP
#include <cblas.h>      // BLAS
#include <stdint.h>
#include <stdbool.h>
#include <numa.h>       // NUMA support

// Enhanced performance configurations
#define SIMD_WIDTH 8        // AVX-256 for 8 floats
#define SIMD_WIDTH_512 16   // AVX-512 for 16 floats
#define CACHE_LINE_SIZE 64
#define PREFETCH_DISTANCE 8
#define MAX_THREADS 64
#define MEMORY_POOL_ALIGNMENT 4096  // Page alignment

// Auto-detect SIMD capabilities
#define SIMD_ALIGN 64  // Support up to AVX-512
#define ALIGNED_MALLOC(size) aligned_alloc(SIMD_ALIGN, (size + SIMD_ALIGN - 1) & ~(SIMD_ALIGN - 1))

// NUMA-aware memory allocation
#define NUMA_MALLOC(size, node) numa_alloc_onnode(size, node)

// CPU feature detection
typedef struct {
    bool has_avx;
    bool has_avx2;
    bool has_avx512f;
    bool has_fma;
    int num_numa_nodes;
    int cache_line_size;
} CPUFeatures;

// Enhanced tensor with NUMA awareness
typedef struct {
    float* data _attribute_((aligned(64)));
    int rows, cols;
    int stride;
    bool transposed;
    int ref_count;
    int numa_node;          // NUMA node affinity
    bool is_quantized;      // Support for INT8 quantization
    float scale;            // Quantization scale
    int8_t* quantized_data; // Quantized data
} EnhancedTensor;

// Quantization support
typedef enum {
    QUANT_NONE,
    QUANT_INT8,
    QUANT_INT4,
    QUANT_DYNAMIC
} QuantizationType;

// Advanced fused operations
typedef enum {
    FUSED_DENSE_RELU,
    FUSED_DENSE_GELU,
    FUSED_DENSE_SOFTMAX,
    FUSED_MATMUL_ADD_RELU,
    FUSED_BACKWARD_DENSE_RELU,
    FUSED_DENSE_DROPOUT_RELU,     // New
    FUSED_LAYERNORM_LINEAR,       // New
    FUSED_ATTENTION_BLOCK,        // New
    FUSED_RESIDUAL_BLOCK          // New
} EnhancedFusedOpType;

// SIMD function pointers (runtime dispatch)
typedef struct {
    void (vector_add)(const float a, const float* b, float* c, int size);
    void (vector_mul)(const float a, float scalar, float* b, int size);
    void (matrix_multiply)(const float A, const float* B, float* C, int M, int N, int K);
    void (relu_forward)(const float input, float* output, int size);
    void (relu_backward)(const float input, const float* grad_output, float* grad_input, int size);
    void (gelu_forward)(const float input, float* output, int size);
    void (softmax)(const float input, float* output, int size);
} SIMDKernels;

// Thread-local storage with NUMA awareness
typedef struct {
    float* workspace;
    int workspace_size;
    int thread_id;
    int numa_node;
    CPUFeatures cpu_features;
} EnhancedThreadContext;

// Memory pool with NUMA support
typedef struct {
    float* pool_data;
    size_t pool_size;
    size_t pool_used;
    int numa_node;
    void** free_blocks;
    size_t* block_sizes;
    int num_free_blocks;
} NUMAMemoryPool;

// Enhanced layer with more optimizations
typedef struct EnhancedLayer {
    LayerType type;
    EnhancedFusedOpType fused_op;
    QuantizationType quant_type;
    
    // Optimized tensor layout
    EnhancedTensor weights;
    EnhancedTensor bias;
    EnhancedTensor grad_weights;
    EnhancedTensor grad_bias;
    
    // Intermediate activations (for backward pass)
    EnhancedTensor activations;
    EnhancedTensor pre_activations;
    
    // Enhanced Adam optimizer with bias correction
    struct {
        EnhancedTensor m_w, v_w, m_b, v_b;
        float beta1, beta2, epsilon;
        int t;  // Time step
        float lr_schedule;  // Learning rate schedule
    } adam_state;
    
    // Dropout state
    struct {
        float dropout_rate;
        uint32_t* dropout_mask;
        bool training;
    } dropout_state;
    
    // Layer normalization parameters
    struct {
        EnhancedTensor gamma, beta;
        EnhancedTensor mean, var;
        float eps;
    } layernorm_state;
    
    // SIMD kernels (runtime dispatch)
    SIMDKernels kernels;
    
    // Thread workspaces
    EnhancedThreadContext* thread_contexts;
    
    // Kernel function pointers
    void (forward_kernel)(struct EnhancedLayer layer, EnhancedTensor* input, EnhancedTensor* output);
    void (backward_kernel)(struct EnhancedLayer layer, EnhancedTensor* grad_out, EnhancedTensor* grad_in);
    
    struct EnhancedLayer* next;
} EnhancedLayer;

// Enhanced model with more features
typedef struct {
    char name[64];
    EnhancedLayer* layers;
    int num_layers;
    
    // NUMA-aware memory pools
    NUMAMemoryPool* numa_pools;
    int num_numa_nodes;
    
    // Parallel execution context
    int num_threads;
    EnhancedThreadContext* global_contexts;
    
    // Batch processing optimization
    int optimal_batch_size;
    EnhancedTensor* batch_workspace;
    
    // Mixed precision training
    bool use_mixed_precision;
    float loss_scale;
    
    // Gradient clipping
    float gradient_clip_norm;
    
    // Learning rate scheduling
    struct {
        float initial_lr;
        float current_lr;
        int warmup_steps;
        int total_steps;
        float decay_rate;
    } lr_schedule;
    
    // Performance counters
    double forward_time;
    double backward_time;
    double update_time;
    uint64_t operations_count;
    uint64_t memory_bandwidth;
    
    // CPU features
    CPUFeatures cpu_features;
    SIMDKernels global_kernels;
    
} EnhancedModel;

// Core API
EnhancedModel* enhanced_model_create(const char* name, int num_threads);
void enhanced_model_free(EnhancedModel* model);
void enhanced_model_add_fused_layer(EnhancedModel* model, EnhancedFusedOpType fused_op, 
                                   int input_size, int output_size);
void enhanced_model_compile_optimized(EnhancedModel* model, float learning_rate);

// Training functions
void enhanced_model_train_optimized(EnhancedModel* model, 
                                   float* input_data, float* target_data,
                                   int num_samples, int input_dim, int output_dim,
                                   int epochs, int batch_size);

// CPU feature detection
CPUFeatures detect_cpu_features(void);
void initialize_simd_kernels(SIMDKernels* kernels, CPUFeatures features);

// AVX-512 optimized operations (when available)
void avx512_matrix_multiply(const float* A, const float* B, float* C, int M, int N, int K);
void avx512_vector_add(const float* a, const float* b, float* c, int size);
void avx512_relu_forward(const float* input, float* output, int size);

// Enhanced SIMD operations with runtime dispatch
void optimized_matrix_multiply(const float* A, const float* B, float* C, 
                              int M, int N, int K, const SIMDKernels* kernels);
void optimized_vector_add(const float* a, const float* b, float* c, int size, 
                         const SIMDKernels* kernels);

// Advanced fused kernels
void fused_dense_dropout_relu_forward(EnhancedLayer* layer, EnhancedTensor* input, 
                                     EnhancedTensor* output);
void fused_layernorm_linear_forward(EnhancedLayer* layer, EnhancedTensor* input, 
                                   EnhancedTensor* output);
void fused_attention_block_forward(EnhancedLayer* layer, EnhancedTensor* input, 
                                  EnhancedTensor* output);

// Quantization functions
void quantize_tensor_int8(EnhancedTensor* tensor);
void dequantize_tensor_int8(EnhancedTensor* tensor);
void quantized_matrix_multiply(const EnhancedTensor* A, const EnhancedTensor* B, 
                              EnhancedTensor* C);

// NUMA-aware memory management
NUMAMemoryPool* create_numa_pool(size_t size, int numa_node);
void* numa_pool_alloc(NUMAMemoryPool* pool, size_t size);
void numa_pool_free(NUMAMemoryPool* pool, void* ptr);

// Advanced optimizations
void optimize_memory_layout_numa(EnhancedTensor* tensor, int numa_node);
void prefetch_data_advanced(const void* addr, int size, int prefetch_type);
void warm_cache_numa(EnhancedModel* model);

// Learning rate scheduling
void update_learning_rate(EnhancedModel* model, int current_step);
void apply_gradient_clipping(EnhancedModel* model, float max_norm);

// Mixed precision training
void convert_to_fp16(const float* input, uint16_t* output, int size);
void convert_to_fp32(const uint16_t* input, float* output, int size);
void scale_gradients(EnhancedModel* model, float scale);

// Performance profiling and debugging
void start_profiling(void);
double get_profiling_time(void);
void print_detailed_performance_stats(EnhancedModel* model);
void analyze_memory_bandwidth(EnhancedModel* model);
void print_numa_topology(void);

// Validation and testing
bool validate_model_correctness(EnhancedModel* model, float* test_input, 
                               float* expected_output, float tolerance);
void benchmark_kernels(const SIMDKernels* kernels, int size);

#endif // ENHANCED_HIGH_PERFORMANCE_TRAINING_H

// Key implementation snippets for the enhanced features

// CPU feature detection
CPUFeatures detect_cpu_features(void) {
    CPUFeatures features = {0};
    
    // CPUID feature detection
    uint32_t eax, ebx, ecx, edx;
    
    // Check for AVX
    _asm_ _volatile_(
        "cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(1)
    );
    features.has_avx = (ecx & (1 << 28)) != 0;
    features.has_fma = (ecx & (1 << 12)) != 0;
    
    // Check for AVX2
    _asm_ _volatile_(
        "cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(7), "c"(0)
    );
    features.has_avx2 = (ebx & (1 << 5)) != 0;
    features.has_avx512f = (ebx & (1 << 16)) != 0;
    
    // NUMA detection
    features.num_numa_nodes = numa_available() ? numa_max_node() + 1 : 1;
    features.cache_line_size = CACHE_LINE_SIZE;
    
    return features;
}

// Runtime SIMD kernel dispatch
void initialize_simd_kernels(SIMDKernels* kernels, CPUFeatures features) {
    if (features.has_avx512f) {
        kernels->vector_add = avx512_vector_add;
        kernels->matrix_multiply = avx512_matrix_multiply;
        kernels->relu_forward = avx512_relu_forward;
    } else if (features.has_avx2) {
        kernels->vector_add = simd_vector_add;
        kernels->matrix_multiply = simd_matrix_multiply;
        kernels->relu_forward = simd_relu_forward;
    } else {
        // Fallback to scalar implementations
        kernels->vector_add = scalar_vector_add;
        kernels->matrix_multiply = scalar_matrix_multiply;
        kernels->relu_forward = scalar_relu_forward;
    }
}

// AVX-512 optimized vector addition (when available)
void avx512_vector_add(const float* a, const float* b, float* c, int size) {
    int simd_size = size & ~15;  // Round down to multiple of 16
    
    #pragma omp parallel for
    for (int i = 0; i < simd_size; i += 16) {
        __m512 va = _mm512_load_ps(&a[i]);
        __m512 vb = _mm512_load_ps(&b[i]);
        __m512 vc = _mm512_add_ps(va, vb);
        _mm512_store_ps(&c[i], vc);
    }
    
    // Handle remainder
    for (int i = simd_size; i < size; i++) {
        c[i] = a[i] + b[i];
    }
}

// Enhanced GELU activation with SIMD
void simd_gelu_forward(const float* input, float* output, int size) {
    const float sqrt_2_pi = 0.7978845608f;  // sqrt(2/π)
    const float coeff = 0.044715f;
    
    int simd_size = size & ~7;
    __m256 v_sqrt_2_pi = _mm256_set1_ps(sqrt_2_pi);
    __m256 v_coeff = _mm256_set1_ps(coeff);
    __m256 v_half = _mm256_set1_ps(0.5f);
    __m256 v_one = _mm256_set1_ps(1.0f);
    
    #pragma omp parallel for
    for (int i = 0; i < simd_size; i += 8) {
        __m256 x = _mm256_load_ps(&input[i]);
        
        // Compute x^3
        __m256 x_squared = _mm256_mul_ps(x, x);
        __m256 x_cubed = _mm256_mul_ps(x_squared, x);
        
        // Compute tanh argument: sqrt(2/π) * (x + 0.044715 * x^3)
        __m256 tanh_arg = _mm256_fmadd_ps(v_coeff, x_cubed, x);
        tanh_arg = _mm256_mul_ps(v_sqrt_2_pi, tanh_arg);
        
        // Approximate tanh using rational approximation
        // tanh(x) ≈ x * (27 + x^2) / (27 + 9*x^2) for |x| < 1
        __m256 tanh_result = fast_tanh_approx(tanh_arg);
        
        // GELU: 0.5 * x * (1 + tanh(...))
        __m256 result = _mm256_mul_ps(v_half, x);
        result = _mm256_mul_ps(result, _mm256_add_ps(v_one, tanh_result));
        
        _mm256_store_ps(&output[i], result);
    }
    
    // Handle remainder with scalar code
    for (int i = simd_size; i < size; i++) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + tanhf(sqrt_2_pi * (x + coeff * x * x * x)));
    }
}

// Fast tanh approximation for SIMD
static inline _m256 fast_tanh_approx(_m256 x) {
    // Rational approximation: tanh(x) ≈ x * (27 + x^2) / (27 + 9*x^2)
    __m256 x_squared = _mm256_mul_ps(x, x);
    __m256 numerator = _mm256_fmadd_ps(x_squared, _mm256_set1_ps(27.0f), _mm256_set1_ps(27.0f));
    __m256 denominator = _mm256_fmadd_ps(x_squared, _mm256_set1_ps(9.0f), _mm256_set1_ps(27.0f));
    
    return _mm256_mul_ps(x, _mm256_div_ps(numerator, denominator));
}

// NUMA-aware memory pool allocation
NUMAMemoryPool* create_numa_pool(size_t size, int numa_node) {
    NUMAMemoryPool* pool = malloc(sizeof(NUMAMemoryPool));
    if (!pool) return NULL;
    
    // Allocate memory on specific NUMA node
    pool->pool_data = numa_alloc_onnode(size, numa_node);
    if (!pool->pool_data) {
        free(pool);
        return NULL;
    }
    
    pool->pool_size = size;
    pool->pool_used = 0;
    pool->numa_node = numa_node;
    
    // Initialize free block tracking
    pool->num_free_blocks = 0;
    pool->free_blocks = malloc(1024 * sizeof(void*));
    pool->block_sizes = malloc(1024 * sizeof(size_t));
    
    return pool;
}

// Enhanced training loop with all optimizations
void enhanced_model_train_optimized(EnhancedModel* model, 
                                   float* input_data, float* target_data,
                                   int num_samples, int input_dim, int output_dim,
                                   int epochs, int batch_size) {
    
    printf("Enhanced High-Performance Training: %s\n", model->name);
    printf("CPU Features: AVX=%d, AVX2=%d, AVX-512=%d, FMA=%d\n", 
           model->cpu_features.has_avx, model->cpu_features.has_avx2,
           model->cpu_features.has_avx512f, model->cpu_features.has_fma);
    printf("NUMA Nodes: %d, Threads: %d, Batch Size: %d\n",
           model->num_numa_nodes, model->num_threads, batch_size);
    
    // Warm up caches and optimize memory layout
    warm_cache_numa(model);
    
    // Initialize learning rate schedule
    model->lr_schedule.total_steps = epochs * ((num_samples + batch_size - 1) / batch_size);
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        start_profiling();
        
        int num_batches = (num_samples + batch_size - 1) / batch_size;
        double total_loss = 0.0;
        
        #pragma omp parallel for reduction(+:total_loss) schedule(dynamic)
        for (int batch = 0; batch < num_batches; batch++) {
            int thread_id = omp_get_thread_num();
            EnhancedThreadContext* ctx = &model->global_contexts[thread_id];
            
            int start_idx = batch * batch_size;
            int end_idx = (start_idx + batch_size > num_samples) ? num_samples : start_idx + batch_size;
            int current_batch_size = end_idx - start_idx;
            
            // Update learning rate
            int current_step = epoch * num_batches + batch;
            update_learning_rate(model, current_step);
            
            // Forward pass with enhanced kernels
            total_loss += forward_backward_pass_optimized(model, ctx, 
                                                        &input_data[start_idx * input_dim],
                                                        &target_data[start_idx * output_dim],
                                                        current_batch_size, input_dim, output_dim);
        }
        
        // Apply gradient clipping if enabled
        if (model->gradient_clip_norm > 0.0f) {
            apply_gradient_clipping(model, model->gradient_clip_norm);
        }
        
        double epoch_time = get_profiling_time();
        model->forward_time += epoch_time;
        
        printf("Epoch %d/%d - Loss: %.6f - Time: %.3fs - Speed: %.1f samples/sec - LR: %.2e\n", 
               epoch + 1, epochs, total_loss / num_batches, epoch_time,
               num_samples / epoch_time, model->lr_schedule.current_lr);
    }
    
    print_detailed_performance_stats(model);
    analyze_memory_bandwidth(model);
}