#ifndef AICRAFT_ADVANCED_H
#define AICRAFT_ADVANCED_H

#include "tensor.h"
#include "training.h"

// === FORWARD DECLARATIONS ===
typedef struct QuantizedTensor QuantizedTensor;
typedef struct GraphOptimizer GraphOptimizer;
typedef struct ProfilerState ProfilerState;

// === QUANTIZATION INTERFACE ===

typedef struct {
    float scale;
    int8_t zero_point;
    int8_t qmin;
    int8_t qmax;
} QuantizationParams;

typedef struct QuantizedTensor {
    int8_t* data;
    int8_t* cuda_data;
    int rows, cols;
    bool on_cuda;
    QuantizationParams qparams;
    char name[64];
} QuantizedTensor;

// Quantization functions
QuantizationParams compute_quantization_params(float* data, int size, int8_t qmin, int8_t qmax);
QuantizedTensor quantize_tensor(Tensor tensor, int8_t qmin, int8_t qmax);
Tensor dequantize_tensor(QuantizedTensor qtensor);
void quantized_tensor_free(QuantizedTensor qtensor);

// Quantized operations
QuantizedTensor quantized_matmul(QuantizedTensor a, QuantizedTensor b);
QuantizedTensor quantized_add(QuantizedTensor a, QuantizedTensor b);
void quantized_relu_inplace(QuantizedTensor* tensor);

// Model quantization
typedef struct {
    bool post_training_quantization;
    bool quantization_aware_training;
    float calibration_data_fraction;
    int num_calibration_samples;
    bool symmetric_quantization;
    bool per_channel_quantization;
} QuantizationConfig;

void quantize_model(Model* model, QuantizationConfig config);
void model_inference_int8(Model* model, Tensor input, Tensor* output);

// === GRAPH OPTIMIZATION INTERFACE ===

typedef enum {
    OP_MATMUL,
    OP_ADD,
    OP_RELU,
    OP_SIGMOID,
    OP_SOFTMAX,
    OP_LAYERNORM,
    OP_DROPOUT,
    OP_GELU,
    OP_FUSED_LINEAR_RELU,
    OP_FUSED_LINEAR_GELU,
    OP_FUSED_LAYERNORM_GELU,
    OP_FUSED_ATTENTION,
    OP_MEMORY_COPY
} OperationType;

typedef struct Operation {
    OperationType type;
    char name[64];
    int* input_ids;
    int num_inputs;
    int output_id;
    
    union {
        struct { int M, N, K; } matmul;
        struct { float dropout_rate; } dropout;
        struct { float eps; } layernorm;
        struct { int num_heads; int head_dim; } attention;
    } params;
    
    double execution_time;
    size_t memory_usage;
    bool can_fuse;
    
    struct Operation* next;
} Operation;

typedef struct GraphOptimizer {
    Operation* operations;
    int num_operations;
    int* fusion_groups;
    int num_fusion_groups;
    bool optimization_enabled;
    
    // Performance metrics
    double original_time;
    double optimized_time;
    size_t memory_saved;
} GraphOptimizer;

// Graph optimization functions
GraphOptimizer* create_graph_optimizer(void);
void add_operation(GraphOptimizer* optimizer, OperationType type, const char* name);
void optimize_graph(GraphOptimizer* optimizer);
void execute_optimized_graph(GraphOptimizer* optimizer, Tensor* inputs, Tensor* outputs);
void free_graph_optimizer(GraphOptimizer* optimizer);

// Tensor fusion operations
Tensor fused_linear_activation(Tensor input, Tensor weights, Tensor bias, ActivationFunction activation);
Tensor fused_layernorm_activation(Tensor input, Tensor gamma, Tensor beta, float eps, ActivationFunction activation);
Tensor fused_attention(Tensor query, Tensor key, Tensor value, int num_heads);

// === ADVANCED OPTIMIZERS INTERFACE ===

// Advanced optimizer functions
void adabound_update(Tensor* weights, Tensor* gradients, Tensor* m, Tensor* v,
                    float lr, float beta1, float beta2, float eps,
                    float final_lr, float gamma, int t, int size);

void radam_update(Tensor* weights, Tensor* gradients, Tensor* m, Tensor* v,
                 float lr, float beta1, float beta2, float eps, int t, int size);

void lamb_update(Tensor* weights, Tensor* gradients, Tensor* m, Tensor* v,
                float lr, float beta1, float beta2, float eps, 
                float weight_decay, int t, int size);

void yogi_update(Tensor* weights, Tensor* gradients, Tensor* m, Tensor* v,
                float lr, float beta1, float beta2, float eps, int t, int size);

// Gradient clipping
void clip_gradients_by_norm(Tensor* gradients, int num_tensors, float max_norm);
void clip_gradients_by_value(Tensor* gradients, int num_tensors, float min_val, float max_val);

// Exponential Moving Average
typedef struct {
    bool enabled;
    float decay;
    Tensor* shadow_weights;
    int num_weights;
    int step_count;
} EMAState;

EMAState* create_ema_state(int num_weights, float decay);
void update_ema(EMAState* ema, Tensor* weights, int num_weights);
void apply_ema(EMAState* ema, Tensor* weights, int num_weights);
void free_ema_state(EMAState* ema);

// Learning rate scheduling
float cosine_annealing_lr(float initial_lr, float min_lr, int current_step, int total_steps);
float polynomial_decay_lr(float initial_lr, float final_lr, int current_step, int total_steps, float power);
float exponential_decay_lr(float initial_lr, float decay_rate, int current_step, int decay_steps);
float warm_restart_lr(float initial_lr, int current_step, int restart_period, float t_mult);

// === MIXED PRECISION INTERFACE ===

// Mixed precision training
typedef struct {
    bool enabled;
    DataType compute_type;
    DataType storage_type;
    float loss_scale;
    float loss_scale_growth_factor;
    int loss_scale_growth_interval;
    int consecutive_unskipped_steps;
    bool dynamic_loss_scaling;
    
    // Overflow detection
    bool overflow_detected;
    int overflow_count;
    
    // Performance tracking
    double fp16_time;
    double fp32_time;
    size_t memory_saved;
} MixedPrecisionState;

MixedPrecisionState* create_mixed_precision_state(MixedPrecisionConfig config);
void scale_loss(MixedPrecisionState* state, Tensor* loss);
bool check_overflow(MixedPrecisionState* state, Tensor* gradients, int num_tensors);
void update_loss_scale(MixedPrecisionState* state, bool overflow);
void free_mixed_precision_state(MixedPrecisionState* state);

// Mixed precision operations
Tensor mixed_precision_matmul(Tensor a, Tensor b, DataType compute_type);
Tensor mixed_precision_forward(Tensor input, Tensor weights, Tensor bias, DataType compute_type);

// === PROFILER INTERFACE ===

typedef struct {
    char name[128];
    double start_time;
    double total_time;
    int call_count;
    size_t memory_used;
    size_t peak_memory;
} ProfilerEntry;

typedef struct ProfilerState {
    ProfilerEntry* entries;
    int num_entries;
    int capacity;
    bool enabled;
    double total_time;
    size_t total_memory;
    
    // TensorBoard integration
    bool tensorboard_enabled;
    char log_dir[256];
    FILE* log_file;
} ProfilerState;

// Profiler functions
ProfilerState* create_profiler(const char* log_dir);
void profiler_start(ProfilerState* profiler, const char* name);
void profiler_end(ProfilerState* profiler, const char* name);
void profiler_log_memory(ProfilerState* profiler, const char* name, size_t memory);
void profiler_print_report(ProfilerState* profiler);
void profiler_save_tensorboard(ProfilerState* profiler, int step);
void free_profiler(ProfilerState* profiler);

// Profiler macros
#define PROFILE_START(profiler, name) profiler_start(profiler, name)
#define PROFILE_END(profiler, name) profiler_end(profiler, name)
#define PROFILE_SCOPE(profiler, name) for(int _i = (profiler_start(profiler, name), 0); _i < 1; _i++, profiler_end(profiler, name))

// === BENCHMARK INTERFACE ===

typedef struct {
    char name[128];
    double aicraft_time;
    double pytorch_time;
    double speedup;
    size_t aicraft_memory;
    size_t pytorch_memory;
    float accuracy_diff;
    bool passed;
} BenchmarkResult;

typedef struct {
    BenchmarkResult* results;
    int num_results;
    int capacity;
    double total_speedup;
    size_t total_memory_saved;
    int passed_tests;
} BenchmarkSuite;

// Benchmark functions
BenchmarkSuite* create_benchmark_suite(void);
void benchmark_gemm(BenchmarkSuite* suite, int M, int N, int K);
void benchmark_activation_functions(BenchmarkSuite* suite);
void benchmark_training_loop(BenchmarkSuite* suite, int batch_size, int epochs);
void benchmark_inference(BenchmarkSuite* suite, Model* model, int batch_size);
void benchmark_print_results(BenchmarkSuite* suite);
void benchmark_save_report(BenchmarkSuite* suite, const char* filename);
void free_benchmark_suite(BenchmarkSuite* suite);

// === ADVANCED TRAINING INTERFACE ===

// Advanced training configuration
typedef struct {
    // Basic training
    int epochs;
    int batch_size;
    float learning_rate;
    LossType loss_type;
    OptimizerV2 optimizer;
    
    // Advanced features
    MixedPrecisionConfig mixed_precision;
    QuantizationConfig quantization;
    GradientConfig gradient_config;
    
    // Regularization
    float weight_decay;
    float dropout_rate;
    bool use_batch_norm;
    bool use_layer_norm;
    
    // Scheduling
    bool use_lr_scheduler;
    bool use_early_stopping;
    int patience;
    float min_delta;
    
    // Monitoring
    bool use_profiler;
    bool use_tensorboard;
    char log_dir[256];
    int log_frequency;
    
    // Optimization
    bool use_graph_optimization;
    bool use_tensor_fusion;
    bool use_auto_tuning;
    
    // Validation
    float validation_split;
    int validation_frequency;
} AdvancedTrainingConfig;

// Advanced training state
typedef struct {
    Model* model;
    AdvancedTrainingConfig config;
    
    // State tracking
    MixedPrecisionState* mixed_precision;
    EMAState* ema_state;
    ProfilerState* profiler;
    GraphOptimizer* graph_optimizer;
    
    // Training metrics
    float* train_losses;
    float* train_accuracies;
    float* val_losses;
    float* val_accuracies;
    float* learning_rates;
    int metrics_size;
    int current_epoch;
    
    // Best model tracking
    float best_val_accuracy;
    int best_epoch;
    Tensor* best_weights;
    int num_best_weights;
    
    // Early stopping
    int patience_counter;
    bool should_stop;
    
    // Performance tracking
    double total_train_time;
    double total_val_time;
    size_t peak_memory_usage;
} AdvancedTrainingState;

// Advanced training functions
AdvancedTrainingState* create_advanced_training_state(Model* model, AdvancedTrainingConfig config);
void advanced_train_model(AdvancedTrainingState* state, Tensor* train_inputs, Tensor* train_targets, 
                         int num_train_samples, Tensor* val_inputs, Tensor* val_targets, int num_val_samples);
void advanced_save_checkpoint(AdvancedTrainingState* state, const char* filename);
void advanced_load_checkpoint(AdvancedTrainingState* state, const char* filename);
void free_advanced_training_state(AdvancedTrainingState* state);

// === UTILITY FUNCTIONS ===

// System utilities
void print_system_info(void);
void print_cuda_info(void);
void print_memory_info(void);

// Model utilities
void print_model_summary(Model* model);
void save_model_onnx(Model* model, const char* filename);
void load_model_onnx(Model* model, const char* filename);

// Data utilities
void shuffle_dataset(Tensor* inputs, Tensor* targets, int num_samples);
void normalize_dataset(Tensor* inputs, int num_samples);
void split_dataset(Tensor* inputs, Tensor* targets, int num_samples, float split_ratio,
                  Tensor** train_inputs, Tensor** train_targets, int* num_train,
                  Tensor** val_inputs, Tensor** val_targets, int* num_val);

#endif // AICRAFT_ADVANCED_H
