#ifndef TRAINING_H
#define TRAINING_H

#include "tensor.h"

// Forward declarations
typedef struct Layer Layer;
typedef struct Model Model;

// Loss functions
typedef enum {
    LOSS_MSE,
    LOSS_CROSSENTROPY,
    LOSS_BINARY_CROSSENTROPY,
    LOSS_HUBER
} LossType;

// Optimizers
typedef enum {
    OPT_SGD,
    OPT_ADAM,
    OPT_ADAMW,
    OPT_RMSPROP,
    OPT_ADABOUND,    // AdaBound - adaptive bounds for learning rates
    OPT_RADAM,       // Rectified Adam
    OPT_LAMB,        // Layer-wise Adaptive Moments for Batch training
    OPT_RANGER,      // RAdam + Lookahead
    OPT_YOGI,        // Yet Another Gradient-based Optimizer
    OPT_ADADELTA,    // Adaptive Delta
    OPT_NADAM        // Nesterov Adam
} OptimizerType;

// Optimizer state
typedef struct {
    OptimizerType type;
    float learning_rate;
    float beta1, beta2;     // For Adam/AdamW
    float epsilon;          // For Adam/AdamW  
    float weight_decay;     // For AdamW
    float momentum;         // For SGD with momentum
    int t;                  // Time step for Adam
    
    // Learning rate scheduling
    float initial_lr;
    float decay_rate;
    int decay_steps;
    bool use_cosine_decay;
    bool use_warm_restart;
    int warm_restart_period;
} Optimizer;

// Advanced optimizer configuration
typedef struct {
    OptimizerType type;
    float learning_rate;
    float beta1, beta2;
    float epsilon;
    float weight_decay;
    float momentum;
    int t;  // time step
    
    // AdaBound specific
    float final_lr;
    float gamma;
    
    // Lookahead specific (for Ranger)
    bool use_lookahead;
    float lookahead_alpha;
    int lookahead_k;
    
    // Learning rate scheduling
    float initial_lr;
    float decay_rate;
    int decay_steps;
    bool use_cosine_decay;
    bool use_warm_restart;
    int warm_restart_period;
    
    // Gradient clipping
    bool clip_gradients;
    float max_grad_norm;
    
    // Mixed precision
    MixedPrecisionConfig mixed_precision;
    
    // Advanced features
    bool use_ema;           // Exponential Moving Average
    float ema_decay;
    bool use_swa;           // Stochastic Weight Averaging
    int swa_start_epoch;
    float swa_lr;
} OptimizerV2;

// Layer types
typedef enum {
    LAYER_DENSE,
    LAYER_RELU,
    LAYER_SIGMOID,
    LAYER_TANH,
    LAYER_SOFTMAX,
    LAYER_GELU,
    LAYER_DROPOUT,
    LAYER_BATCH_NORM,
    LAYER_LAYER_NORM
} LayerType;

// Activation function pointers
typedef Tensor (*ActivationFunction)(Tensor input);
typedef Tensor (*ActivationDerivative)(Tensor input);

// Layer structure with enhanced features
struct Layer {
    LayerType type;
    char name[64];
    
    // Dimensions
    int input_size;
    int output_size;
    int batch_size;
    
    // Parameters
    Tensor weights;
    Tensor bias;
    
    // Gradients
    Tensor grad_weights;
    Tensor grad_bias;
    
    // Optimizer states (for Adam/AdamW)
    Tensor m_weights, v_weights;  // Momentum for weights
    Tensor m_bias, v_bias;        // Momentum for bias
    
    // Forward/backward cache
    Tensor input_cache;   // Cached input for backward pass
    Tensor output_cache;  // Cached output
    Tensor pre_activation; // Before activation function
    
    // Activation function
    ActivationFunction activation_fn;
    ActivationDerivative activation_derivative;
    
    // Dropout specific
    struct {
        float dropout_rate;
        Tensor mask;
        bool training;
    } dropout_state;
    
    // Batch normalization specific
    struct {
        Tensor running_mean;
        Tensor running_var;
        Tensor gamma;       // Scale parameter
        Tensor beta;        // Shift parameter
        float momentum;     // For running statistics
        float eps;          // Small constant for numerical stability
        bool training;
    } batchnorm_state;
    
    // Layer normalization specific
    struct {
        Tensor gamma;       // Scale parameter
        Tensor beta;        // Shift parameter
        float eps;          // Small constant
    } layernorm_state;
    
    // Performance metrics
    double forward_time;
    double backward_time;
    
    struct Layer* next;
};

// Enhanced model structure
struct Model {
    char name[64];
    Layer* layers;
    int num_layers;
    LossType loss_type;
    Optimizer optimizer;
    BackendType backend;
    
    // Training state
    bool training_mode;
    int current_epoch;
    int total_epochs;
    
    // Training metrics
    float current_loss;
    float current_accuracy;
    float best_accuracy;
    float validation_loss;
    float validation_accuracy;
    
    // Learning rate scheduling
    float* lr_history;
    int lr_history_size;
    
    // Performance monitoring
    double total_forward_time;
    double total_backward_time;
    double total_training_time;
    
    // Memory usage tracking
    size_t total_parameters;
    size_t memory_usage;
    
    // Callbacks
    void (*on_epoch_start)(Model* model, int epoch);
    void (*on_epoch_end)(Model* model, int epoch, float loss, float accuracy);
    void (*on_batch_end)(Model* model, int batch, float loss);
};

// Training configuration
typedef struct {
    int epochs;
    int batch_size;
    float validation_split;
    bool shuffle;
    bool verbose;
    int print_every;
    
    // Early stopping
    bool use_early_stopping;
    int patience;
    float min_delta;
    
    // Checkpointing
    bool save_checkpoints;
    char checkpoint_dir[256];
    int save_every;
    
    // Data augmentation
    bool use_augmentation;
    float noise_std;
} TrainingConfig;

// === CORE MODEL FUNCTIONS ===

// Model creation and management
Model* model_create(const char* name, LossType loss_type, BackendType backend);
void model_free(Model* model);
void model_summary(Model* model);
void model_print_architecture(Model* model);

// Layer management
void model_add_layer(Model* model, LayerType type, int input_size, int output_size);
void model_add_dense(Model* model, int units, const char* activation);
void model_add_activation(Model* model, LayerType activation_type);
void model_add_dropout(Model* model, float rate);
void model_add_batch_norm(Model* model);
void model_add_layer_norm(Model* model);

// Model compilation
void model_compile(Model* model, OptimizerType opt_type, float learning_rate);
void model_compile_advanced(Model* model, Optimizer* optimizer);

// === TRAINING FUNCTIONS ===

// Forward and backward propagation
Tensor model_forward(Model* model, Tensor input);
void model_backward(Model* model, Tensor predictions, Tensor targets);
void model_update_weights(Model* model);

// Loss and metrics computation
float model_compute_loss(Model* model, Tensor predictions, Tensor targets);
float model_compute_accuracy(Model* model, Tensor predictions, Tensor targets);
float model_compute_precision(Model* model, Tensor predictions, Tensor targets);
float model_compute_recall(Model* model, Tensor predictions, Tensor targets);
float model_compute_f1_score(Model* model, Tensor predictions, Tensor targets);

// Training loops
void model_train_epoch(Model* model, Tensor* inputs, Tensor* targets, int num_samples, TrainingConfig* config);
void model_train(Model* model, Tensor* inputs, Tensor* targets, int num_samples, TrainingConfig* config);
void model_train_with_validation(Model* model, 
                                 Tensor* train_inputs, Tensor* train_targets, int train_samples,
                                 Tensor* val_inputs, Tensor* val_targets, int val_samples,
                                 TrainingConfig* config);

// Evaluation
float model_evaluate(Model* model, Tensor* inputs, Tensor* targets, int num_samples);
Tensor model_predict(Model* model, Tensor input);
void model_predict_batch(Model* model, Tensor* inputs, Tensor* outputs, int batch_size);

// === LAYER OPERATIONS ===

// Layer forward/backward
Tensor layer_forward(Layer* layer, Tensor input);
Tensor layer_backward(Layer* layer, Tensor grad_output);
void layer_update_weights(Layer* layer, Optimizer* opt);

// Layer initialization
void layer_init_weights(Layer* layer, const char* init_method);
void xavier_init(Tensor* tensor);
void he_init(Tensor* tensor);
void random_normal_init(Tensor* tensor, float std);
void zero_init(Tensor* tensor);

// === ACTIVATION FUNCTIONS ===

// Forward activations
Tensor relu_forward(Tensor input);
Tensor sigmoid_forward(Tensor input);
Tensor tanh_forward(Tensor input);
Tensor softmax_forward(Tensor input);
Tensor gelu_forward(Tensor input);
Tensor leaky_relu_forward(Tensor input, float alpha);

// Activation derivatives
Tensor relu_derivative(Tensor input);
Tensor sigmoid_derivative(Tensor input);
Tensor tanh_derivative(Tensor input);
Tensor gelu_derivative(Tensor input);

// === REGULARIZATION ===

// Dropout
Tensor dropout_forward(Tensor input, float rate, bool training);
Tensor dropout_backward(Tensor grad_output, Tensor mask);
void generate_dropout_mask(Tensor* mask, float rate);

// Batch normalization
Tensor batch_norm_forward(Tensor input, Tensor gamma, Tensor beta, 
                         Tensor* running_mean, Tensor* running_var,
                         float momentum, float eps, bool training);
Tensor batch_norm_backward(Tensor grad_output, Tensor input, 
                          Tensor gamma, Tensor* grad_gamma, Tensor* grad_beta);

// Layer normalization
Tensor layer_norm_forward(Tensor input, Tensor gamma, Tensor beta, float eps);
Tensor layer_norm_backward(Tensor grad_output, Tensor input, Tensor gamma,
                          Tensor* grad_gamma, Tensor* grad_beta, float eps);

// === LOSS FUNCTIONS ===

float mse_loss(Tensor predictions, Tensor targets);
float crossentropy_loss(Tensor predictions, Tensor targets);
float binary_crossentropy_loss(Tensor predictions, Tensor targets);
float huber_loss(Tensor predictions, Tensor targets, float delta);

Tensor mse_loss_derivative(Tensor predictions, Tensor targets);
Tensor crossentropy_loss_derivative(Tensor predictions, Tensor targets);

// === OPTIMIZERS ===

void sgd_update(Tensor* weights, Tensor* gradients, float learning_rate);
void sgd_momentum_update(Tensor* weights, Tensor* gradients, Tensor* momentum,
                        float learning_rate, float momentum_factor);
void adam_update(Tensor* weights, Tensor* gradients, 
                Tensor* m, Tensor* v, 
                float learning_rate, float beta1, float beta2, float eps, int t);
void adamw_update(Tensor* weights, Tensor* gradients,
                 Tensor* m, Tensor* v,
                 float learning_rate, float beta1, float beta2, float eps, 
                 float weight_decay, int t);

// === UTILITIES ===

// Data preprocessing
void normalize_data(Tensor* data, float* mean, float* std);
void standardize_data(Tensor* data);
void shuffle_data(Tensor* inputs, Tensor* targets, int num_samples);
void train_test_split(Tensor* inputs, Tensor* targets, int num_samples,
                     float test_ratio, 
                     Tensor** train_inputs, Tensor** train_targets, int* train_size,
                     Tensor** test_inputs, Tensor** test_targets, int* test_size);

// Learning rate scheduling
float cosine_decay_lr(float initial_lr, int current_step, int total_steps);
float exponential_decay_lr(float initial_lr, int current_step, float decay_rate, int decay_steps);
float step_decay_lr(float initial_lr, int current_epoch, float drop_rate, int epochs_drop);

// Model persistence
void model_save(Model* model, const char* filepath);
Model* model_load(const char* filepath);
void model_save_weights(Model* model, const char* filepath);
void model_load_weights(Model* model, const char* filepath);

// Performance profiling
void start_profiling(void);
void end_profiling(const char* operation_name);
void print_training_summary(Model* model);
void print_performance_stats(Model* model);

// Gradient checking and debugging
bool check_gradients(Model* model, Tensor input, Tensor target, float epsilon);
void print_layer_gradients(Model* model);
void visualize_weights(Model* model, int layer_index);

// Advanced training techniques
void apply_gradient_clipping(Model* model, float max_norm);
void apply_weight_decay(Model* model, float decay_rate);
void update_learning_rate(Model* model, float new_lr);

// Callbacks and monitoring
typedef struct {
    void (*on_train_begin)(Model* model);
    void (*on_train_end)(Model* model);
    void (*on_epoch_begin)(Model* model, int epoch);
    void (*on_epoch_end)(Model* model, int epoch);
    void (*on_batch_begin)(Model* model, int batch);
    void (*on_batch_end)(Model* model, int batch);
} Callbacks;

void register_callbacks(Model* model, Callbacks* callbacks);

#endif // TRAINING_H
