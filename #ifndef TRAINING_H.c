#ifndef TRAINING_H
#define TRAINING_H

#include "tensor.h"

// Forward declarations
typedef struct Layer Layer;
typedef struct Model Model;

// Loss functions
typedef enum {
    LOSS_MSE,
    LOSS_CROSSENTROPY
} LossType;

// Optimizers
typedef enum {
    OPT_SGD,
    OPT_ADAM
} OptimizerType;

typedef struct {
    OptimizerType type;
    float learning_rate;
    float beta1, beta2;  // For Adam
    float epsilon;       // For Adam
    int t;              // Time step for Adam
} Optimizer;

// Layer types
typedef enum {
    LAYER_DENSE,
    LAYER_RELU,
    LAYER_SOFTMAX
} LayerType;

// Layer structure
struct Layer {
    LayerType type;
    char name[64];
    
    // Dimensions
    int input_size;
    int output_size;
    
    // Parameters
    Tensor weights;
    Tensor bias;
    
    // Gradients
    Tensor grad_weights;
    Tensor grad_bias;
    
    // Adam optimizer states
    Tensor m_weights, v_weights;  // Momentum for weights
    Tensor m_bias, v_bias;        // Momentum for bias
    
    // Forward/backward cache
    Tensor input_cache;   // Cached input for backward pass
    Tensor output_cache;  // Cached output
    
    struct Layer* next;
};

// Model structure
struct Model {
    char name[64];
    Layer* layers;
    int num_layers;
    LossType loss_type;
    Optimizer optimizer;
    
    // Training metrics
    float current_loss;
    float current_accuracy;
    int epoch;
};

// Function prototypes
Model* model_create(const char* name, LossType loss_type);
void model_free(Model* model);
void model_add_layer(Model* model, LayerType type, int input_size, int output_size);
void model_compile(Model* model, OptimizerType opt_type, float learning_rate);

// Training functions
Tensor model_forward(Model* model, Tensor input);
void model_backward(Model* model, Tensor predictions, Tensor targets);
void model_update_weights(Model* model);
float model_compute_loss(Model* model, Tensor predictions, Tensor targets);
float model_compute_accuracy(Model* model, Tensor predictions, Tensor targets);

// Training loop
void model_train_epoch(Model* model, Tensor* inputs, Tensor* targets, int batch_size, int num_samples);
void model_train(Model* model, Tensor* inputs, Tensor* targets, int num_samples, int epochs, int batch_size);

// Layer operations
Tensor layer_forward(Layer* layer, Tensor input);
Tensor layer_backward(Layer* layer, Tensor grad_output);
void layer_update_weights(Layer* layer, Optimizer* opt);

// Utility functions
void xavier_init(Tensor* tensor);
void zero_init(Tensor* tensor);
Tensor softmax(Tensor input);
Tensor relu_derivative(Tensor input);

#endif