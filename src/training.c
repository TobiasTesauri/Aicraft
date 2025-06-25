#include "training.h"
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

// === MODEL CREATION AND MANAGEMENT ===

Model* model_create(const char* name, LossType loss_type, BackendType backend) {
    if (!g_initialized) aicraft_init();
    
    Model* model = (Model*)malloc(sizeof(Model));
    if (!model) {
        aicraft_log(LOG_ERROR, "[AiCraft] Errore allocazione memoria per modello");
        return NULL;
    }
    
    // Initialize model
    memset(model, 0, sizeof(Model));
    strncpy(model->name, name, sizeof(model->name) - 1);
    model->loss_type = loss_type;
    model->backend = (backend == BACKEND_AUTO) ? g_default_backend : backend;
    model->training_mode = true;
    model->best_accuracy = 0.0f;
    
    aicraft_log(LOG_INFO, "[AiCraft] Modello '%s' creato (Backend: %s)", 
                name, model->backend == BACKEND_CUDA ? "CUDA" : "CPU");
    
    return model;
}

void model_free(Model* model) {
    if (!model) return;
    
    aicraft_log(LOG_INFO, "[AiCraft] Liberazione modello '%s'", model->name);
    
    // Free all layers
    Layer* layer = model->layers;
    while (layer) {
        Layer* next = layer->next;
        
        // Free layer tensors
        if (layer->type == LAYER_DENSE) {
            tensor_free(layer->weights);
            tensor_free(layer->bias);
            tensor_free(layer->grad_weights);
            tensor_free(layer->grad_bias);
            tensor_free(layer->m_weights);
            tensor_free(layer->v_weights);
            tensor_free(layer->m_bias);
            tensor_free(layer->v_bias);
        }
        
        if (layer->input_cache.data || layer->input_cache.cuda_data) {
            tensor_free(layer->input_cache);
        }
        if (layer->output_cache.data || layer->output_cache.cuda_data) {
            tensor_free(layer->output_cache);
        }
        if (layer->pre_activation.data || layer->pre_activation.cuda_data) {
            tensor_free(layer->pre_activation);
        }
        
        // Free dropout mask
        if (layer->dropout_state.mask.data || layer->dropout_state.mask.cuda_data) {
            tensor_free(layer->dropout_state.mask);
        }
        
        // Free batch norm parameters
        if (layer->type == LAYER_BATCH_NORM) {
            tensor_free(layer->batchnorm_state.running_mean);
            tensor_free(layer->batchnorm_state.running_var);
            tensor_free(layer->batchnorm_state.gamma);
            tensor_free(layer->batchnorm_state.beta);
        }
        
        // Free layer norm parameters
        if (layer->type == LAYER_LAYER_NORM) {
            tensor_free(layer->layernorm_state.gamma);
            tensor_free(layer->layernorm_state.beta);
        }
        
        free(layer);
        layer = next;
    }
    
    // Free learning rate history
    if (model->lr_history) {
        free(model->lr_history);
    }
    
    free(model);
}

void model_summary(Model* model) {
    if (!model) return;
    
    printf("\n");
    aicraft_log(LOG_INFO, "=== RIEPILOGO MODELLO: %s ===", model->name);
    aicraft_log(LOG_INFO, "Backend: %s", model->backend == BACKEND_CUDA ? "CUDA" : "CPU");
    aicraft_log(LOG_INFO, "Funzione di perdita: %s", 
                model->loss_type == LOSS_MSE ? "MSE" :
                model->loss_type == LOSS_CROSSENTROPY ? "CrossEntropy" :
                model->loss_type == LOSS_BINARY_CROSSENTROPY ? "BinaryCrossEntropy" : "Huber");
    
    printf("┌─────────────────────────┬─────────────────┬─────────────────┬─────────────────┐\n");
    printf("│ Layer (Type)            │ Output Shape    │ Parameters      │ Backend         │\n");
    printf("├─────────────────────────┼─────────────────┼─────────────────┼─────────────────┤\n");
    
    Layer* layer = model->layers;
    int layer_idx = 0;
    size_t total_params = 0;
    
    while (layer) {
        const char* type_name = "";
        char output_shape[32] = "";
        size_t layer_params = 0;
        const char* backend_name = layer->weights.on_cuda ? "CUDA" : "CPU";
        
        switch (layer->type) {
            case LAYER_DENSE:
                type_name = "Dense";
                snprintf(output_shape, sizeof(output_shape), "(%d,)", layer->output_size);
                layer_params = layer->weights.rows * layer->weights.cols + layer->bias.rows * layer->bias.cols;
                break;
            case LAYER_RELU:
                type_name = "ReLU";
                snprintf(output_shape, sizeof(output_shape), "(%d,)", layer->output_size);
                break;
            case LAYER_SIGMOID:
                type_name = "Sigmoid";
                snprintf(output_shape, sizeof(output_shape), "(%d,)", layer->output_size);
                break;
            case LAYER_SOFTMAX:
                type_name = "Softmax";
                snprintf(output_shape, sizeof(output_shape), "(%d,)", layer->output_size);
                break;
            case LAYER_GELU:
                type_name = "GELU";
                snprintf(output_shape, sizeof(output_shape), "(%d,)", layer->output_size);
                break;
            case LAYER_DROPOUT:
                type_name = "Dropout";
                snprintf(output_shape, sizeof(output_shape), "(%d,)", layer->output_size);
                break;
            case LAYER_BATCH_NORM:
                type_name = "BatchNorm";
                snprintf(output_shape, sizeof(output_shape), "(%d,)", layer->output_size);
                layer_params = 2 * layer->output_size; // gamma + beta
                break;
            case LAYER_LAYER_NORM:
                type_name = "LayerNorm";
                snprintf(output_shape, sizeof(output_shape), "(%d,)", layer->output_size);
                layer_params = 2 * layer->output_size; // gamma + beta
                break;
        }
        
        printf("│ %-23s │ %-15s │ %-15zu │ %-15s │\n", 
               type_name, output_shape, layer_params, backend_name);
        
        total_params += layer_params;
        layer = layer->next;
        layer_idx++;
    }
    
    printf("└─────────────────────────┴─────────────────┴─────────────────┴─────────────────┘\n");
    aicraft_log(LOG_INFO, "Totale parametri: %zu", total_params);
    aicraft_log(LOG_INFO, "Memoria stimata: %.2f MB", 
                (total_params * sizeof(float)) / 1024.0 / 1024.0);
    
    if (model->backend == BACKEND_CUDA) {
        aicraft_log(LOG_INFO, "Memoria GPU utilizzata: %.2f MB", 
                    aicraft_cuda_memory_usage() / 1024.0 / 1024.0);
    }
    
    printf("\n");
}

// === LAYER MANAGEMENT ===

void model_add_layer(Model* model, LayerType type, int input_size, int output_size) {
    if (!model) return;
    
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    if (!layer) {
        aicraft_log(LOG_ERROR, "[AiCraft] Errore allocazione layer");
        return;
    }
    
    // Initialize layer
    memset(layer, 0, sizeof(Layer));
    layer->type = type;
    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->next = NULL;
    
    // Set layer name
    const char* type_names[] = {
        "Dense", "ReLU", "Sigmoid", "Tanh", "Softmax", "GELU", 
        "Dropout", "BatchNorm", "LayerNorm"
    };
    snprintf(layer->name, sizeof(layer->name), "%s_%d", 
             type_names[type], model->num_layers);
    
    // Initialize layer-specific parameters
    switch (type) {
        case LAYER_DENSE:
            layer->weights = tensor_create(input_size, output_size, model->backend);
            layer->bias = tensor_create(1, output_size, model->backend);
            layer->grad_weights = tensor_create(input_size, output_size, model->backend);
            layer->grad_bias = tensor_create(1, output_size, model->backend);
            
            // Initialize Adam moments
            layer->m_weights = tensor_create(input_size, output_size, model->backend);
            layer->v_weights = tensor_create(input_size, output_size, model->backend);
            layer->m_bias = tensor_create(1, output_size, model->backend);
            layer->v_bias = tensor_create(1, output_size, model->backend);
            
            // Initialize weights with Xavier initialization
            xavier_init(&layer->weights);
            zero_init(&layer->bias);
            zero_init(&layer->grad_weights);
            zero_init(&layer->grad_bias);
            zero_init(&layer->m_weights);
            zero_init(&layer->v_weights);
            zero_init(&layer->m_bias);
            zero_init(&layer->v_bias);
            
            aicraft_log(LOG_DEBUG, "[AiCraft] Layer Dense aggiunto: %dx%d", input_size, output_size);
            break;
            
        case LAYER_DROPOUT:
            layer->dropout_state.dropout_rate = 0.5f; // Default
            layer->dropout_state.training = true;
            aicraft_log(LOG_DEBUG, "[AiCraft] Layer Dropout aggiunto (rate=%.2f)", 
                       layer->dropout_state.dropout_rate);
            break;
            
        case LAYER_BATCH_NORM:
            layer->batchnorm_state.running_mean = tensor_zeros(1, output_size, model->backend);
            layer->batchnorm_state.running_var = tensor_ones(1, output_size, model->backend);
            layer->batchnorm_state.gamma = tensor_ones(1, output_size, model->backend);
            layer->batchnorm_state.beta = tensor_zeros(1, output_size, model->backend);
            layer->batchnorm_state.momentum = 0.1f;
            layer->batchnorm_state.eps = 1e-5f;
            layer->batchnorm_state.training = true;
            aicraft_log(LOG_DEBUG, "[AiCraft] Layer BatchNorm aggiunto");
            break;
            
        case LAYER_LAYER_NORM:
            layer->layernorm_state.gamma = tensor_ones(1, output_size, model->backend);
            layer->layernorm_state.beta = tensor_zeros(1, output_size, model->backend);
            layer->layernorm_state.eps = 1e-5f;
            aicraft_log(LOG_DEBUG, "[AiCraft] Layer LayerNorm aggiunto");
            break;
            
        default:
            aicraft_log(LOG_DEBUG, "[AiCraft] Layer %s aggiunto", type_names[type]);
            break;
    }
    
    // Set activation functions
    switch (type) {
        case LAYER_RELU:
            layer->activation_fn = relu_forward;
            layer->activation_derivative = relu_derivative;
            break;
        case LAYER_SIGMOID:
            layer->activation_fn = sigmoid_forward;
            layer->activation_derivative = sigmoid_derivative;
            break;
        case LAYER_SOFTMAX:
            layer->activation_fn = softmax_forward;
            layer->activation_derivative = NULL; // Usually combined with cross-entropy
            break;
        case LAYER_GELU:
            layer->activation_fn = gelu_forward;
            layer->activation_derivative = gelu_derivative;
            break;
    }
    
    // Add to model
    if (!model->layers) {
        model->layers = layer;
    } else {
        Layer* last = model->layers;
        while (last->next) last = last->next;
        last->next = layer;
    }
    model->num_layers++;
}

void model_add_dense(Model* model, int units, const char* activation) {
    if (!model) return;
    
    // Determine input size from previous layer
    int input_size = units; // Default
    if (model->layers) {
        Layer* last = model->layers;
        while (last->next) last = last->next;
        input_size = last->output_size;
    }
    
    // Add dense layer
    model_add_layer(model, LAYER_DENSE, input_size, units);
    
    // Add activation if specified
    if (activation) {
        if (strcmp(activation, "relu") == 0) {
            model_add_layer(model, LAYER_RELU, units, units);
        } else if (strcmp(activation, "sigmoid") == 0) {
            model_add_layer(model, LAYER_SIGMOID, units, units);
        } else if (strcmp(activation, "tanh") == 0) {
            model_add_layer(model, LAYER_TANH, units, units);
        } else if (strcmp(activation, "softmax") == 0) {
            model_add_layer(model, LAYER_SOFTMAX, units, units);
        } else if (strcmp(activation, "gelu") == 0) {
            model_add_layer(model, LAYER_GELU, units, units);
        } else {
            aicraft_log(LOG_WARNING, "[AiCraft] Attivazione '%s' non riconosciuta", activation);
        }
    }
}

void model_add_dropout(Model* model, float rate) {
    if (!model || !model->layers) return;
    
    Layer* last = model->layers;
    while (last->next) last = last->next;
    
    model_add_layer(model, LAYER_DROPOUT, last->output_size, last->output_size);
    
    // Set dropout rate
    last = model->layers;
    while (last->next) last = last->next;
    last->dropout_state.dropout_rate = rate;
    
    aicraft_log(LOG_DEBUG, "[AiCraft] Dropout aggiunto con rate %.2f", rate);
}

// === MODEL COMPILATION ===

void model_compile(Model* model, OptimizerType opt_type, float learning_rate) {
    if (!model) return;
    
    // Initialize optimizer
    model->optimizer.type = opt_type;
    model->optimizer.learning_rate = learning_rate;
    model->optimizer.initial_lr = learning_rate;
    
    // Set optimizer-specific parameters
    switch (opt_type) {
        case OPT_SGD:
            model->optimizer.momentum = 0.0f;
            break;
        case OPT_ADAM:
        case OPT_ADAMW:
            model->optimizer.beta1 = 0.9f;
            model->optimizer.beta2 = 0.999f;
            model->optimizer.epsilon = 1e-8f;
            model->optimizer.weight_decay = (opt_type == OPT_ADAMW) ? 0.01f : 0.0f;
            break;
        case OPT_RMSPROP:
            model->optimizer.beta2 = 0.999f;
            model->optimizer.epsilon = 1e-8f;
            break;
    }
    
    model->optimizer.t = 0;
    
    const char* opt_names[] = {"SGD", "Adam", "AdamW", "RMSprop"};
    aicraft_log(LOG_INFO, "[AiCraft] Modello compilato con ottimizzatore %s (lr=%.6f)", 
                opt_names[opt_type], learning_rate);
}

// === ACTIVATION FUNCTIONS ===

Tensor relu_forward(Tensor input) {
    return tensor_relu(input);
}

Tensor relu_derivative(Tensor input) {
    return tensor_relu_derivative(input);
}

Tensor sigmoid_forward(Tensor input) {
    Tensor result = tensor_create(input.rows, input.cols, input.backend);
    
    if (result.on_cuda && input.on_cuda) {
#ifdef CUDA_AVAILABLE
        extern void aicraft_cuda_sigmoid(const float* input, float* output, int size);
        aicraft_cuda_sigmoid(input.cuda_data, result.cuda_data, input.rows * input.cols);
#endif
    } else {
        // CPU implementation
        tensor_sync_to_cpu((Tensor*)&input);
        if (result.on_cuda) tensor_to_cpu(&result);
        
        for (int i = 0; i < input.rows * input.cols; i++) {
            result.data[i] = 1.0f / (1.0f + expf(-input.data[i]));
        }
    }
    
    return result;
}

Tensor sigmoid_derivative(Tensor input) {
    Tensor sigmoid_out = sigmoid_forward(input);
    Tensor result = tensor_create(input.rows, input.cols, input.backend);
    
    if (result.on_cuda && sigmoid_out.on_cuda) {
        // GPU: result = sigmoid_out * (1 - sigmoid_out)
        Tensor ones = tensor_ones(input.rows, input.cols, input.backend);
        Tensor one_minus_sig = tensor_sub(ones, sigmoid_out);
        result = tensor_mul(sigmoid_out, one_minus_sig);
        tensor_free(ones);
        tensor_free(one_minus_sig);
    } else {
        // CPU implementation
        tensor_sync_to_cpu(&sigmoid_out);
        if (result.on_cuda) tensor_to_cpu(&result);
        
        for (int i = 0; i < input.rows * input.cols; i++) {
            float s = sigmoid_out.data[i];
            result.data[i] = s * (1.0f - s);
        }
    }
    
    tensor_free(sigmoid_out);
    return result;
}

Tensor tanh_forward(Tensor input) {
    Tensor output = tensor_create(input.rows, input.cols,
                                 input.on_cuda ? BACKEND_CUDA : BACKEND_CPU);
    
    if (input.on_cuda && output.on_cuda) {
#ifdef CUDA_AVAILABLE
        extern void aicraft_cuda_tanh(float* output, const float* input, int size);
        aicraft_cuda_tanh(output.cuda_data, input.cuda_data, input.rows * input.cols);
#endif
    } else {
        tensor_sync_to_cpu(&input);
        if (output.on_cuda) tensor_to_cpu(&output);
        
        int size = input.rows * input.cols;
        for (int i = 0; i < size; i++) {
            output.data[i] = tanhf(input.data[i]);
        }
        
        if (input.on_cuda) tensor_to_cuda(&output);
    }
    
    return output;
}

Tensor softmax_forward(Tensor input) {
    Tensor output = tensor_create(input.rows, input.cols,
                                 input.on_cuda ? BACKEND_CUDA : BACKEND_CPU);
    
    if (input.on_cuda && output.on_cuda) {
#ifdef CUDA_AVAILABLE
        extern void aicraft_cuda_softmax(float* output, const float* input, int rows, int cols);
        aicraft_cuda_softmax(output.cuda_data, input.cuda_data, input.rows, input.cols);
#endif
    } else {
        tensor_sync_to_cpu(&input);
        if (output.on_cuda) tensor_to_cpu(&output);
        
        // Softmax for each row
        for (int i = 0; i < input.rows; i++) {
            float* input_row = &input.data[i * input.cols];
            float* output_row = &output.data[i * input.cols];
            
            // Find max for numerical stability
            float max_val = input_row[0];
            for (int j = 1; j < input.cols; j++) {
                if (input_row[j] > max_val) {
                    max_val = input_row[j];
                }
            }
            
            // Compute exp and sum
            float sum = 0.0f;
            for (int j = 0; j < input.cols; j++) {
                output_row[j] = expf(input_row[j] - max_val);
                sum += output_row[j];
            }
            
            // Normalize
            for (int j = 0; j < input.cols; j++) {
                output_row[j] /= sum;
            }
        }
        
        if (input.on_cuda) tensor_to_cuda(&output);
    }
    
    return output;
}

Tensor gelu_forward(Tensor input) {
    Tensor output = tensor_create(input.rows, input.cols,
                                 input.on_cuda ? BACKEND_CUDA : BACKEND_CPU);
    
    if (input.on_cuda && output.on_cuda) {
#ifdef CUDA_AVAILABLE
        extern void aicraft_cuda_gelu(float* output, const float* input, int size);
        aicraft_cuda_gelu(output.cuda_data, input.cuda_data, input.rows * input.cols);
#endif
    } else {
        tensor_sync_to_cpu(&input);
        if (output.on_cuda) tensor_to_cpu(&output);
        
        int size = input.rows * input.cols;
        const float sqrt_2_pi = sqrtf(2.0f / M_PI);
        
        for (int i = 0; i < size; i++) {
            float x = input.data[i];
            float tanh_arg = sqrt_2_pi * (x + 0.044715f * x * x * x);
            output.data[i] = 0.5f * x * (1.0f + tanhf(tanh_arg));
        }
        
        if (input.on_cuda) tensor_to_cuda(&output);
    }
    
    return output;
}

Tensor tanh_derivative(Tensor input) {
    Tensor grad = tensor_create(input.rows, input.cols,
                               input.on_cuda ? BACKEND_CUDA : BACKEND_CPU);
    
    tensor_sync_to_cpu(&input);
    if (grad.on_cuda) tensor_to_cpu(&grad);
    
    int size = input.rows * input.cols;
    for (int i = 0; i < size; i++) {
        float tanh_val = tanhf(input.data[i]);
        grad.data[i] = 1.0f - tanh_val * tanh_val;
    }
    
    if (input.on_cuda) tensor_to_cuda(&grad);
    
    return grad;
}

Tensor gelu_derivative(Tensor input) {
    Tensor grad = tensor_create(input.rows, input.cols,
                               input.on_cuda ? BACKEND_CUDA : BACKEND_CPU);
    
    tensor_sync_to_cpu(&input);
    if (grad.on_cuda) tensor_to_cpu(&grad);
    
    int size = input.rows * input.cols;
    const float sqrt_2_pi = sqrtf(2.0f / M_PI);
    
    for (int i = 0; i < size; i++) {
        float x = input.data[i];
        float tanh_arg = sqrt_2_pi * (x + 0.044715f * x * x * x);
        float tanh_val = tanhf(tanh_arg);
        float sech2 = 1.0f - tanh_val * tanh_val;
        
        float term1 = 0.5f * (1.0f + tanh_val);
        float term2 = 0.5f * x * sech2 * sqrt_2_pi * (1.0f + 3.0f * 0.044715f * x * x);
        
        grad.data[i] = term1 + term2;
    }
    
    if (input.on_cuda) tensor_to_cuda(&grad);
    
    return grad;
}

// === WEIGHT INITIALIZATION ===

void xavier_init(Tensor* tensor) {
    if (!tensor || (!tensor->data && !tensor->cuda_data)) return;
    
    srand((unsigned int)time(NULL));
    float scale = sqrtf(2.0f / (tensor->rows + tensor->cols));
    
    if (tensor->on_cuda) {
        // Generate on CPU first, then copy
        float* temp_data = (float*)malloc(tensor->rows * tensor->cols * sizeof(float));
        for (int i = 0; i < tensor->rows * tensor->cols; i++) {
            temp_data[i] = scale * ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        }
        
#ifdef CUDA_AVAILABLE
        extern void aicraft_cuda_memcpy_h2d(float* dst, const float* src, size_t size);
        aicraft_cuda_memcpy_h2d(tensor->cuda_data, temp_data, 
                               tensor->rows * tensor->cols * sizeof(float));
#endif
        free(temp_data);
    } else {
        for (int i = 0; i < tensor->rows * tensor->cols; i++) {
            tensor->data[i] = scale * ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        }
    }
}

void zero_init(Tensor* tensor) {
    tensor_zero(tensor);
}

// === LAYER OPERATIONS ===

Tensor layer_forward(Layer* layer, Tensor input) {
    start_timer();
    
    Tensor result;
    
    switch (layer->type) {
        case LAYER_DENSE: {
            // Dense layer: output = input * weights + bias
            result = tensor_matmul(input, layer->weights);
            
            // Add bias (broadcast)
            for (int i = 0; i < result.rows; i++) {
                Tensor row_result = {
                    .data = result.on_cuda ? NULL : (result.data + i * result.cols),
                    .cuda_data = result.on_cuda ? (result.cuda_data + i * result.cols) : NULL,
                    .rows = 1,
                    .cols = result.cols,
                    .on_cuda = result.on_cuda,
                    .backend = result.backend
                };
                Tensor row_sum = tensor_add(row_result, layer->bias);
                
                if (result.on_cuda) {
#ifdef CUDA_AVAILABLE
                    extern void aicraft_cuda_memcpy_d2d(float* dst, const float* src, size_t size);
                    aicraft_cuda_memcpy_d2d(result.cuda_data + i * result.cols, 
                                           row_sum.cuda_data, result.cols * sizeof(float));
#endif
                } else {
                    memcpy(result.data + i * result.cols, row_sum.data, result.cols * sizeof(float));
                }
                
                tensor_free(row_sum);
            }
            break;
        }
        
        case LAYER_RELU:
            result = relu_forward(input);
            break;
            
        case LAYER_SIGMOID:
            result = sigmoid_forward(input);
            break;
            
        case LAYER_TANH:
            result = tanh_forward(input);
            break;
            
        case LAYER_SOFTMAX:
            result = softmax_forward(input);
            break;
            
        case LAYER_GELU:
            result = gelu_forward(input);
            break;
            
        case LAYER_DROPOUT:
            if (layer->dropout_state.training) {
                result = dropout_forward(input, layer->dropout_state.dropout_rate, true);
                // Store mask for backward pass
                // TODO: Implement mask storage
            } else {
                result = tensor_copy(input);
            }
            break;
            
        default:
            aicraft_log(LOG_ERROR, "[AiCraft] Tipo layer non supportato nel forward: %d", layer->type);
            result = tensor_copy(input);
            break;
    }
    
    // Cache input and output for backward pass
    if (layer->input_cache.data || layer->input_cache.cuda_data) {
        tensor_free(layer->input_cache);
    }
    layer->input_cache = tensor_copy(input);
    
    if (layer->output_cache.data || layer->output_cache.cuda_data) {
        tensor_free(layer->output_cache);
    }
    layer->output_cache = tensor_copy(result);
    
    layer->forward_time += get_elapsed_time();
    
    return result;
}

// === DROPOUT IMPLEMENTATION ===

Tensor dropout_forward(Tensor input, float rate, bool training) {
    Tensor result = tensor_create(input.rows, input.cols, input.backend);
    
    if (!training || rate <= 0.0f) {
        // During inference or no dropout, just copy input
        if (result.on_cuda && input.on_cuda) {
#ifdef CUDA_AVAILABLE
            extern void aicraft_cuda_memcpy_d2d(float* dst, const float* src, size_t size);
            aicraft_cuda_memcpy_d2d(result.cuda_data, input.cuda_data, 
                                   input.rows * input.cols * sizeof(float));
#endif
        } else {
            tensor_sync_to_cpu((Tensor*)&input);
            if (result.on_cuda) tensor_to_cpu(&result);
            memcpy(result.data, input.data, input.rows * input.cols * sizeof(float));
        }
        return result;
    }
    
    // Generate dropout mask and apply
    float keep_prob = 1.0f - rate;
    float scale = 1.0f / keep_prob;
    
    if (result.on_cuda && input.on_cuda) {
        // TODO: Implement CUDA dropout
        tensor_to_cpu(&result);
        tensor_sync_to_cpu((Tensor*)&input);
    }
    
    // CPU implementation
    srand((unsigned int)time(NULL));
    for (int i = 0; i < input.rows * input.cols; i++) {
        float rand_val = (float)rand() / RAND_MAX;
        if (rand_val < keep_prob) {
            result.data[i] = input.data[i] * scale;
        } else {
            result.data[i] = 0.0f;
        }
    }
    
    return result;
}

// === FORWARD PROPAGATION ===

Tensor model_forward(Model* model, Tensor input) {
    if (!model || !model->layers) {
        aicraft_log(LOG_ERROR, "[AiCraft] Modello o layer non validi per forward");
        return tensor_create(1, 1, BACKEND_CPU);
    }
    
    Tensor current = input;
    Layer* layer = model->layers;
    
    while (layer) {
        // Cache input for backward pass
        if (layer->input_cache.data || layer->input_cache.cuda_data) {
            tensor_free(layer->input_cache);
        }
        layer->input_cache = tensor_copy(current);
        
        // Forward through layer
        Tensor next = layer_forward(layer, current);
        
        // Free previous tensor (except original input)
        if (layer != model->layers) {
            tensor_free(current);
        }
        
        current = next;
        layer = layer->next;
    }
    
    return current;
}

Tensor layer_forward(Layer* layer, Tensor input) {
    if (!layer) return input;
    
    clock_t start = clock();
    Tensor output;
    
    switch (layer->type) {
        case LAYER_DENSE: {
            // Linear transformation: output = input * weights + bias
            Tensor linear = tensor_matmul(input, layer->weights);
            output = tensor_add(linear, layer->bias);
            tensor_free(linear);
            
            // Cache pre-activation for backward pass
            if (layer->pre_activation.data || layer->pre_activation.cuda_data) {
                tensor_free(layer->pre_activation);
            }
            layer->pre_activation = tensor_copy(output);
            break;
        }
        
        case LAYER_RELU:
            output = relu_forward(input);
            break;
            
        case LAYER_SIGMOID:
            output = sigmoid_forward(input);
            break;
            
        case LAYER_TANH:
            output = tanh_forward(input);
            break;
            
        case LAYER_SOFTMAX:
            output = softmax_forward(input);
            break;
            
        case LAYER_GELU:
            output = gelu_forward(input);
            break;
            
        case LAYER_DROPOUT:
            output = dropout_forward(input, layer->dropout_state.dropout_rate, 
                                   layer->dropout_state.training);
            break;
            
        case LAYER_BATCH_NORM:
            output = batch_norm_forward(input, 
                                      layer->batchnorm_state.gamma,
                                      layer->batchnorm_state.beta,
                                      &layer->batchnorm_state.running_mean,
                                      &layer->batchnorm_state.running_var,
                                      layer->batchnorm_state.momentum,
                                      layer->batchnorm_state.eps,
                                      layer->batchnorm_state.training);
            break;
            
        case LAYER_LAYER_NORM:
            output = layer_norm_forward(input,
                                      layer->layernorm_state.gamma,
                                      layer->layernorm_state.beta,
                                      layer->layernorm_state.eps);
            break;
            
        default:
            output = tensor_copy(input);
            break;
    }
    
    // Cache output
    if (layer->output_cache.data || layer->output_cache.cuda_data) {
        tensor_free(layer->output_cache);
    }
    layer->output_cache = tensor_copy(output);
    
    // Update timing
    layer->forward_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    
    return output;
}

// === BACKWARD PROPAGATION ===

void model_backward(Model* model, Tensor predictions, Tensor targets) {
    if (!model || !model->layers) return;
    
    // Compute loss gradient
    Tensor grad_loss = mse_loss_derivative(predictions, targets);
    if (model->loss_type == LOSS_CROSSENTROPY) {
        tensor_free(grad_loss);
        grad_loss = crossentropy_loss_derivative(predictions, targets);
    }
    
    // Backward through layers in reverse order
    Tensor current_grad = grad_loss;
    
    // Find last layer
    Layer* layers[256]; // Max layers
    int num_layers = 0;
    Layer* layer = model->layers;
    while (layer && num_layers < 256) {
        layers[num_layers++] = layer;
        layer = layer->next;
    }
    
    // Backward pass
    for (int i = num_layers - 1; i >= 0; i--) {
        Tensor next_grad = layer_backward(layers[i], current_grad);
        tensor_free(current_grad);
        current_grad = next_grad;
    }
    
    tensor_free(current_grad);
}

Tensor layer_backward(Layer* layer, Tensor grad_output) {
    if (!layer) return grad_output;
    
    clock_t start = clock();
    Tensor grad_input;
    
    switch (layer->type) {
        case LAYER_DENSE: {
            // Compute gradients
            // grad_weights = input^T * grad_output
            Tensor input_t = tensor_transpose(layer->input_cache);
            tensor_free(layer->grad_weights);
            layer->grad_weights = tensor_matmul(input_t, grad_output);
            tensor_free(input_t);
            
            // grad_bias = sum(grad_output, axis=0)
            tensor_free(layer->grad_bias);
            layer->grad_bias = tensor_sum_axis(grad_output, 0);
            
            // grad_input = grad_output * weights^T
            Tensor weights_t = tensor_transpose(layer->weights);
            grad_input = tensor_matmul(grad_output, weights_t);
            tensor_free(weights_t);
            break;
        }
        
        case LAYER_RELU:
            grad_input = relu_derivative(layer->input_cache);
            tensor_elementwise_mul(&grad_input, grad_output);
            break;
            
        case LAYER_SIGMOID:
            grad_input = sigmoid_derivative(layer->output_cache);
            tensor_elementwise_mul(&grad_input, grad_output);
            break;
            
        case LAYER_TANH:
            grad_input = tanh_derivative(layer->output_cache);
            tensor_elementwise_mul(&grad_input, grad_output);
            break;
            
        case LAYER_GELU:
            grad_input = gelu_derivative(layer->input_cache);
            tensor_elementwise_mul(&grad_input, grad_output);
            break;
            
        default:
            grad_input = tensor_copy(grad_output);
            break;
    }
    
    layer->backward_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    return grad_input;
}

// === WEIGHT UPDATES ===

void model_update_weights(Model* model) {
    if (!model) return;
    
    Layer* layer = model->layers;
    while (layer) {
        layer_update_weights(layer, &model->optimizer);
        layer = layer->next;
    }
}

void layer_update_weights(Layer* layer, Optimizer* opt) {
    if (!layer || !opt) return;
    
    if (layer->type == LAYER_DENSE) {
        switch (opt->type) {
            case OPT_SGD:
                sgd_update(&layer->weights, &layer->grad_weights, opt->learning_rate);
                sgd_update(&layer->bias, &layer->grad_bias, opt->learning_rate);
                break;
                
            case OPT_ADAM:
                adam_update(&layer->weights, &layer->grad_weights,
                           &layer->m_weights, &layer->v_weights,
                           opt->learning_rate, opt->beta1, opt->beta2, opt->epsilon, opt->t);
                adam_update(&layer->bias, &layer->grad_bias,
                           &layer->m_bias, &layer->v_bias,
                           opt->learning_rate, opt->beta1, opt->beta2, opt->epsilon, opt->t);
                break;
                
            default:
                sgd_update(&layer->weights, &layer->grad_weights, opt->learning_rate);
                sgd_update(&layer->bias, &layer->grad_bias, opt->learning_rate);
                break;
        }
    }
}

// === LOSS FUNCTIONS ===

float model_compute_loss(Model* model, Tensor predictions, Tensor targets) {
    switch (model->loss_type) {
        case LOSS_MSE:
            return mse_loss(predictions, targets);
        case LOSS_CROSSENTROPY:
            return crossentropy_loss(predictions, targets);
        case LOSS_BINARY_CROSSENTROPY:
            return binary_crossentropy_loss(predictions, targets);
        case LOSS_HUBER:
            return huber_loss(predictions, targets, 1.0f);
        default:
            return mse_loss(predictions, targets);
    }
}

float mse_loss(Tensor predictions, Tensor targets) {
    if (predictions.rows != targets.rows || predictions.cols != targets.cols) {
        aicraft_log(LOG_ERROR, "[AiCraft] Dimensioni incompatibili per MSE loss");
        return 0.0f;
    }
    
    // Sync to CPU for computation
    tensor_sync_to_cpu(&predictions);
    tensor_sync_to_cpu(&targets);
    
    float loss = 0.0f;
    int size = predictions.rows * predictions.cols;
    
    for (int i = 0; i < size; i++) {
        float diff = predictions.data[i] - targets.data[i];
        loss += diff * diff;
    }
    
    return loss / (2.0f * size);
}

float crossentropy_loss(Tensor predictions, Tensor targets) {
    tensor_sync_to_cpu(&predictions);
    tensor_sync_to_cpu(&targets);
    
    float loss = 0.0f;
    int size = predictions.rows * predictions.cols;
    
    for (int i = 0; i < size; i++) {
        float pred = fmaxf(predictions.data[i], 1e-15f); // Clip to avoid log(0)
        loss -= targets.data[i] * logf(pred);
    }
    
    return loss / predictions.rows;
}

Tensor mse_loss_derivative(Tensor predictions, Tensor targets) {
    Tensor grad = tensor_create(predictions.rows, predictions.cols, 
                               predictions.on_cuda ? BACKEND_CUDA : BACKEND_CPU);
    
    tensor_sync_to_cpu(&predictions);
    tensor_sync_to_cpu(&targets);
    
    if (grad.on_cuda) tensor_to_cpu(&grad);
    
    int size = predictions.rows * predictions.cols;
    for (int i = 0; i < size; i++) {
        grad.data[i] = (predictions.data[i] - targets.data[i]) / size;
    }
    
    if (predictions.on_cuda) tensor_to_cuda(&grad);
    
    return grad;
}

Tensor crossentropy_loss_derivative(Tensor predictions, Tensor targets) {
    Tensor grad = tensor_create(predictions.rows, predictions.cols,
                               predictions.on_cuda ? BACKEND_CUDA : BACKEND_CPU);
    
    tensor_sync_to_cpu(&predictions);
    tensor_sync_to_cpu(&targets);
    
    if (grad.on_cuda) tensor_to_cpu(&grad);
    
    int size = predictions.rows * predictions.cols;
    for (int i = 0; i < size; i++) {
        float pred = fmaxf(predictions.data[i], 1e-15f);
        grad.data[i] = (pred - targets.data[i]) / predictions.rows;
    }
    
    if (predictions.on_cuda) tensor_to_cuda(&grad);
    
    return grad;
}


