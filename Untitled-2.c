#include "training.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

// Model creation and management
Model* model_create(const char* name, LossType loss_type) {
    Model* model = malloc(sizeof(Model));
    strcpy(model->name, name);
    model->layers = NULL;
    model->num_layers = 0;
    model->loss_type = loss_type;
    model->current_loss = 0.0f;
    model->current_accuracy = 0.0f;
    model->epoch = 0;
    return model;
}

void model_free(Model* model) {
    Layer* layer = model->layers;
    while (layer) {
        Layer* next = layer->next;
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
        tensor_free(layer->input_cache);
        tensor_free(layer->output_cache);
        free(layer);
        layer = next;
    }
    free(model);
}

void model_add_layer(Model* model, LayerType type, int input_size, int output_size) {
    Layer* layer = malloc(sizeof(Layer));
    layer->type = type;
    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->next = NULL;
    
    sprintf(layer->name, "%s_%d", 
            type == LAYER_DENSE ? "Dense" : 
            type == LAYER_RELU ? "ReLU" : "Softmax", 
            model->num_layers);
    
    if (type == LAYER_DENSE) {
        layer->weights = tensor_create(input_size, output_size);
        layer->bias = tensor_create(1, output_size);
        layer->grad_weights = tensor_create(input_size, output_size);
        layer->grad_bias = tensor_create(1, output_size);
        layer->m_weights = tensor_create(input_size, output_size);
        layer->v_weights = tensor_create(input_size, output_size);
        layer->m_bias = tensor_create(1, output_size);
        layer->v_bias = tensor_create(1, output_size);
        
        xavier_init(&layer->weights);
        zero_init(&layer->bias);
        zero_init(&layer->grad_weights);
        zero_init(&layer->grad_bias);
        zero_init(&layer->m_weights);
        zero_init(&layer->v_weights);
        zero_init(&layer->m_bias);
        zero_init(&layer->v_bias);
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

void model_compile(Model* model, OptimizerType opt_type, float learning_rate) {
    model->optimizer.type = opt_type;
    model->optimizer.learning_rate = learning_rate;
    model->optimizer.beta1 = 0.9f;
    model->optimizer.beta2 = 0.999f;
    model->optimizer.epsilon = 1e-8f;
    model->optimizer.t = 0;
}

// Forward propagation
Tensor model_forward(Model* model, Tensor input) {
    Tensor current = input;
    Layer* layer = model->layers;
    
    while (layer) {
        // Cache input for backward pass
        if (layer->input_cache.data) tensor_free(layer->input_cache);
        layer->input_cache = tensor_create(current.rows, current.cols);
        memcpy(layer->input_cache.data, current.data, sizeof(float) * current.rows * current.cols);
        
        Tensor output = layer_forward(layer, current);
        
        // Cache output
        if (layer->output_cache.data) tensor_free(layer->output_cache);
        layer->output_cache = tensor_create(output.rows, output.cols);
        memcpy(layer->output_cache.data, output.data, sizeof(float) * output.rows * output.cols);
        
        if (current.data != input.data) tensor_free(current);
        current = output;
        layer = layer->next;
    }
    
    return current;
}

// Backward propagation
void model_backward(Model* model, Tensor predictions, Tensor targets) {
    // Compute initial gradient (loss derivative)
    Tensor grad = tensor_create(predictions.rows, predictions.cols);
    
    if (model->loss_type == LOSS_MSE) {
        // MSE: grad = 2 * (pred - target) / batch_size
        for (int i = 0; i < predictions.rows * predictions.cols; i++) {
            grad.data[i] = 2.0f * (predictions.data[i] - targets.data[i]) / predictions.rows;
        }
    } else { // LOSS_CROSSENTROPY
        // Cross-entropy with softmax: grad = (pred - target) / batch_size
        for (int i = 0; i < predictions.rows * predictions.cols; i++) {
            grad.data[i] = (predictions.data[i] - targets.data[i]) / predictions.rows;
        }
    }
    
    // Backward through layers (reverse order)
    Layer* layers[256]; // Assume max 256 layers
    int layer_count = 0;
    Layer* layer = model->layers;
    while (layer) {
        layers[layer_count++] = layer;
        layer = layer->next;
    }
    
    Tensor current_grad = grad;
    for (int i = layer_count - 1; i >= 0; i--) {
        Tensor next_grad = layer_backward(layers[i], current_grad);
        if (current_grad.data != grad.data) tensor_free(current_grad);
        current_grad = next_grad;
    }
    
    if (current_grad.data != grad.data) tensor_free(current_grad);
    tensor_free(grad);
}

// Update model weights
void model_update_weights(Model* model) {
    model->optimizer.t++;
    Layer* layer = model->layers;
    while (layer) {
        layer_update_weights(layer, &model->optimizer);
        layer = layer->next;
    }
}

// Loss computation
float model_compute_loss(Model* model, Tensor predictions, Tensor targets) {
    float loss = 0.0f;
    int batch_size = predictions.rows;
    
    if (model->loss_type == LOSS_MSE) {
        for (int i = 0; i < predictions.rows * predictions.cols; i++) {
            float diff = predictions.data[i] - targets.data[i];
            loss += diff * diff;
        }
        loss /= batch_size;
    } else { // LOSS_CROSSENTROPY
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < predictions.cols; j++) {
                float pred = predictions.data[i * predictions.cols + j];
                float target = targets.data[i * targets.cols + j];
                if (target > 0) loss -= target * logf(fmaxf(pred, 1e-15f));
            }
        }
        loss /= batch_size;
    }
    
    return loss;
}

// Accuracy computation
float model_compute_accuracy(Model* model, Tensor predictions, Tensor targets) {
    int correct = 0;
    int batch_size = predictions.rows;
    
    for (int i = 0; i < batch_size; i++) {
        int pred_class = 0, true_class = 0;
        float max_pred = predictions.data[i * predictions.cols];
        float max_true = targets.data[i * targets.cols];
        
        for (int j = 1; j < predictions.cols; j++) {
            if (predictions.data[i * predictions.cols + j] > max_pred) {
                max_pred = predictions.data[i * predictions.cols + j];
                pred_class = j;
            }
            if (targets.data[i * targets.cols + j] > max_true) {
                max_true = targets.data[i * targets.cols + j];
                true_class = j;
            }
        }
        
        if (pred_class == true_class) correct++;
    }
    
    return (float)correct / batch_size;
}

// Training epoch
void model_train_epoch(Model* model, Tensor* inputs, Tensor* targets, int batch_size, int num_samples) {
    float total_loss = 0.0f;
    float total_accuracy = 0.0f;
    int num_batches = (num_samples + batch_size - 1) / batch_size;
    
    for (int batch = 0; batch < num_batches; batch++) {
        int start_idx = batch * batch_size;
        int end_idx = fmin(start_idx + batch_size, num_samples);
        int current_batch_size = end_idx - start_idx;
        
        // Create batch tensors
        Tensor batch_input = tensor_create(current_batch_size, inputs[0].cols);
        Tensor batch_target = tensor_create(current_batch_size, targets[0].cols);
        
        for (int i = 0; i < current_batch_size; i++) {
            memcpy(&batch_input.data[i * batch_input.cols], 
                   inputs[start_idx + i].data, 
                   sizeof(float) * batch_input.cols);
            memcpy(&batch_target.data[i * batch_target.cols], 
                   targets[start_idx + i].data, 
                   sizeof(float) * batch_target.cols);
        }
        
        // Forward pass
        Tensor predictions = model_forward(model, batch_input);
        
        // Compute metrics
        float loss = model_compute_loss(model, predictions, batch_target);
        float accuracy = model_compute_accuracy(model, predictions, batch_target);
        
        total_loss += loss;
        total_accuracy += accuracy;
        
        // Backward pass
        model_backward(model, predictions, batch_target);
        
        // Update weights
        model_update_weights(model);
        
        // Cleanup
        tensor_free(batch_input);
        tensor_free(batch_target);
        tensor_free(predictions);
    }
    
    model->current_loss = total_loss / num_batches;
    model->current_accuracy = total_accuracy / num_batches;
    model->epoch++;
}

// Full training
void model_train(Model* model, Tensor* inputs, Tensor* targets, int num_samples, int epochs, int batch_size) {
    printf("Training model '%s' for %d epochs...\n", model->name, epochs);
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        model_train_epoch(model, inputs, targets, batch_size, num_samples);
        
        printf("Epoch %d/%d - Loss: %.4f - Accuracy: %.4f\n", 
               epoch + 1, epochs, model->current_loss, model->current_accuracy);
    }
}

// Layer operations
Tensor layer_forward(Layer* layer, Tensor input) {
    switch (layer->type) {
        case LAYER_DENSE: {
            return dense_forward(input, layer->weights, layer->bias);
        }
        case LAYER_RELU: {
            return relu_forward(input);
        }
        case LAYER_SOFTMAX: {
            return softmax(input);
        }
        default:
            printf("Unknown layer type\n");
            exit(1);
    }
}

Tensor layer_backward(Layer* layer, Tensor grad_output) {
    switch (layer->type) {
        case LAYER_DENSE: {
            // Compute gradients for weights and bias
            Tensor input = layer->input_cache;
            
            // grad_weights = input^T * grad_output
            for (int i = 0; i < layer->grad_weights.rows; i++) {
                for (int j = 0; j < layer->grad_weights.cols; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < input.rows; k++) {
                        sum += input.data[k * input.cols + i] * grad_output.data[k * grad_output.cols + j];
                    }
                    layer->grad_weights.data[i * layer->grad_weights.cols + j] = sum;
                }
            }
            
            // grad_bias = sum(grad_output, axis=0)
            for (int j = 0; j < layer->grad_bias.cols; j++) {
                float sum = 0.0f;
                for (int i = 0; i < grad_output.rows; i++) {
                    sum += grad_output.data[i * grad_output.cols + j];
                }
                layer->grad_bias.data[j] = sum;
            }
            
            // grad_input = grad_output * weights^T
            Tensor grad_input = tensor_create(grad_output.rows, layer->weights.rows);
            for (int i = 0; i < grad_input.rows; i++) {
                for (int j = 0; j < grad_input.cols; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < grad_output.cols; k++) {
                        sum += grad_output.data[i * grad_output.cols + k] * layer->weights.data[j * layer->weights.cols + k];
                    }
                    grad_input.data[i * grad_input.cols + j] = sum;
                }
            }
            return grad_input;
        }
        case LAYER_RELU: {
            Tensor grad_input = tensor_create(grad_output.rows, grad_output.cols);
            Tensor relu_mask = relu_derivative(layer->input_cache);
            for (int i = 0; i < grad_output.rows * grad_output.cols; i++) {
                grad_input.data[i] = grad_output.data[i] * relu_mask.data[i];
            }
            tensor_free(relu_mask);
            return grad_input;
        }
        case LAYER_SOFTMAX: {
            // For softmax + cross-entropy, gradient is already computed correctly
            Tensor grad_input = tensor_create(grad_output.rows, grad_output.cols);
            memcpy(grad_input.data, grad_output.data, sizeof(float) * grad_output.rows * grad_output.cols);
            return grad_input;
        }
        default:
            printf("Unknown layer type in backward\n");
            exit(1);
    }
}

void layer_update_weights(Layer* layer, Optimizer* opt) {
    if (layer->type != LAYER_DENSE) return;
    
    if (opt->type == OPT_SGD) {
        // Simple SGD: w = w - lr * grad
        for (int i = 0; i < layer->weights.rows * layer->weights.cols; i++) {
            layer->weights.data[i] -= opt->learning_rate * layer->grad_weights.data[i];
        }
        for (int i = 0; i < layer->bias.rows * layer->bias.cols; i++) {
            layer->bias.data[i] -= opt->learning_rate * layer->grad_bias.data[i];
        }
    } else { // OPT_ADAM
        float lr_t = opt->learning_rate * sqrtf(1.0f - powf(opt->beta2, opt->t)) / (1.0f - powf(opt->beta1, opt->t));
        
        // Update weights
        for (int i = 0; i < layer->weights.rows * layer->weights.cols; i++) {
            layer->m_weights.data[i] = opt->beta1 * layer->m_weights.data[i] + (1 - opt->beta1) * layer->grad_weights.data[i];
            layer->v_weights.data[i] = opt->beta2 * layer->v_weights.data[i] + (1 - opt->beta2) * layer->grad_weights.data[i] * layer->grad_weights.data[i];
            layer->weights.data[i] -= lr_t * layer->m_weights.data[i] / (sqrtf(layer->v_weights.data[i]) + opt->epsilon);
        }
        
        // Update bias
        for (int i = 0; i < layer->bias.rows * layer->bias.cols; i++) {
            layer->m_bias.data[i] = opt->beta1 * layer->m_bias.data[i] + (1 - opt->beta1) * layer->grad_bias.data[i];
            layer->v_bias.data[i] = opt->beta2 * layer->v_bias.data[i] + (1 - opt->beta2) * layer->grad_bias.data[i] * layer->grad_bias.data[i];
            layer->bias.data[i] -= lr_t * layer->m_bias.data[i] / (sqrtf(layer->v_bias.data[i]) + opt->epsilon);
        }
    }
}

// Utility functions
void xavier_init(Tensor* tensor) {
    srand(time(NULL));
    float scale = sqrtf(2.0f / (tensor->rows + tensor->cols));
    for (int i = 0; i < tensor->rows * tensor->cols; i++) {
        tensor->data[i] = scale * ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
}

void zero_init(Tensor* tensor) {
    for (int i = 0; i < tensor->rows * tensor->cols; i++) {
        tensor->data[i] = 0.0f;
    }
}

Tensor softmax(Tensor input) {
    Tensor output = tensor_create(input.rows, input.cols);
    
    for (int i = 0; i < input.rows; i++) {
        // Find max for numerical stability
        float max_val = input.data[i * input.cols];
        for (int j = 1; j < input.cols; j++) {
            if (input.data[i * input.cols + j] > max_val) {
                max_val = input.data[i * input.cols + j];
            }
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int j = 0; j < input.cols; j++) {
            output.data[i * input.cols + j] = expf(input.data[i * input.cols + j] - max_val);
            sum += output.data[i * input.cols + j];
        }
        
        // Normalize
        for (int j = 0; j < input.cols; j++) {
            output.data[i * input.cols + j] /= sum;
        }
    }
    
    return output;
}

Tensor relu_derivative(Tensor input) {
    Tensor output = tensor_create(input.rows, input.cols);
    for (int i = 0; i < input.rows * input.cols; i++) {
        output.data[i] = input.data[i] > 0.0f ? 1.0f : 0.0f;
    }
    return output;
}