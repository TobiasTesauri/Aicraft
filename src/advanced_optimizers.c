#include "training.h"
#include <math.h>
#include <string.h>
#include <time.h>

// === ADVANCED OPTIMIZERS IMPLEMENTATION ===

// AdaBound: Adaptive Gradient Methods with Dynamic Bound of Learning Rate
void adabound_update(Tensor* weights, Tensor* gradients, 
                    Tensor* m, Tensor* v,
                    float lr, float beta1, float beta2, float eps,
                    float final_lr, float gamma, int t, int size) {
    
    if (weights->on_cuda && gradients->on_cuda) {
#ifdef CUDA_AVAILABLE
        extern void aicraft_cuda_adabound_kernel(float* weights, const float* gradients,
                                                float* m, float* v,
                                                float lr, float beta1, float beta2, float eps,
                                                float final_lr, float gamma, int t, int size);
        
        dim3 block(256);
        dim3 grid((size + block.x - 1) / block.x);
        aicraft_cuda_adabound_kernel<<<grid, block>>>(weights->cuda_data, gradients->cuda_data,
                                                     m->cuda_data, v->cuda_data,
                                                     lr, beta1, beta2, eps,
                                                     final_lr, gamma, t, size);
#endif
    } else {
        tensor_sync_to_cpu(weights);
        tensor_sync_to_cpu(gradients);
        tensor_sync_to_cpu(m);
        tensor_sync_to_cpu(v);
        
        float beta1_t = powf(beta1, t);
        float beta2_t = powf(beta2, t);
        
        for (int i = 0; i < size; i++) {
            // Update biased first moment estimate
            m->data[i] = beta1 * m->data[i] + (1.0f - beta1) * gradients->data[i];
            
            // Update biased second raw moment estimate
            v->data[i] = beta2 * v->data[i] + (1.0f - beta2) * gradients->data[i] * gradients->data[i];
            
            // Compute bias-corrected estimates
            float m_hat = m->data[i] / (1.0f - beta1_t);
            float v_hat = v->data[i] / (1.0f - beta2_t);
            
            // Compute adaptive bounds
            float final_lr_t = final_lr * lr / sqrtf(v_hat + eps);
            float lower_bound = final_lr_t * (1.0f - 1.0f / (gamma * t + 1.0f));
            float upper_bound = final_lr_t * (1.0f + 1.0f / (gamma * t));
            
            // Compute step size
            float step_size = lr / sqrtf(v_hat + eps);
            step_size = fmaxf(step_size, lower_bound);
            step_size = fminf(step_size, upper_bound);
            
            // Update weights
            weights->data[i] -= step_size * m_hat;
        }
        
        if (weights->on_cuda) tensor_sync_to_cuda(weights);
    }
}

// RAdam: Rectified Adam
void radam_update(Tensor* weights, Tensor* gradients,
                 Tensor* m, Tensor* v,
                 float lr, float beta1, float beta2, float eps, int t, int size) {
    
    float rho_inf = 2.0f / (1.0f - beta2) - 1.0f;
    float beta2_t = powf(beta2, t);
    float rho_t = rho_inf - 2.0f * t * beta2_t / (1.0f - beta2_t);
    
    if (weights->on_cuda && gradients->on_cuda) {
#ifdef CUDA_AVAILABLE
        extern void aicraft_cuda_radam_kernel(float* weights, const float* gradients,
                                             float* m, float* v,
                                             float lr, float beta1, float beta2, float eps,
                                             int t, int size);
        
        dim3 block(256);
        dim3 grid((size + block.x - 1) / block.x);
        aicraft_cuda_radam_kernel<<<grid, block>>>(weights->cuda_data, gradients->cuda_data,
                                                  m->cuda_data, v->cuda_data,
                                                  lr, beta1, beta2, eps, t, size);
#endif
    } else {
        tensor_sync_to_cpu(weights);
        tensor_sync_to_cpu(gradients);
        tensor_sync_to_cpu(m);
        tensor_sync_to_cpu(v);
        
        float beta1_t = powf(beta1, t);
        
        for (int i = 0; i < size; i++) {
            // Update biased first moment estimate
            m->data[i] = beta1 * m->data[i] + (1.0f - beta1) * gradients->data[i];
            
            // Update biased second raw moment estimate
            v->data[i] = beta2 * v->data[i] + (1.0f - beta2) * gradients->data[i] * gradients->data[i];
            
            // Bias correction for first moment
            float m_hat = m->data[i] / (1.0f - beta1_t);
            
            if (rho_t > 4.0f) {
                // Compute variance rectification term
                float var_rect = sqrtf((rho_t - 4.0f) * (rho_t - 2.0f) * rho_inf / 
                                     ((rho_inf - 4.0f) * (rho_inf - 2.0f) * rho_t));
                
                // Bias correction for second moment
                float v_hat = sqrtf(v->data[i] / (1.0f - beta2_t)) + eps;
                
                // Update weights with variance rectification
                weights->data[i] -= lr * var_rect * m_hat / v_hat;
            } else {
                // Fall back to momentum update
                weights->data[i] -= lr * m_hat;
            }
        }
        
        if (weights->on_cuda) tensor_sync_to_cuda(weights);
    }
}

// LAMB: Layer-wise Adaptive Moments optimizer for Batch training
void lamb_update(Tensor* weights, Tensor* gradients,
                Tensor* m, Tensor* v,
                float lr, float beta1, float beta2, float eps,
                float weight_decay, int t, int size) {
    
    if (weights->on_cuda && gradients->on_cuda) {
#ifdef CUDA_AVAILABLE
        extern void aicraft_cuda_lamb_kernel(float* weights, const float* gradients,
                                            float* m, float* v,
                                            float lr, float beta1, float beta2, float eps,
                                            float weight_decay, int t, int size);
        
        dim3 block(256);
        dim3 grid((size + block.x - 1) / block.x);
        aicraft_cuda_lamb_kernel<<<grid, block>>>(weights->cuda_data, gradients->cuda_data,
                                                 m->cuda_data, v->cuda_data,
                                                 lr, beta1, beta2, eps, weight_decay, t, size);
#endif
    } else {
        tensor_sync_to_cpu(weights);
        tensor_sync_to_cpu(gradients);
        tensor_sync_to_cpu(m);
        tensor_sync_to_cpu(v);
        
        float beta1_t = powf(beta1, t);
        float beta2_t = powf(beta2, t);
        
        // Compute norms
        float weight_norm = 0.0f;
        float grad_norm = 0.0f;
        
        for (int i = 0; i < size; i++) {
            weight_norm += weights->data[i] * weights->data[i];
            grad_norm += gradients->data[i] * gradients->data[i];
        }
        weight_norm = sqrtf(weight_norm);
        grad_norm = sqrtf(grad_norm);
        
        // Compute update
        float update_norm = 0.0f;
        float* update = (float*)malloc(size * sizeof(float));
        
        for (int i = 0; i < size; i++) {
            // Apply weight decay
            float grad_with_decay = gradients->data[i] + weight_decay * weights->data[i];
            
            // Update biased first moment estimate
            m->data[i] = beta1 * m->data[i] + (1.0f - beta1) * grad_with_decay;
            
            // Update biased second raw moment estimate
            v->data[i] = beta2 * v->data[i] + (1.0f - beta2) * grad_with_decay * grad_with_decay;
            
            // Compute bias-corrected estimates
            float m_hat = m->data[i] / (1.0f - beta1_t);
            float v_hat = v->data[i] / (1.0f - beta2_t);
            
            // Compute update
            update[i] = m_hat / (sqrtf(v_hat) + eps);
            update_norm += update[i] * update[i];
        }
        update_norm = sqrtf(update_norm);
        
        // Compute trust ratio
        float trust_ratio = 1.0f;
        if (weight_norm > 0 && update_norm > 0) {
            trust_ratio = weight_norm / update_norm;
        }
        
        // Apply update with trust ratio
        for (int i = 0; i < size; i++) {
            weights->data[i] -= lr * trust_ratio * update[i];
        }
        
        free(update);
        
        if (weights->on_cuda) tensor_sync_to_cuda(weights);
    }
}

// Mixed Precision Training Support
typedef struct {
    float loss_scale;
    float growth_factor;
    int growth_interval;
    int consecutive_unskipped_steps;
    bool dynamic_scaling;
} LossScaler;

void init_loss_scaler(LossScaler* scaler, float initial_scale) {
    scaler->loss_scale = initial_scale;
    scaler->growth_factor = 2.0f;
    scaler->growth_interval = 2000;
    scaler->consecutive_unskipped_steps = 0;
    scaler->dynamic_scaling = true;
}

bool check_gradients_finite(Tensor* gradients) {
    tensor_sync_to_cpu(gradients);
    
    int size = gradients->rows * gradients->cols;
    for (int i = 0; i < size; i++) {
        if (!isfinite(gradients->data[i])) {
            return false;
        }
    }
    return true;
}

void scale_gradients(Tensor* gradients, float scale) {
    if (gradients->on_cuda) {
#ifdef CUDA_AVAILABLE
        extern void aicraft_cuda_scale_gradients_kernel(float* gradients, float scale, int size);
        
        int size = gradients->rows * gradients->cols;
        dim3 block(256);
        dim3 grid((size + block.x - 1) / block.x);
        aicraft_cuda_scale_gradients_kernel<<<grid, block>>>(gradients->cuda_data, scale, size);
#endif
    } else {
        int size = gradients->rows * gradients->cols;
        for (int i = 0; i < size; i++) {
            gradients->data[i] *= scale;
        }
    }
}

void unscale_gradients(Tensor* gradients, float scale) {
    scale_gradients(gradients, 1.0f / scale);
}

bool step_loss_scaler(LossScaler* scaler, bool gradients_finite) {
    if (gradients_finite) {
        scaler->consecutive_unskipped_steps++;
        
        // Increase loss scale if we've had enough consecutive successful steps
        if (scaler->dynamic_scaling && 
            scaler->consecutive_unskipped_steps >= scaler->growth_interval) {
            scaler->loss_scale *= scaler->growth_factor;
            scaler->consecutive_unskipped_steps = 0;
        }
        return true;
    } else {
        // Decrease loss scale on overflow
        if (scaler->dynamic_scaling) {
            scaler->loss_scale /= scaler->growth_factor;
            scaler->consecutive_unskipped_steps = 0;
        }
        return false;
    }
}

// Gradient Clipping
void clip_gradients_by_norm(Model* model, float max_norm) {
    float total_norm = 0.0f;
    
    // Compute total gradient norm
    Layer* layer = model->layers;
    while (layer) {
        if (layer->type == LAYER_DENSE) {
            tensor_sync_to_cpu(&layer->grad_weights);
            tensor_sync_to_cpu(&layer->grad_bias);
            
            int weight_size = layer->grad_weights.rows * layer->grad_weights.cols;
            int bias_size = layer->grad_bias.rows * layer->grad_bias.cols;
            
            for (int i = 0; i < weight_size; i++) {
                total_norm += layer->grad_weights.data[i] * layer->grad_weights.data[i];
            }
            for (int i = 0; i < bias_size; i++) {
                total_norm += layer->grad_bias.data[i] * layer->grad_bias.data[i];
            }
        }
        layer = layer->next;
    }
    
    total_norm = sqrtf(total_norm);
    
    // Clip gradients if necessary
    if (total_norm > max_norm) {
        float scale = max_norm / total_norm;
        
        layer = model->layers;
        while (layer) {
            if (layer->type == LAYER_DENSE) {
                scale_gradients(&layer->grad_weights, scale);
                scale_gradients(&layer->grad_bias, scale);
            }
            layer = layer->next;
        }
        
        aicraft_log(LOG_DEBUG, "[AiCraft] Gradients clipped: norm %.6f -> %.6f", 
                   total_norm, max_norm);
    }
}

// Exponential Moving Average (EMA) for model weights
typedef struct {
    bool enabled;
    float decay;
    Tensor** ema_weights;
    int num_tensors;
} EMAState;

void init_ema(EMAState* ema, Model* model, float decay) {
    ema->enabled = true;
    ema->decay = decay;
    
    // Count total tensors
    int tensor_count = 0;
    Layer* layer = model->layers;
    while (layer) {
        if (layer->type == LAYER_DENSE) {
            tensor_count += 2; // weights + bias
        }
        layer = layer->next;
    }
    
    ema->num_tensors = tensor_count;
    ema->ema_weights = (Tensor**)malloc(tensor_count * sizeof(Tensor*));
    
    // Initialize EMA tensors
    int idx = 0;
    layer = model->layers;
    while (layer) {
        if (layer->type == LAYER_DENSE) {
            ema->ema_weights[idx] = (Tensor*)malloc(sizeof(Tensor));
            *ema->ema_weights[idx] = tensor_copy(layer->weights);
            idx++;
            
            ema->ema_weights[idx] = (Tensor*)malloc(sizeof(Tensor));
            *ema->ema_weights[idx] = tensor_copy(layer->bias);
            idx++;
        }
        layer = layer->next;
    }
    
    aicraft_log(LOG_INFO, "[AiCraft] EMA initialized with decay %.6f", decay);
}

void update_ema(EMAState* ema, Model* model) {
    if (!ema->enabled) return;
    
    int idx = 0;
    Layer* layer = model->layers;
    while (layer) {
        if (layer->type == LAYER_DENSE) {
            // Update EMA weights: ema = decay * ema + (1-decay) * current
            tensor_sync_to_cpu(ema->ema_weights[idx]);
            tensor_sync_to_cpu(&layer->weights);
            
            int size = layer->weights.rows * layer->weights.cols;
            for (int i = 0; i < size; i++) {
                ema->ema_weights[idx]->data[i] = ema->decay * ema->ema_weights[idx]->data[i] + 
                                               (1.0f - ema->decay) * layer->weights.data[i];
            }
            
            if (layer->weights.on_cuda) tensor_sync_to_cuda(ema->ema_weights[idx]);
            idx++;
            
            // Update EMA bias
            tensor_sync_to_cpu(ema->ema_weights[idx]);
            tensor_sync_to_cpu(&layer->bias);
            
            size = layer->bias.rows * layer->bias.cols;
            for (int i = 0; i < size; i++) {
                ema->ema_weights[idx]->data[i] = ema->decay * ema->ema_weights[idx]->data[i] + 
                                               (1.0f - ema->decay) * layer->bias.data[i];
            }
            
            if (layer->bias.on_cuda) tensor_sync_to_cuda(ema->ema_weights[idx]);
            idx++;
        }
        layer = layer->next;
    }
}

// Advanced learning rate schedulers
float cosine_annealing_with_warm_restart(float initial_lr, int current_step, 
                                        int t_max, int t_mult, int restart_count) {
    // Compute current period
    int period = t_max;
    int step_in_period = current_step;
    
    for (int i = 0; i < restart_count; i++) {
        if (step_in_period >= period) {
            step_in_period -= period;
            period *= t_mult;
        } else {
            break;
        }
    }
    
    return initial_lr * 0.5f * (1.0f + cosf(M_PI * step_in_period / period));
}

float polynomial_decay_lr(float initial_lr, int current_step, int total_steps, float power) {
    if (current_step >= total_steps) {
        return 0.0f;
    }
    return initial_lr * powf(1.0f - (float)current_step / total_steps, power);
}

float linear_warmup_cosine_decay_lr(float initial_lr, int current_step, 
                                   int warmup_steps, int total_steps) {
    if (current_step < warmup_steps) {
        // Linear warmup
        return initial_lr * (float)current_step / warmup_steps;
    } else {
        // Cosine decay
        int decay_steps = total_steps - warmup_steps;
        int step_in_decay = current_step - warmup_steps;
        return initial_lr * 0.5f * (1.0f + cosf(M_PI * step_in_decay / decay_steps));
    }
}

// Performance profiling
typedef struct {
    clock_t start_time;
    double total_time;
    int call_count;
    char name[64];
} ProfilerEntry;

typedef struct {
    ProfilerEntry* entries;
    int num_entries;
    int capacity;
} Profiler;

static Profiler g_profiler = {NULL, 0, 0};

void profiler_start(const char* name) {
    // Find or create entry
    int entry_idx = -1;
    for (int i = 0; i < g_profiler.num_entries; i++) {
        if (strcmp(g_profiler.entries[i].name, name) == 0) {
            entry_idx = i;
            break;
        }
    }
    
    if (entry_idx == -1) {
        // Create new entry
        if (g_profiler.num_entries >= g_profiler.capacity) {
            g_profiler.capacity = g_profiler.capacity == 0 ? 16 : g_profiler.capacity * 2;
            g_profiler.entries = (ProfilerEntry*)realloc(g_profiler.entries, 
                                                        g_profiler.capacity * sizeof(ProfilerEntry));
        }
        
        entry_idx = g_profiler.num_entries++;
        strcpy(g_profiler.entries[entry_idx].name, name);
        g_profiler.entries[entry_idx].total_time = 0.0;
        g_profiler.entries[entry_idx].call_count = 0;
    }
    
    g_profiler.entries[entry_idx].start_time = clock();
}

void profiler_end(const char* name) {
    clock_t end_time = clock();
    
    for (int i = 0; i < g_profiler.num_entries; i++) {
        if (strcmp(g_profiler.entries[i].name, name) == 0) {
            double elapsed = (double)(end_time - g_profiler.entries[i].start_time) / CLOCKS_PER_SEC;
            g_profiler.entries[i].total_time += elapsed;
            g_profiler.entries[i].call_count++;
            break;
        }
    }
}

void profiler_print_summary() {
    printf("\n=== PERFORMANCE PROFILER SUMMARY ===\n");
    printf("┌─────────────────────────┬───────────┬───────────┬───────────┐\n");
    printf("│ Operation               │ Calls     │ Total (s) │ Avg (ms)  │\n");
    printf("├─────────────────────────┼───────────┼───────────┼───────────┤\n");
    
    for (int i = 0; i < g_profiler.num_entries; i++) {
        ProfilerEntry* entry = &g_profiler.entries[i];
        double avg_time = entry->total_time / entry->call_count * 1000; // ms
        
        printf("│ %-23s │ %9d │ %9.3f │ %9.3f │\n",
               entry->name, entry->call_count, entry->total_time, avg_time);
    }
    printf("└─────────────────────────┴───────────┴───────────┴───────────┘\n");
}

void profiler_reset() {
    for (int i = 0; i < g_profiler.num_entries; i++) {
        g_profiler.entries[i].total_time = 0.0;
        g_profiler.entries[i].call_count = 0;
    }
}
