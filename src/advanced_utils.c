#include "aicraft_advanced.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// === MIXED PRECISION IMPLEMENTATION ===

MixedPrecisionState* create_mixed_precision_state(MixedPrecisionConfig config) {
    MixedPrecisionState* state = (MixedPrecisionState*)malloc(sizeof(MixedPrecisionState));
    if (!state) {
        aicraft_log(LOG_ERROR, "[AiCraft] Errore nell'allocazione dello stato mixed precision");
        return NULL;
    }
    
    state->enabled = config.enabled;
    state->compute_type = config.compute_type;
    state->storage_type = config.storage_type;
    state->loss_scale = config.loss_scale;
    state->loss_scale_growth_factor = config.loss_scale_growth_factor;
    state->loss_scale_growth_interval = config.loss_scale_growth_interval;
    state->consecutive_unskipped_steps = 0;
    state->dynamic_loss_scaling = config.dynamic_loss_scaling;
    
    state->overflow_detected = false;
    state->overflow_count = 0;
    state->fp16_time = 0.0;
    state->fp32_time = 0.0;
    state->memory_saved = 0;
    
    aicraft_log(LOG_INFO, "[AiCraft] Mixed precision state creato (loss_scale: %.1f)", state->loss_scale);
    return state;
}

void scale_loss(MixedPrecisionState* state, Tensor* loss) {
    if (!state->enabled) return;
    
    tensor_scalar_mul(*loss, state->loss_scale);
}

bool check_overflow(MixedPrecisionState* state, Tensor* gradients, int num_tensors) {
    if (!state->enabled) return false;
    
    for (int i = 0; i < num_tensors; i++) {
        float* data = gradients[i].data;
        int size = gradients[i].rows * gradients[i].cols;
        
        for (int j = 0; j < size; j++) {
            if (!isfinite(data[j]) || fabs(data[j]) > 1e4f) {
                state->overflow_detected = true;
                state->overflow_count++;
                return true;
            }
        }
    }
    
    state->overflow_detected = false;
    return false;
}

void update_loss_scale(MixedPrecisionState* state, bool overflow) {
    if (!state->enabled || !state->dynamic_loss_scaling) return;
    
    if (overflow) {
        state->loss_scale /= 2.0f;
        state->consecutive_unskipped_steps = 0;
        aicraft_log(LOG_DEBUG, "[AiCraft] Loss scale ridotto a %.1f (overflow)", state->loss_scale);
    } else {
        state->consecutive_unskipped_steps++;
        if (state->consecutive_unskipped_steps > state->loss_scale_growth_interval) {
            state->loss_scale *= state->loss_scale_growth_factor;
            state->consecutive_unskipped_steps = 0;
            aicraft_log(LOG_DEBUG, "[AiCraft] Loss scale aumentato a %.1f", state->loss_scale);
        }
    }
    
    // Clamp loss scale
    state->loss_scale = fmaxf(1.0f, fminf(65536.0f, state->loss_scale));
}

void free_mixed_precision_state(MixedPrecisionState* state) {
    if (state) {
        free(state);
    }
}

Tensor mixed_precision_forward(Tensor input, Tensor weights, Tensor bias, DataType compute_type) {
    // Mixed precision forward pass
    if (compute_type == DATA_FP16 && input.on_cuda) {
        // Use FP16 computation on CUDA
        #ifdef CUDA_AVAILABLE
        extern Tensor cuda_mixed_precision_linear(Tensor input, Tensor weights, Tensor bias);
        return cuda_mixed_precision_linear(input, weights, bias);
        #endif
    }
    
    // Fallback to standard FP32 computation
    Tensor output = tensor_matmul(input, weights);
    if (bias.data || bias.cuda_data) {
        Tensor result = tensor_add(output, bias);
        tensor_free(output);
        return result;
    }
    return output;
}

// === EMA IMPLEMENTATION ===

EMAState* create_ema_state(int num_weights, float decay) {
    EMAState* state = (EMAState*)malloc(sizeof(EMAState));
    if (!state) return NULL;
    
    state->enabled = true;
    state->decay = decay;
    state->num_weights = num_weights;
    state->step_count = 0;
    
    state->shadow_weights = (Tensor*)malloc(num_weights * sizeof(Tensor));
    for (int i = 0; i < num_weights; i++) {
        // Initialize as empty tensors, will be set during first update
        state->shadow_weights[i] = (Tensor){0};
    }
    
    aicraft_log(LOG_INFO, "[AiCraft] EMA state creato (decay: %.4f)", decay);
    return state;
}

void update_ema(EMAState* ema, Tensor* weights, int num_weights) {
    if (!ema->enabled) return;
    
    float decay = ema->decay;
    
    // Use bias correction for early steps
    if (ema->step_count < 100) {
        decay = fminf(decay, (1.0f + ema->step_count) / (10.0f + ema->step_count));
    }
    
    for (int i = 0; i < num_weights; i++) {
        if (ema->shadow_weights[i].data == NULL) {
            // Initialize shadow weights
            ema->shadow_weights[i] = tensor_copy(weights[i]);
        } else {
            // EMA update: shadow = decay * shadow + (1 - decay) * weights
            tensor_scalar_mul(ema->shadow_weights[i], decay);
            Tensor scaled_weights = tensor_scalar_mul(weights[i], 1.0f - decay);
            Tensor new_shadow = tensor_add(ema->shadow_weights[i], scaled_weights);
            
            tensor_free(ema->shadow_weights[i]);
            tensor_free(scaled_weights);
            ema->shadow_weights[i] = new_shadow;
        }
    }
    
    ema->step_count++;
}

void apply_ema(EMAState* ema, Tensor* weights, int num_weights) {
    if (!ema->enabled) return;
    
    for (int i = 0; i < num_weights; i++) {
        if (ema->shadow_weights[i].data) {
            // Copy shadow weights to current weights
            memcpy(weights[i].data, ema->shadow_weights[i].data,
                   weights[i].rows * weights[i].cols * sizeof(float));
        }
    }
}

void free_ema_state(EMAState* ema) {
    if (!ema) return;
    
    if (ema->shadow_weights) {
        for (int i = 0; i < ema->num_weights; i++) {
            if (ema->shadow_weights[i].data) {
                tensor_free(ema->shadow_weights[i]);
            }
        }
        free(ema->shadow_weights);
    }
    
    free(ema);
}

// === LEARNING RATE SCHEDULING ===

float cosine_annealing_lr(float initial_lr, float min_lr, int current_step, int total_steps) {
    if (current_step >= total_steps) return min_lr;
    
    float progress = (float)current_step / total_steps;
    float lr = min_lr + (initial_lr - min_lr) * (1.0f + cosf(M_PI * progress)) / 2.0f;
    return lr;
}

float polynomial_decay_lr(float initial_lr, float final_lr, int current_step, int total_steps, float power) {
    if (current_step >= total_steps) return final_lr;
    
    float progress = (float)current_step / total_steps;
    float lr = (initial_lr - final_lr) * powf(1.0f - progress, power) + final_lr;
    return lr;
}

float exponential_decay_lr(float initial_lr, float decay_rate, int current_step, int decay_steps) {
    if (decay_steps <= 0) return initial_lr;
    
    float decay_factor = powf(decay_rate, (float)current_step / decay_steps);
    return initial_lr * decay_factor;
}

float warm_restart_lr(float initial_lr, int current_step, int restart_period, float t_mult) {
    if (restart_period <= 0) return initial_lr;
    
    int cycle = current_step / restart_period;
    int step_in_cycle = current_step % restart_period;
    
    float current_period = restart_period * powf(t_mult, cycle);
    float progress = step_in_cycle / current_period;
    
    return initial_lr * (1.0f + cosf(M_PI * progress)) / 2.0f;
}

// === PROFILER IMPLEMENTATION ===

ProfilerState* create_profiler(const char* log_dir) {
    ProfilerState* profiler = (ProfilerState*)malloc(sizeof(ProfilerState));
    if (!profiler) return NULL;
    
    profiler->entries = (ProfilerEntry*)malloc(100 * sizeof(ProfilerEntry));
    profiler->num_entries = 0;
    profiler->capacity = 100;
    profiler->enabled = true;
    profiler->total_time = 0.0;
    profiler->total_memory = 0;
    
    profiler->tensorboard_enabled = (log_dir != NULL);
    if (log_dir) {
        strncpy(profiler->log_dir, log_dir, sizeof(profiler->log_dir) - 1);
        profiler->log_dir[sizeof(profiler->log_dir) - 1] = '\0';
        
        char log_path[512];
        snprintf(log_path, sizeof(log_path), "%s/events.txt", log_dir);
        profiler->log_file = fopen(log_path, "w");
        if (profiler->log_file) {
            fprintf(profiler->log_file, "# AiCraft Training Log\n");
            fprintf(profiler->log_file, "step,train_loss,train_acc,val_loss,val_acc,learning_rate,time\n");
        }
    } else {
        profiler->log_file = NULL;
    }
    
    aicraft_log(LOG_INFO, "[AiCraft] Profiler creato (TensorBoard: %s)", 
               profiler->tensorboard_enabled ? "attivo" : "disattivo");
    return profiler;
}

void profiler_start(ProfilerState* profiler, const char* name) {
    if (!profiler || !profiler->enabled) return;
    
    // Find or create entry
    ProfilerEntry* entry = NULL;
    for (int i = 0; i < profiler->num_entries; i++) {
        if (strcmp(profiler->entries[i].name, name) == 0) {
            entry = &profiler->entries[i];
            break;
        }
    }
    
    if (!entry) {
        if (profiler->num_entries >= profiler->capacity) {
            profiler->capacity *= 2;
            profiler->entries = (ProfilerEntry*)realloc(profiler->entries, 
                                                       profiler->capacity * sizeof(ProfilerEntry));
        }
        
        entry = &profiler->entries[profiler->num_entries++];
        strncpy(entry->name, name, sizeof(entry->name) - 1);
        entry->name[sizeof(entry->name) - 1] = '\0';
        entry->total_time = 0.0;
        entry->call_count = 0;
        entry->memory_used = 0;
        entry->peak_memory = 0;
    }
    
    entry->start_time = get_elapsed_time();
}

void profiler_end(ProfilerState* profiler, const char* name) {
    if (!profiler || !profiler->enabled) return;
    
    double end_time = get_elapsed_time();
    
    for (int i = 0; i < profiler->num_entries; i++) {
        if (strcmp(profiler->entries[i].name, name) == 0) {
            profiler->entries[i].total_time += end_time - profiler->entries[i].start_time;
            profiler->entries[i].call_count++;
            break;
        }
    }
}

void profiler_log_memory(ProfilerState* profiler, const char* name, size_t memory) {
    if (!profiler || !profiler->enabled) return;
    
    for (int i = 0; i < profiler->num_entries; i++) {
        if (strcmp(profiler->entries[i].name, name) == 0) {
            profiler->entries[i].memory_used = memory;
            if (memory > profiler->entries[i].peak_memory) {
                profiler->entries[i].peak_memory = memory;
            }
            break;
        }
    }
}

void profiler_print_report(ProfilerState* profiler) {
    if (!profiler) return;
    
    printf("\n=== PROFILER REPORT ===\n");
    printf("%-30s %10s %10s %15s %15s\n", "Operation", "Calls", "Total(s)", "Avg(ms)", "Memory(MB)");
    printf("%-30s %10s %10s %15s %15s\n", "----------", "-----", "--------", "-------", "---------");
    
    double total_time = 0.0;
    for (int i = 0; i < profiler->num_entries; i++) {
        ProfilerEntry* entry = &profiler->entries[i];
        double avg_time = (entry->call_count > 0) ? (entry->total_time / entry->call_count * 1000.0) : 0.0;
        double memory_mb = entry->peak_memory / (1024.0 * 1024.0);
        
        printf("%-30s %10d %10.3f %15.3f %15.1f\n",
               entry->name, entry->call_count, entry->total_time, avg_time, memory_mb);
        
        total_time += entry->total_time;
    }
    
    printf("%-30s %10s %10.3f %15s %15.1f\n", "TOTAL", "", total_time, "", 
           profiler->total_memory / (1024.0 * 1024.0));
    printf("=======================\n\n");
}

void profiler_save_tensorboard(ProfilerState* profiler, int step) {
    if (!profiler || !profiler->log_file) return;
    
    // This is a simplified TensorBoard-like logging
    // In a full implementation, you'd write proper TensorBoard format
    fprintf(profiler->log_file, "%d", step);
    for (int i = 0; i < profiler->num_entries; i++) {
        fprintf(profiler->log_file, ",%.6f", profiler->entries[i].total_time);
    }
    fprintf(profiler->log_file, "\n");
    fflush(profiler->log_file);
}

void free_profiler(ProfilerState* profiler) {
    if (!profiler) return;
    
    if (profiler->entries) {
        free(profiler->entries);
    }
    
    if (profiler->log_file) {
        fclose(profiler->log_file);
    }
    
    free(profiler);
}

// === UTILITY FUNCTIONS ===

void print_system_info(void) {
    printf("\n=== SYSTEM INFORMATION ===\n");
    printf("AiCraft Deep Learning Framework\n");
    printf("Version: 2.0 Advanced\n");
    printf("Built: %s %s\n", __DATE__, __TIME__);
    
#ifdef __CUDACC__
    printf("CUDA Support: Enabled\n");
#else
    printf("CUDA Support: Disabled\n");
#endif

#ifdef _WIN32
    printf("Platform: Windows\n");
#elif __linux__
    printf("Platform: Linux\n");
#elif __APPLE__
    printf("Platform: macOS\n");
#else
    printf("Platform: Unknown\n");
#endif

    printf("Backend: %s\n", 
           g_default_backend == BACKEND_CUDA ? "CUDA" : 
           g_default_backend == BACKEND_CPU ? "CPU" : "AUTO");
    
    printf("Features:\n");
    printf("  - Mixed Precision Training\n");
    printf("  - Advanced Optimizers (AdaBound, RAdam, LAMB)\n");
    printf("  - Quantization (INT8 Inference)\n");
    printf("  - Graph Optimization\n");
    printf("  - Tensor Fusion\n");
    printf("  - Exponential Moving Average\n");
    printf("  - Gradient Clipping\n");
    printf("  - Learning Rate Scheduling\n");
    printf("  - Profiling & TensorBoard\n");
    printf("===========================\n\n");
}

void print_model_summary(Model* model) {
    if (!model) return;
    
    printf("\n=== MODEL SUMMARY ===\n");
    printf("Model: %s\n", model->name);
    printf("Backend: %s\n", 
           model->backend == BACKEND_CUDA ? "CUDA" : 
           model->backend == BACKEND_CPU ? "CPU" : "AUTO");
    printf("Training Mode: %s\n", model->training_mode ? "True" : "False");
    printf("Loss Function: %s\n", 
           model->loss_type == LOSS_MSE ? "MSE" : 
           model->loss_type == LOSS_CROSSENTROPY ? "CrossEntropy" : 
           model->loss_type == LOSS_BINARY_CROSSENTROPY ? "BinaryCrossEntropy" : "Unknown");
    
    printf("\nLayers:\n");
    printf("%-20s %-15s %-15s %-15s\n", "Name", "Type", "Input", "Output");
    printf("%-20s %-15s %-15s %-15s\n", "----", "----", "-----", "------");
    
    Layer* layer = model->layers;
    int total_params = 0;
    
    while (layer) {
        const char* type_str = "";
        char input_str[32] = "";
        char output_str[32] = "";
        int layer_params = 0;
        
        switch (layer->type) {
            case LAYER_DENSE:
                type_str = "Dense";
                snprintf(input_str, sizeof(input_str), "%d", layer->input_size);
                snprintf(output_str, sizeof(output_str), "%d", layer->output_size);
                layer_params = layer->input_size * layer->output_size + layer->output_size;
                break;
            case LAYER_RELU:
                type_str = "ReLU";
                snprintf(input_str, sizeof(input_str), "%d", layer->input_size);
                snprintf(output_str, sizeof(output_str), "%d", layer->output_size);
                break;
            case LAYER_SIGMOID:
                type_str = "Sigmoid";
                snprintf(input_str, sizeof(input_str), "%d", layer->input_size);
                snprintf(output_str, sizeof(output_str), "%d", layer->output_size);
                break;
            case LAYER_SOFTMAX:
                type_str = "Softmax";
                snprintf(input_str, sizeof(input_str), "%d", layer->input_size);
                snprintf(output_str, sizeof(output_str), "%d", layer->output_size);
                break;
            default:
                type_str = "Unknown";
                break;
        }
        
        printf("%-20s %-15s %-15s %-15s\n", layer->name, type_str, input_str, output_str);
        total_params += layer_params;
        layer = layer->next;
    }
    
    printf("\nTotal Parameters: %d\n", total_params);
    printf("Model Memory: %.2f MB\n", tensor_memory_usage(model->layers->weights) / (1024.0 * 1024.0));
    printf("======================\n\n");
}

void shuffle_dataset(Tensor* inputs, Tensor* targets, int num_samples) {
    srand(time(NULL));
    
    for (int i = num_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        
        // Swap inputs
        Tensor temp_input = inputs[i];
        inputs[i] = inputs[j];
        inputs[j] = temp_input;
        
        // Swap targets
        Tensor temp_target = targets[i];
        targets[i] = targets[j];
        targets[j] = temp_target;
    }
}

void normalize_dataset(Tensor* inputs, int num_samples) {
    if (num_samples == 0) return;
    
    int input_size = inputs[0].rows * inputs[0].cols;
    
    // Compute mean and std
    float* means = (float*)calloc(input_size, sizeof(float));
    float* stds = (float*)calloc(input_size, sizeof(float));
    
    // Compute means
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < input_size; j++) {
            means[j] += inputs[i].data[j];
        }
    }
    
    for (int j = 0; j < input_size; j++) {
        means[j] /= num_samples;
    }
    
    // Compute standard deviations
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < input_size; j++) {
            float diff = inputs[i].data[j] - means[j];
            stds[j] += diff * diff;
        }
    }
    
    for (int j = 0; j < input_size; j++) {
        stds[j] = sqrtf(stds[j] / num_samples);
        if (stds[j] == 0.0f) stds[j] = 1.0f; // Avoid division by zero
    }
    
    // Normalize
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < input_size; j++) {
            inputs[i].data[j] = (inputs[i].data[j] - means[j]) / stds[j];
        }
    }
    
    free(means);
    free(stds);
    
    aicraft_log(LOG_INFO, "[AiCraft] Dataset normalizzato (%d campioni)", num_samples);
}
