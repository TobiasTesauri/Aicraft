#include "aicraft_advanced.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// === ADVANCED TRAINING IMPLEMENTATION ===

// External functions from other modules
extern void adabound_update(Tensor* weights, Tensor* gradients, Tensor* m, Tensor* v,
                           float lr, float beta1, float beta2, float eps,
                           float final_lr, float gamma, int t, int size);
extern void radam_update(Tensor* weights, Tensor* gradients, Tensor* m, Tensor* v,
                        float lr, float beta1, float beta2, float eps, int t, int size);
extern void lamb_update(Tensor* weights, Tensor* gradients, Tensor* m, Tensor* v,
                       float lr, float beta1, float beta2, float eps, 
                       float weight_decay, int t, int size);
extern void clip_gradients_by_norm(Tensor* gradients, int num_tensors, float max_norm);
extern QuantizedTensor quantize_tensor(Tensor tensor, int8_t qmin, int8_t qmax);
extern GraphOptimizer* create_graph_optimizer(void);
extern void optimize_graph(GraphOptimizer* optimizer);

AdvancedTrainingState* create_advanced_training_state(Model* model, AdvancedTrainingConfig config) {
    aicraft_log(LOG_INFO, "[AiCraft] Inizializzazione stato di training avanzato...");
    
    AdvancedTrainingState* state = (AdvancedTrainingState*)malloc(sizeof(AdvancedTrainingState));
    if (!state) {
        aicraft_log(LOG_ERROR, "[AiCraft] Errore nell'allocazione dello stato di training");
        return NULL;
    }
    
    state->model = model;
    state->config = config;
    state->current_epoch = 0;
    state->best_val_accuracy = -1.0f;
    state->best_epoch = -1;
    state->patience_counter = 0;
    state->should_stop = false;
    state->total_train_time = 0.0;
    state->total_val_time = 0.0;
    state->peak_memory_usage = 0;
    
    // Allocate metrics arrays
    state->metrics_size = config.epochs;
    state->train_losses = (float*)calloc(config.epochs, sizeof(float));
    state->train_accuracies = (float*)calloc(config.epochs, sizeof(float));
    state->val_losses = (float*)calloc(config.epochs, sizeof(float));
    state->val_accuracies = (float*)calloc(config.epochs, sizeof(float));
    state->learning_rates = (float*)calloc(config.epochs, sizeof(float));
    
    // Initialize mixed precision if enabled
    state->mixed_precision = NULL;
    if (config.mixed_precision.enabled) {
        aicraft_log(LOG_INFO, "[AiCraft] Inizializzazione mixed precision training...");
        state->mixed_precision = create_mixed_precision_state(config.mixed_precision);
    }
    
    // Initialize EMA if enabled
    state->ema_state = NULL;
    if (config.optimizer.use_ema) {
        aicraft_log(LOG_INFO, "[AiCraft] Inizializzazione Exponential Moving Average...");
        // Count number of weight tensors in model
        int num_weights = 0;
        Layer* layer = model->layers;
        while (layer) {
            if (layer->type == LAYER_DENSE) {
                num_weights += 2; // weights + bias
            }
            layer = layer->next;
        }
        state->ema_state = create_ema_state(num_weights, config.optimizer.ema_decay);
    }
    
    // Initialize profiler if enabled
    state->profiler = NULL;
    if (config.use_profiler) {
        aicraft_log(LOG_INFO, "[AiCraft] Inizializzazione profiler...");
        state->profiler = create_profiler(config.log_dir);
    }
    
    // Initialize graph optimizer if enabled
    state->graph_optimizer = NULL;
    if (config.use_graph_optimization) {
        aicraft_log(LOG_INFO, "[AiCraft] Inizializzazione graph optimizer...");
        state->graph_optimizer = create_graph_optimizer();
        // Add model operations to graph
        Layer* layer = model->layers;
        while (layer) {
            add_operation(state->graph_optimizer, OP_MATMUL, layer->name);
            if (layer->activation_fn) {
                if (layer->activation_fn == tensor_relu) {
                    add_operation(state->graph_optimizer, OP_RELU, layer->name);
                } else if (layer->activation_fn == tensor_gelu) {
                    add_operation(state->graph_optimizer, OP_GELU, layer->name);
                }
            }
            layer = layer->next;
        }
        optimize_graph(state->graph_optimizer);
    }
    
    // Initialize best weights storage
    state->best_weights = NULL;
    state->num_best_weights = 0;
    
    aicraft_log(LOG_INFO, "[AiCraft] Stato di training avanzato inizializzato con successo");
    return state;
}

static void save_best_weights(AdvancedTrainingState* state) {
    // Free previous best weights
    if (state->best_weights) {
        for (int i = 0; i < state->num_best_weights; i++) {
            tensor_free(state->best_weights[i]);
        }
        free(state->best_weights);
    }
    
    // Count and copy current weights
    int weight_count = 0;
    Layer* layer = state->model->layers;
    while (layer) {
        if (layer->type == LAYER_DENSE) {
            weight_count += 2; // weights + bias
        }
        layer = layer->next;
    }
    
    state->best_weights = (Tensor*)malloc(weight_count * sizeof(Tensor));
    state->num_best_weights = weight_count;
    
    int idx = 0;
    layer = state->model->layers;
    while (layer) {
        if (layer->type == LAYER_DENSE) {
            state->best_weights[idx++] = tensor_copy(layer->weights);
            state->best_weights[idx++] = tensor_copy(layer->bias);
        }
        layer = layer->next;
    }
}

static void restore_best_weights(AdvancedTrainingState* state) {
    if (!state->best_weights) return;
    
    int idx = 0;
    Layer* layer = state->model->layers;
    while (layer) {
        if (layer->type == LAYER_DENSE) {
            // Copy data from best weights
            memcpy(layer->weights.data, state->best_weights[idx].data, 
                   layer->weights.rows * layer->weights.cols * sizeof(float));
            memcpy(layer->bias.data, state->best_weights[idx + 1].data,
                   layer->bias.rows * layer->bias.cols * sizeof(float));
            idx += 2;
        }
        layer = layer->next;
    }
    
    aicraft_log(LOG_INFO, "[AiCraft] Pesi migliori ripristinati dall'epoca %d", state->best_epoch);
}

static float compute_accuracy(Tensor predictions, Tensor targets, int num_samples) {
    int correct = 0;
    
    for (int i = 0; i < num_samples; i++) {
        if (targets.cols == 1) {
            // Binary classification or regression
            float pred = tensor_get(predictions, i, 0);
            float target = tensor_get(targets, i, 0);
            if (fabs(pred - target) < 0.5f) correct++;
        } else {
            // Multi-class classification
            int pred_class = 0;
            int target_class = 0;
            float max_pred = tensor_get(predictions, i, 0);
            float max_target = tensor_get(targets, i, 0);
            
            for (int j = 1; j < targets.cols; j++) {
                float pred_val = tensor_get(predictions, i, j);
                float target_val = tensor_get(targets, i, j);
                if (pred_val > max_pred) {
                    max_pred = pred_val;
                    pred_class = j;
                }
                if (target_val > max_target) {
                    max_target = target_val;
                    target_class = j;
                }
            }
            
            if (pred_class == target_class) correct++;
        }
    }
    
    return (float)correct / num_samples;
}

static void update_learning_rate(AdvancedTrainingState* state, int epoch) {
    float lr = state->config.learning_rate;
    
    if (state->config.use_lr_scheduler) {
        OptimizerV2* opt = &state->config.optimizer;
        
        if (opt->use_cosine_decay) {
            lr = cosine_annealing_lr(opt->initial_lr, opt->initial_lr * 0.01f, 
                                   epoch, state->config.epochs);
        } else if (opt->decay_rate > 0.0f) {
            lr = exponential_decay_lr(opt->initial_lr, opt->decay_rate, 
                                    epoch, opt->decay_steps);
        }
        
        if (opt->use_warm_restart && opt->warm_restart_period > 0) {
            lr = warm_restart_lr(lr, epoch, opt->warm_restart_period, 1.5f);
        }
    }
    
    state->config.optimizer.learning_rate = lr;
    state->learning_rates[epoch] = lr;
}

static void advanced_optimizer_step(AdvancedTrainingState* state, Layer* layer) {
    OptimizerV2* opt = &state->config.optimizer;
    int size = layer->weights.rows * layer->weights.cols;
    
    // Apply gradient clipping if enabled
    if (opt->clip_gradients) {
        Tensor* grads[] = {&layer->grad_weights, &layer->grad_bias};
        clip_gradients_by_norm(grads, 2, opt->max_grad_norm);
    }
    
    // Mixed precision scaling
    if (state->mixed_precision && state->mixed_precision->enabled) {
        float scale = 1.0f / state->mixed_precision->loss_scale;
        tensor_scalar_mul(layer->grad_weights, scale);
        tensor_scalar_mul(layer->grad_bias, scale);
    }
    
    // Apply optimizer update
    switch (opt->type) {
        case OPT_ADABOUND:
            adabound_update(&layer->weights, &layer->grad_weights,
                           &layer->m_weights, &layer->v_weights,
                           opt->learning_rate, opt->beta1, opt->beta2, opt->epsilon,
                           opt->final_lr, opt->gamma, opt->t, size);
            break;
            
        case OPT_RADAM:
            radam_update(&layer->weights, &layer->grad_weights,
                        &layer->m_weights, &layer->v_weights,
                        opt->learning_rate, opt->beta1, opt->beta2, opt->epsilon,
                        opt->t, size);
            break;
            
        case OPT_LAMB:
            lamb_update(&layer->weights, &layer->grad_weights,
                       &layer->m_weights, &layer->v_weights,
                       opt->learning_rate, opt->beta1, opt->beta2, opt->epsilon,
                       opt->weight_decay, opt->t, size);
            break;
            
        default:
            // Fall back to standard optimizers
            optimizer_step(&state->model->optimizer, layer);
            break;
    }
    
    // Update EMA if enabled
    if (state->ema_state && state->ema_state->enabled) {
        Tensor* weights[] = {&layer->weights, &layer->bias};
        update_ema(state->ema_state, weights, 2);
    }
}

void advanced_train_model(AdvancedTrainingState* state, Tensor* train_inputs, Tensor* train_targets, 
                         int num_train_samples, Tensor* val_inputs, Tensor* val_targets, int num_val_samples) {
    
    aicraft_log(LOG_INFO, "[AiCraft] Avvio training avanzato per %d epoche...", state->config.epochs);
    
    Model* model = state->model;
    model->training_mode = true;
    
    // Training loop
    for (int epoch = 0; epoch < state->config.epochs && !state->should_stop; epoch++) {
        if (state->profiler) {
            PROFILE_START(state->profiler, "epoch_training");
        }
        
        double epoch_start = get_elapsed_time();
        state->current_epoch = epoch;
        
        // Update learning rate
        update_learning_rate(state, epoch);
        
        float epoch_loss = 0.0f;
        int num_batches = (num_train_samples + state->config.batch_size - 1) / state->config.batch_size;
        
        // Training phase
        for (int batch = 0; batch < num_batches; batch++) {
            int batch_start = batch * state->config.batch_size;
            int batch_end = fminf(batch_start + state->config.batch_size, num_train_samples);
            int batch_size = batch_end - batch_start;
            
            if (state->profiler) {
                PROFILE_START(state->profiler, "forward_pass");
            }
            
            // Forward pass for batch
            float batch_loss = 0.0f;
            for (int i = batch_start; i < batch_end; i++) {
                Tensor prediction = model_forward(model, train_inputs[i]);
                float loss = compute_loss(prediction, train_targets[i], model->loss_type);
                batch_loss += loss;
                
                // Scale loss for mixed precision
                if (state->mixed_precision && state->mixed_precision->enabled) {
                    scale_loss(state->mixed_precision, &prediction);
                }
                
                // Backward pass
                model_backward(model, train_inputs[i], train_targets[i], prediction);
                tensor_free(prediction);
            }
            
            if (state->profiler) {
                PROFILE_END(state->profiler, "forward_pass");
                PROFILE_START(state->profiler, "optimizer_step");
            }
            
            // Check for overflow in mixed precision
            bool overflow = false;
            if (state->mixed_precision && state->mixed_precision->enabled) {
                Layer* layer = model->layers;
                int grad_count = 0;
                Tensor* all_grads = NULL;
                
                // Collect all gradients
                while (layer) {
                    if (layer->type == LAYER_DENSE) {
                        grad_count += 2;
                    }
                    layer = layer->next;
                }
                
                if (grad_count > 0) {
                    all_grads = (Tensor*)malloc(grad_count * sizeof(Tensor));
                    int idx = 0;
                    layer = model->layers;
                    while (layer) {
                        if (layer->type == LAYER_DENSE) {
                            all_grads[idx++] = layer->grad_weights;
                            all_grads[idx++] = layer->grad_bias;
                        }
                        layer = layer->next;
                    }
                    
                    overflow = check_overflow(state->mixed_precision, all_grads, grad_count);
                    free(all_grads);
                }
            }
            
            // Update weights if no overflow
            if (!overflow) {
                Layer* layer = model->layers;
                while (layer) {
                    if (layer->type == LAYER_DENSE) {
                        advanced_optimizer_step(state, layer);
                    }
                    layer = layer->next;
                }
                
                // Update optimizer time step
                state->config.optimizer.t++;
            }
            
            // Update loss scale
            if (state->mixed_precision && state->mixed_precision->enabled) {
                update_loss_scale(state->mixed_precision, overflow);
            }
            
            if (state->profiler) {
                PROFILE_END(state->profiler, "optimizer_step");
            }
            
            epoch_loss += batch_loss / batch_size;
        }
        
        epoch_loss /= num_batches;
        state->train_losses[epoch] = epoch_loss;
        
        // Compute training accuracy
        float train_accuracy = 0.0f;
        for (int i = 0; i < fminf(1000, num_train_samples); i++) { // Sample for efficiency
            Tensor prediction = model_forward(model, train_inputs[i]);
            train_accuracy += compute_accuracy(prediction, train_targets[i], 1);
            tensor_free(prediction);
        }
        train_accuracy /= fminf(1000, num_train_samples);
        state->train_accuracies[epoch] = train_accuracy;
        
        if (state->profiler) {
            PROFILE_END(state->profiler, "epoch_training");
        }
        
        // Validation phase
        if (val_inputs && val_targets && epoch % state->config.validation_frequency == 0) {
            if (state->profiler) {
                PROFILE_START(state->profiler, "validation");
            }
            
            model->training_mode = false;
            float val_loss = 0.0f;
            float val_accuracy = 0.0f;
            
            for (int i = 0; i < num_val_samples; i++) {
                Tensor prediction = model_forward(model, val_inputs[i]);
                val_loss += compute_loss(prediction, val_targets[i], model->loss_type);
                val_accuracy += compute_accuracy(prediction, val_targets[i], 1);
                tensor_free(prediction);
            }
            
            val_loss /= num_val_samples;
            val_accuracy /= num_val_samples;
            state->val_losses[epoch] = val_loss;
            state->val_accuracies[epoch] = val_accuracy;
            
            model->training_mode = true;
            
            // Check for best model
            if (val_accuracy > state->best_val_accuracy) {
                state->best_val_accuracy = val_accuracy;
                state->best_epoch = epoch;
                save_best_weights(state);
                state->patience_counter = 0;
                
                aicraft_log(LOG_INFO, "[AiCraft] Nuovo miglior modello salvato (epoca %d, accuratezza: %.4f)", 
                           epoch, val_accuracy);
            } else {
                state->patience_counter++;
                
                // Early stopping
                if (state->config.use_early_stopping && 
                    state->patience_counter >= state->config.patience) {
                    aicraft_log(LOG_INFO, "[AiCraft] Early stopping attivato dopo %d epoche senza miglioramenti", 
                               state->patience_counter);
                    state->should_stop = true;
                }
            }
            
            if (state->profiler) {
                PROFILE_END(state->profiler, "validation");
            }
        }
        
        // Logging
        if (epoch % state->config.log_frequency == 0) {
            double epoch_time = get_elapsed_time() - epoch_start;
            state->total_train_time += epoch_time;
            
            aicraft_log(LOG_INFO, "[AiCraft] Epoca %d/%d - Loss: %.6f, Acc: %.4f, Val Loss: %.6f, Val Acc: %.4f, LR: %.6f, Tempo: %.2fs", 
                       epoch + 1, state->config.epochs, epoch_loss, train_accuracy,
                       state->val_losses[epoch], state->val_accuracies[epoch],
                       state->learning_rates[epoch], epoch_time);
        }
        
        // TensorBoard logging
        if (state->profiler && state->profiler->tensorboard_enabled) {
            profiler_save_tensorboard(state->profiler, epoch);
        }
        
        // Memory monitoring
        size_t current_memory = tensor_memory_usage(train_inputs[0]) * num_train_samples;
        if (current_memory > state->peak_memory_usage) {
            state->peak_memory_usage = current_memory;
        }
    }
    
    // Restore best weights if early stopping was used
    if (state->config.use_early_stopping && state->best_weights) {
        restore_best_weights(state);
    }
    
    // Apply EMA weights if enabled
    if (state->ema_state && state->ema_state->enabled) {
        Layer* layer = model->layers;
        while (layer) {
            if (layer->type == LAYER_DENSE) {
                Tensor* weights[] = {&layer->weights, &layer->bias};
                apply_ema(state->ema_state, weights, 2);
            }
            layer = layer->next;
        }
        aicraft_log(LOG_INFO, "[AiCraft] Pesi EMA applicati al modello finale");
    }
    
    aicraft_log(LOG_INFO, "[AiCraft] Training completato dopo %d epoche", state->current_epoch + 1);
    aicraft_log(LOG_INFO, "[AiCraft] Migliore accuratezza di validazione: %.4f (epoca %d)", 
               state->best_val_accuracy, state->best_epoch + 1);
    
    // Print final profiler report
    if (state->profiler) {
        profiler_print_report(state->profiler);
    }
}

void advanced_save_checkpoint(AdvancedTrainingState* state, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        aicraft_log(LOG_ERROR, "[AiCraft] Impossibile salvare checkpoint: %s", filename);
        return;
    }
    
    // Save training state
    fwrite(&state->current_epoch, sizeof(int), 1, file);
    fwrite(&state->best_val_accuracy, sizeof(float), 1, file);
    fwrite(&state->best_epoch, sizeof(int), 1, file);
    fwrite(&state->patience_counter, sizeof(int), 1, file);
    
    // Save metrics
    fwrite(state->train_losses, sizeof(float), state->metrics_size, file);
    fwrite(state->train_accuracies, sizeof(float), state->metrics_size, file);
    fwrite(state->val_losses, sizeof(float), state->metrics_size, file);
    fwrite(state->val_accuracies, sizeof(float), state->metrics_size, file);
    fwrite(state->learning_rates, sizeof(float), state->metrics_size, file);
    
    // Save optimizer state
    fwrite(&state->config.optimizer, sizeof(OptimizerV2), 1, file);
    
    fclose(file);
    aicraft_log(LOG_INFO, "[AiCraft] Checkpoint salvato: %s", filename);
}

void advanced_load_checkpoint(AdvancedTrainingState* state, const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        aicraft_log(LOG_ERROR, "[AiCraft] Impossibile caricare checkpoint: %s", filename);
        return;
    }
    
    // Load training state
    fread(&state->current_epoch, sizeof(int), 1, file);
    fread(&state->best_val_accuracy, sizeof(float), 1, file);
    fread(&state->best_epoch, sizeof(int), 1, file);
    fread(&state->patience_counter, sizeof(int), 1, file);
    
    // Load metrics
    fread(state->train_losses, sizeof(float), state->metrics_size, file);
    fread(state->train_accuracies, sizeof(float), state->metrics_size, file);
    fread(state->val_losses, sizeof(float), state->metrics_size, file);
    fread(state->val_accuracies, sizeof(float), state->metrics_size, file);
    fread(state->learning_rates, sizeof(float), state->metrics_size, file);
    
    // Load optimizer state
    fread(&state->config.optimizer, sizeof(OptimizerV2), 1, file);
    
    fclose(file);
    aicraft_log(LOG_INFO, "[AiCraft] Checkpoint caricato: %s", filename);
}

void free_advanced_training_state(AdvancedTrainingState* state) {
    if (!state) return;
    
    // Free metrics arrays
    free(state->train_losses);
    free(state->train_accuracies);
    free(state->val_losses);
    free(state->val_accuracies);
    free(state->learning_rates);
    
    // Free best weights
    if (state->best_weights) {
        for (int i = 0; i < state->num_best_weights; i++) {
            tensor_free(state->best_weights[i]);
        }
        free(state->best_weights);
    }
    
    // Free advanced components
    if (state->mixed_precision) {
        free_mixed_precision_state(state->mixed_precision);
    }
    
    if (state->ema_state) {
        free_ema_state(state->ema_state);
    }
    
    if (state->profiler) {
        free_profiler(state->profiler);
    }
    
    if (state->graph_optimizer) {
        free_graph_optimizer(state->graph_optimizer);
    }
    
    free(state);
    aicraft_log(LOG_INFO, "[AiCraft] Stato di training avanzato liberato");
}
