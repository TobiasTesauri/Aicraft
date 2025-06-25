#include "training.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// === GRAPH OPTIMIZATION AND TENSOR FUSION ===

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
    
    // Operation-specific parameters
    union {
        struct { int M, N, K; } matmul;
        struct { float dropout_rate; } dropout;
        struct { float eps; } layernorm;
        struct { int num_heads; int head_dim; } attention;
    } params;
    
    // Performance metrics
    double execution_time;
    size_t memory_usage;
    bool can_fuse;
    
    struct Operation* next;
} Operation;

typedef struct {
    Operation* operations;
    int num_operations;
    int* tensor_ids;
    int num_tensors;
    bool optimized;
} ComputationGraph;

// Graph optimization functions
ComputationGraph* create_computation_graph() {
    ComputationGraph* graph = (ComputationGraph*)malloc(sizeof(ComputationGraph));
    graph->operations = NULL;
    graph->num_operations = 0;
    graph->tensor_ids = NULL;
    graph->num_tensors = 0;
    graph->optimized = false;
    return graph;
}

void add_operation(ComputationGraph* graph, OperationType type, 
                  int* input_ids, int num_inputs, int output_id) {
    Operation* op = (Operation*)malloc(sizeof(Operation));
    op->type = type;
    op->input_ids = (int*)malloc(num_inputs * sizeof(int));
    memcpy(op->input_ids, input_ids, num_inputs * sizeof(int));
    op->num_inputs = num_inputs;
    op->output_id = output_id;
    op->execution_time = 0.0;
    op->memory_usage = 0;
    op->can_fuse = true;
    op->next = NULL;
    
    // Add to graph
    if (graph->operations == NULL) {
        graph->operations = op;
    } else {
        Operation* current = graph->operations;
        while (current->next) current = current->next;
        current->next = op;
    }
    graph->num_operations++;
}

// Fusion patterns
bool can_fuse_linear_activation(Operation* linear_op, Operation* activation_op) {
    if (linear_op->type != OP_MATMUL) return false;
    if (linear_op->output_id != activation_op->input_ids[0]) return false;
    
    return (activation_op->type == OP_RELU || 
            activation_op->type == OP_GELU ||
            activation_op->type == OP_SIGMOID);
}

bool can_fuse_layernorm_activation(Operation* norm_op, Operation* activation_op) {
    if (norm_op->type != OP_LAYERNORM) return false;
    if (norm_op->output_id != activation_op->input_ids[0]) return false;
    
    return (activation_op->type == OP_GELU || activation_op->type == OP_RELU);
}

// Fused operations implementation
Tensor fused_linear_relu(Tensor input, Tensor weight, Tensor bias) {
    profiler_start("fused_linear_relu");
    
    int batch_size = input.rows;
    int input_size = input.cols;
    int output_size = weight.cols;
    
    Tensor output = tensor_create(batch_size, output_size, 
                                 input.on_cuda ? BACKEND_CUDA : BACKEND_CPU);
    
    if (input.on_cuda && weight.on_cuda && bias.on_cuda) {
#ifdef CUDA_AVAILABLE
        extern void aicraft_cuda_fused_linear_relu_kernel(float* output, const float* input,
                                                         const float* weight, const float* bias,
                                                         int batch_size, int input_size, 
                                                         int output_size);
        
        dim3 block(16, 16);
        dim3 grid((output_size + block.x - 1) / block.x, 
                 (batch_size + block.y - 1) / block.y);
        
        aicraft_cuda_fused_linear_relu_kernel<<<grid, block>>>(
            output.cuda_data, input.cuda_data, weight.cuda_data, bias.cuda_data,
            batch_size, input_size, output_size);
#endif
    } else {
        // CPU implementation
        tensor_sync_to_cpu(&input);
        tensor_sync_to_cpu(&weight);
        tensor_sync_to_cpu(&bias);
        if (output.on_cuda) tensor_to_cpu(&output);
        
        for (int b = 0; b < batch_size; b++) {
            for (int o = 0; o < output_size; o++) {
                float sum = bias.data[o];
                for (int i = 0; i < input_size; i++) {
                    sum += input.data[b * input_size + i] * weight.data[i * output_size + o];
                }
                // Apply ReLU
                output.data[b * output_size + o] = fmaxf(0.0f, sum);
            }
        }
        
        if (input.on_cuda) tensor_to_cuda(&output);
    }
    
    profiler_end("fused_linear_relu");
    return output;
}

Tensor fused_linear_gelu(Tensor input, Tensor weight, Tensor bias) {
    profiler_start("fused_linear_gelu");
    
    int batch_size = input.rows;
    int input_size = input.cols;
    int output_size = weight.cols;
    
    Tensor output = tensor_create(batch_size, output_size,
                                 input.on_cuda ? BACKEND_CUDA : BACKEND_CPU);
    
    if (input.on_cuda && weight.on_cuda && bias.on_cuda) {
#ifdef CUDA_AVAILABLE
        extern void aicraft_cuda_fused_linear_gelu_kernel(float* output, const float* input,
                                                         const float* weight, const float* bias,
                                                         int batch_size, int input_size, 
                                                         int output_size);
        
        dim3 block(16, 16);
        dim3 grid((output_size + block.x - 1) / block.x,
                 (batch_size + block.y - 1) / block.y);
        
        aicraft_cuda_fused_linear_gelu_kernel<<<grid, block>>>(
            output.cuda_data, input.cuda_data, weight.cuda_data, bias.cuda_data,
            batch_size, input_size, output_size);
#endif
    } else {
        // CPU implementation with fused linear + GELU
        tensor_sync_to_cpu(&input);
        tensor_sync_to_cpu(&weight);
        tensor_sync_to_cpu(&bias);
        if (output.on_cuda) tensor_to_cpu(&output);
        
        const float sqrt_2_pi = sqrtf(2.0f / M_PI);
        
        for (int b = 0; b < batch_size; b++) {
            for (int o = 0; o < output_size; o++) {
                float sum = bias.data[o];
                for (int i = 0; i < input_size; i++) {
                    sum += input.data[b * input_size + i] * weight.data[i * output_size + o];
                }
                
                // Apply GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
                float x = sum;
                float tanh_arg = sqrt_2_pi * (x + 0.044715f * x * x * x);
                output.data[b * output_size + o] = 0.5f * x * (1.0f + tanhf(tanh_arg));
            }
        }
        
        if (input.on_cuda) tensor_to_cuda(&output);
    }
    
    profiler_end("fused_linear_gelu");
    return output;
}

Tensor fused_layernorm_gelu(Tensor input, Tensor gamma, Tensor beta, float eps) {
    profiler_start("fused_layernorm_gelu");
    
    Tensor output = tensor_create(input.rows, input.cols,
                                 input.on_cuda ? BACKEND_CUDA : BACKEND_CPU);
    
    if (input.on_cuda && gamma.on_cuda && beta.on_cuda) {
#ifdef CUDA_AVAILABLE
        extern void aicraft_cuda_fused_layernorm_gelu_kernel(float* output, const float* input,
                                                            const float* gamma, const float* beta,
                                                            float eps, int batch_size, 
                                                            int hidden_size);
        
        dim3 block(256);
        dim3 grid((input.rows + block.x - 1) / block.x);
        
        aicraft_cuda_fused_layernorm_gelu_kernel<<<grid, block>>>(
            output.cuda_data, input.cuda_data, gamma.cuda_data, beta.cuda_data,
            eps, input.rows, input.cols);
#endif
    } else {
        // CPU implementation
        Tensor layernorm_out = layer_norm_forward(input, gamma, beta, eps);
        
        // Apply GELU
        tensor_sync_to_cpu(&layernorm_out);
        if (output.on_cuda) tensor_to_cpu(&output);
        
        const float sqrt_2_pi = sqrtf(2.0f / M_PI);
        int size = input.rows * input.cols;
        
        for (int i = 0; i < size; i++) {
            float x = layernorm_out.data[i];
            float tanh_arg = sqrt_2_pi * (x + 0.044715f * x * x * x);
            output.data[i] = 0.5f * x * (1.0f + tanhf(tanh_arg));
        }
        
        tensor_free(layernorm_out);
        
        if (input.on_cuda) tensor_to_cuda(&output);
    }
    
    profiler_end("fused_layernorm_gelu");
    return output;
}

// Graph optimization algorithms
void optimize_computation_graph(ComputationGraph* graph) {
    if (graph->optimized) return;
    
    aicraft_log(LOG_INFO, "[AiCraft] Optimizing computation graph with %d operations...", 
               graph->num_operations);
    
    int fusions = 0;
    
    // Pattern matching for fusion opportunities
    Operation* current = graph->operations;
    while (current && current->next) {
        Operation* next = current->next;
        
        // Try to fuse linear + activation
        if (can_fuse_linear_activation(current, next)) {
            if (next->type == OP_RELU) {
                current->type = OP_FUSED_LINEAR_RELU;
                strcpy(current->name, "fused_linear_relu");
            } else if (next->type == OP_GELU) {
                current->type = OP_FUSED_LINEAR_GELU;
                strcpy(current->name, "fused_linear_gelu");
            }
            
            // Remove the activation operation
            current->next = next->next;
            free(next->input_ids);
            free(next);
            graph->num_operations--;
            fusions++;
            
            aicraft_log(LOG_DEBUG, "[AiCraft] Fused linear + activation operations");
            continue;
        }
        
        // Try to fuse layernorm + activation
        if (can_fuse_layernorm_activation(current, next)) {
            current->type = OP_FUSED_LAYERNORM_GELU;
            strcpy(current->name, "fused_layernorm_gelu");
            
            // Remove the activation operation
            current->next = next->next;
            free(next->input_ids);
            free(next);
            graph->num_operations--;
            fusions++;
            
            aicraft_log(LOG_DEBUG, "[AiCraft] Fused layernorm + GELU operations");
            continue;
        }
        
        current = current->next;
    }
    
    graph->optimized = true;
    aicraft_log(LOG_INFO, "[AiCraft] Graph optimization complete: %d fusions applied", fusions);
}

// Memory layout optimization
void optimize_memory_layout(Tensor* tensors, int num_tensors) {
    profiler_start("memory_layout_optimization");
    
    // Analyze memory access patterns
    size_t total_memory = 0;
    for (int i = 0; i < num_tensors; i++) {
        total_memory += tensors[i].rows * tensors[i].cols * sizeof(float);
    }
    
    // Sort tensors by size (largest first) for better memory coalescing
    for (int i = 0; i < num_tensors - 1; i++) {
        for (int j = i + 1; j < num_tensors; j++) {
            size_t size_i = tensors[i].rows * tensors[i].cols;
            size_t size_j = tensors[j].rows * tensors[j].cols;
            
            if (size_j > size_i) {
                Tensor temp = tensors[i];
                tensors[i] = tensors[j];
                tensors[j] = temp;
            }
        }
    }
    
    aicraft_log(LOG_DEBUG, "[AiCraft] Memory layout optimized for %d tensors (%.2f MB total)",
               num_tensors, (float)total_memory / (1024 * 1024));
    
    profiler_end("memory_layout_optimization");
}

// Dead code elimination
void eliminate_dead_operations(ComputationGraph* graph) {
    bool* is_used = (bool*)calloc(graph->num_operations, sizeof(bool));
    
    // Mark operations that produce outputs used by other operations
    Operation* current = graph->operations;
    int op_idx = 0;
    
    while (current) {
        // Check if this operation's output is used
        Operation* check = graph->operations;
        while (check) {
            for (int i = 0; i < check->num_inputs; i++) {
                if (check->input_ids[i] == current->output_id) {
                    is_used[op_idx] = true;
                    break;
                }
            }
            if (is_used[op_idx]) break;
            check = check->next;
        }
        
        current = current->next;
        op_idx++;
    }
    
    // Remove unused operations
    Operation* prev = NULL;
    current = graph->operations;
    op_idx = 0;
    int removed = 0;
    
    while (current) {
        if (!is_used[op_idx]) {
            if (prev) {
                prev->next = current->next;
            } else {
                graph->operations = current->next;
            }
            
            Operation* to_remove = current;
            current = current->next;
            
            free(to_remove->input_ids);
            free(to_remove);
            graph->num_operations--;
            removed++;
        } else {
            prev = current;
            current = current->next;
        }
        op_idx++;
    }
    
    free(is_used);
    
    if (removed > 0) {
        aicraft_log(LOG_INFO, "[AiCraft] Eliminated %d dead operations", removed);
    }
}

// Multi-stream execution scheduling
typedef struct {
    cudaStream_t stream;
    Operation** operations;
    int num_operations;
    int capacity;
} ExecutionStream;

void schedule_operations_multi_stream(ComputationGraph* graph, int num_streams) {
#ifdef CUDA_AVAILABLE
    if (num_streams <= 1) return;
    
    ExecutionStream* streams = (ExecutionStream*)malloc(num_streams * sizeof(ExecutionStream));
    
    // Initialize streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i].stream);
        streams[i].operations = NULL;
        streams[i].num_operations = 0;
        streams[i].capacity = 0;
    }
    
    // Simple round-robin scheduling (can be improved with dependency analysis)
    Operation* current = graph->operations;
    int stream_idx = 0;
    
    while (current) {
        ExecutionStream* stream = &streams[stream_idx];
        
        // Add operation to stream
        if (stream->num_operations >= stream->capacity) {
            stream->capacity = stream->capacity == 0 ? 8 : stream->capacity * 2;
            stream->operations = (Operation**)realloc(stream->operations,
                                                    stream->capacity * sizeof(Operation*));
        }
        
        stream->operations[stream->num_operations++] = current;
        
        current = current->next;
        stream_idx = (stream_idx + 1) % num_streams;
    }
    
    aicraft_log(LOG_INFO, "[AiCraft] Scheduled operations across %d CUDA streams", num_streams);
    
    // Cleanup
    for (int i = 0; i < num_streams; i++) {
        free(streams[i].operations);
        cudaStreamDestroy(streams[i].stream);
    }
    free(streams);
#endif
}

// Automatic kernel tuning
typedef struct {
    int block_x, block_y;
    int shared_memory;
    float execution_time;
} KernelConfig;

KernelConfig auto_tune_kernel(const char* kernel_name, int problem_size) {
    KernelConfig best_config = {256, 1, 0, INFINITY};
    
    // Common block size configurations to test
    int block_sizes[][2] = {
        {64, 1}, {128, 1}, {256, 1}, {512, 1},
        {16, 16}, {32, 32}, {16, 32}, {32, 16}
    };
    int num_configs = sizeof(block_sizes) / sizeof(block_sizes[0]);
    
    for (int i = 0; i < num_configs; i++) {
        KernelConfig config;
        config.block_x = block_sizes[i][0];
        config.block_y = block_sizes[i][1];
        config.shared_memory = 0;
        
        // Benchmark this configuration
        clock_t start = clock();
        
        // TODO: Launch kernel with this configuration
        // This would involve running the actual kernel multiple times
        
        clock_t end = clock();
        config.execution_time = (double)(end - start) / CLOCKS_PER_SEC;
        
        if (config.execution_time < best_config.execution_time) {
            best_config = config;
        }
    }
    
    aicraft_log(LOG_DEBUG, "[AiCraft] Auto-tuned %s: block(%d,%d) -> %.4f ms",
               kernel_name, best_config.block_x, best_config.block_y, 
               best_config.execution_time * 1000);
    
    return best_config;
}

// Comprehensive graph execution
void execute_optimized_graph(ComputationGraph* graph, Tensor* tensors) {
    profiler_start("graph_execution");
    
    if (!graph->optimized) {
        optimize_computation_graph(graph);
        eliminate_dead_operations(graph);
    }
    
    Operation* current = graph->operations;
    while (current) {
        profiler_start(current->name);
        
        // Execute based on operation type
        switch (current->type) {
            case OP_FUSED_LINEAR_RELU: {
                // Execute fused linear + ReLU
                Tensor input = tensors[current->input_ids[0]];
                Tensor weight = tensors[current->input_ids[1]];
                Tensor bias = tensors[current->input_ids[2]];
                tensors[current->output_id] = fused_linear_relu(input, weight, bias);
                break;
            }
            
            case OP_FUSED_LINEAR_GELU: {
                // Execute fused linear + GELU
                Tensor input = tensors[current->input_ids[0]];
                Tensor weight = tensors[current->input_ids[1]];
                Tensor bias = tensors[current->input_ids[2]];
                tensors[current->output_id] = fused_linear_gelu(input, weight, bias);
                break;
            }
            
            case OP_FUSED_LAYERNORM_GELU: {
                // Execute fused layernorm + GELU
                Tensor input = tensors[current->input_ids[0]];
                Tensor gamma = tensors[current->input_ids[1]];
                Tensor beta = tensors[current->input_ids[2]];
                tensors[current->output_id] = fused_layernorm_gelu(input, gamma, beta, 1e-5f);
                break;
            }
            
            default:
                // Execute regular operations
                break;
        }
        
        profiler_end(current->name);
        current = current->next;
    }
    
    profiler_end("graph_execution");
}
