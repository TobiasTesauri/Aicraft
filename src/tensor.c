#include "tensor.h"
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdarg.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

// Global state
DeviceInfo g_device_info = {0};
BackendType g_default_backend = BACKEND_AUTO;
static LogLevel g_log_level = LOG_INFO;
static bool g_initialized = false;

// Performance timing
static clock_t g_timer_start = 0;

// === INITIALIZATION AND BACKEND DETECTION ===

void aicraft_init(void) {
    if (g_initialized) return;
    
    aicraft_log(LOG_INFO, "[AiCraft] Inizializzazione sistema...");
    
    // Initialize device info
    memset(&g_device_info, 0, sizeof(DeviceInfo));
    
    // Detect available backends
    BackendType detected = aicraft_detect_backend();
    
    if (detected == BACKEND_CUDA) {
        g_default_backend = BACKEND_CUDA;
        aicraft_log(LOG_INFO, "[AiCraft] CUDA rilevato. Utilizzo accelerazione GPU.");
    } else {
        g_default_backend = BACKEND_CPU;
        aicraft_log(LOG_WARNING, "[AiCraft] CUDA non rilevato. Eseguo su CPU.");
    }
    
    aicraft_print_device_info();
    g_initialized = true;
}

BackendType aicraft_detect_backend(void) {
#ifdef CUDA_AVAILABLE
    // Try to initialize CUDA
    extern bool cuda_init_device(DeviceInfo* info);
    if (cuda_init_device(&g_device_info)) {
        g_device_info.cuda_available = true;
        return BACKEND_CUDA;
    }
#endif
    
    // Fallback to CPU
    g_device_info.cuda_available = false;
    g_device_info.cuda_device_count = 0;
    strcpy(g_device_info.device_name, "CPU (Fallback)");
    
    return BACKEND_CPU;
}

void aicraft_print_device_info(void) {
    aicraft_log(LOG_INFO, "[AiCraft] === Informazioni Dispositivo ===");
    aicraft_log(LOG_INFO, "[AiCraft] Backend: %s", 
                g_device_info.cuda_available ? "CUDA + CPU" : "CPU Only");
    
    if (g_device_info.cuda_available) {
        aicraft_log(LOG_INFO, "[AiCraft] GPU: %s", g_device_info.device_name);
        aicraft_log(LOG_INFO, "[AiCraft] Compute Capability: %d.%d", 
                    g_device_info.compute_capability_major, 
                    g_device_info.compute_capability_minor);
        aicraft_log(LOG_INFO, "[AiCraft] Memoria GPU: %.1f MB totale, %.1f MB libera",
                    g_device_info.total_memory / 1024.0 / 1024.0,
                    g_device_info.free_memory / 1024.0 / 1024.0);
    } else {
        aicraft_log(LOG_INFO, "[AiCraft] Processore: CPU multithreading");
    }
    aicraft_log(LOG_INFO, "[AiCraft] ================================");
}

void aicraft_cleanup(void) {
    if (!g_initialized) return;
    
    aicraft_log(LOG_INFO, "[AiCraft] Pulizia risorse...");
    
#ifdef CUDA_AVAILABLE
    if (g_device_info.cuda_available) {
        extern void cuda_cleanup(void);
        cuda_cleanup();
    }
#endif
    
    g_initialized = false;
}

// === TENSOR CREATION AND MANAGEMENT ===

Tensor tensor_create(int rows, int cols, BackendType backend) {
    if (!g_initialized) aicraft_init();
    
    Tensor tensor;
    memset(&tensor, 0, sizeof(Tensor));
    
    tensor.rows = rows;
    tensor.cols = cols;
    tensor.stride = cols;
    tensor.owns_data = true;
    tensor.ref_count = 1;
    
    // Determine actual backend
    if (backend == BACKEND_AUTO) {
        backend = g_default_backend;
    }
    
    tensor.backend = backend;
    
    size_t size = rows * cols * sizeof(float);
    
    if (backend == BACKEND_CUDA && g_device_info.cuda_available) {
#ifdef CUDA_AVAILABLE
        extern float* cuda_malloc(size_t size);
        tensor.cuda_data = cuda_malloc(size);
        if (tensor.cuda_data) {
            tensor.on_cuda = true;
            tensor.data = NULL;  // No CPU copy initially
        } else {
            // Fallback to CPU
            aicraft_log(LOG_WARNING, "[AiCraft] CUDA malloc fallito, uso CPU");
            tensor.data = (float*)malloc(size);
            tensor.on_cuda = false;
            tensor.backend = BACKEND_CPU;
        }
#endif
    } else {
        // CPU allocation
        tensor.data = (float*)malloc(size);
        tensor.on_cuda = false;
        tensor.cuda_data = NULL;
    }
    
    if (!tensor.data && !tensor.cuda_data) {
        aicraft_log(LOG_ERROR, "[AiCraft] Errore allocazione memoria per tensore %dx%d", rows, cols);
        exit(1);
    }
    
    return tensor;
}

void tensor_free(Tensor tensor) {
    if (tensor.owns_data) {
        if (tensor.data) {
            free(tensor.data);
        }
        
#ifdef CUDA_AVAILABLE
        if (tensor.cuda_data) {
            extern void cuda_free(float* ptr);
            cuda_free(tensor.cuda_data);
        }
#endif
    }
}

// === MEMORY MANAGEMENT ===

void tensor_to_cuda(Tensor* tensor) {
#ifdef CUDA_AVAILABLE
    if (!g_device_info.cuda_available || tensor->on_cuda) return;
    
    extern float* cuda_malloc(size_t size);
    extern void cuda_memcpy_to_device(float* dst, const float* src, size_t size);
    
    size_t size = tensor->rows * tensor->cols * sizeof(float);
    tensor->cuda_data = cuda_malloc(size);
    
    if (tensor->cuda_data) {
        if (tensor->data) {
            cuda_memcpy_to_device(tensor->cuda_data, tensor->data, size);
            if (tensor->owns_data) {
                free(tensor->data);
            }
            tensor->data = NULL;
        }
        tensor->on_cuda = true;
        tensor->backend = BACKEND_CUDA;
    } else {
        aicraft_log(LOG_ERROR, "[AiCraft] Impossibile trasferire tensore su GPU");
    }
#else
    aicraft_log(LOG_WARNING, "[AiCraft] CUDA non disponibile, tensore rimane su CPU");
#endif
}

void tensor_to_cpu(Tensor* tensor) {
    if (!tensor->on_cuda) return;
    
#ifdef CUDA_AVAILABLE
    extern void cuda_memcpy_to_host(float* dst, const float* src, size_t size);
    extern void cuda_free(float* ptr);
    
    size_t size = tensor->rows * tensor->cols * sizeof(float);
    tensor->data = (float*)malloc(size);
    
    if (tensor->data) {
        cuda_memcpy_to_host(tensor->data, tensor->cuda_data, size);
        cuda_free(tensor->cuda_data);
        tensor->cuda_data = NULL;
        tensor->on_cuda = false;
        tensor->backend = BACKEND_CPU;
    }
#endif
}

// === BASIC OPERATIONS (CPU IMPLEMENTATIONS) ===

void tensor_fill(Tensor* tensor, float value) {
    if (tensor->on_cuda) {
#ifdef CUDA_AVAILABLE
        extern void cuda_fill(float* data, float value, int size);
        cuda_fill(tensor->cuda_data, value, tensor->rows * tensor->cols);
#endif
    } else {
        for (int i = 0; i < tensor->rows * tensor->cols; i++) {
            tensor->data[i] = value;
        }
    }
}

void tensor_zero(Tensor* tensor) {
    tensor_fill(tensor, 0.0f);
}

Tensor tensor_random(int rows, int cols, float min_val, float max_val, BackendType backend) {
    if (!g_initialized) aicraft_init();
    
    Tensor tensor = tensor_create(rows, cols, backend);
    
    // Initialize random seed if not already done
    static bool seed_initialized = false;
    if (!seed_initialized) {
        srand((unsigned int)time(NULL));
        seed_initialized = true;
    }
    
    if (tensor.on_cuda) {
        // For CUDA tensors, generate on CPU first then copy to GPU
        float* temp_data = (float*)malloc(rows * cols * sizeof(float));
        for (int i = 0; i < rows * cols; i++) {
            float random_val = (float)rand() / RAND_MAX;
            temp_data[i] = min_val + random_val * (max_val - min_val);
        }
        
#ifdef CUDA_AVAILABLE
        extern void cuda_memcpy_to_device(float* dst, const float* src, size_t size);
        cuda_memcpy_to_device(tensor.cuda_data, temp_data, rows * cols * sizeof(float));
#endif
        free(temp_data);
    } else {
        // CPU tensor - fill directly
        for (int i = 0; i < rows * cols; i++) {
            float random_val = (float)rand() / RAND_MAX;
            tensor.data[i] = min_val + random_val * (max_val - min_val);
        }
    }
    
    return tensor;
}

Tensor tensor_add(Tensor a, Tensor b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        aicraft_log(LOG_ERROR, "[AiCraft] Dimensioni incompatibili per addizione: %dx%d vs %dx%d", 
                    a.rows, a.cols, b.rows, b.cols);
        exit(1);
    }
    
    Tensor result = tensor_create(a.rows, a.cols, a.backend);
    
    if (result.on_cuda && a.on_cuda && b.on_cuda) {
#ifdef CUDA_AVAILABLE
        extern void cuda_tensor_add(const float* a, const float* b, float* c, int size);
        cuda_tensor_add(a.cuda_data, b.cuda_data, result.cuda_data, a.rows * a.cols);
#endif
    } else {
        // Ensure data is on CPU
        if (a.on_cuda) tensor_sync_to_cpu((Tensor*)&a);
        if (b.on_cuda) tensor_sync_to_cpu((Tensor*)&b);
        if (result.on_cuda) tensor_to_cpu(&result);
        
        for (int i = 0; i < a.rows * a.cols; i++) {
            result.data[i] = a.data[i] + b.data[i];
        }
    }
    
    return result;
}

Tensor tensor_matmul(Tensor a, Tensor b) {
    if (a.cols != b.rows) {
        aicraft_log(LOG_ERROR, "[AiCraft] Dimensioni incompatibili per moltiplicazione: %dx%d * %dx%d", 
                    a.rows, a.cols, b.rows, b.cols);
        exit(1);
    }
    
    Tensor result = tensor_create(a.rows, b.cols, a.backend);
    
    if (result.on_cuda && a.on_cuda && b.on_cuda) {
#ifdef CUDA_AVAILABLE
        extern void cuda_gemm(const float* A, const float* B, float* C, int M, int N, int K);
        cuda_gemm(a.cuda_data, b.cuda_data, result.cuda_data, a.rows, b.cols, a.cols);
#endif
    } else {
        // CPU implementation
        if (a.on_cuda) tensor_sync_to_cpu((Tensor*)&a);
        if (b.on_cuda) tensor_sync_to_cpu((Tensor*)&b);
        if (result.on_cuda) tensor_to_cpu(&result);
        
        // Simple CPU GEMM
        tensor_zero(&result);
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < b.cols; j++) {
                for (int k = 0; k < a.cols; k++) {
                    result.data[i * result.cols + j] += 
                        a.data[i * a.cols + k] * b.data[k * b.cols + j];
                }
            }
        }
    }
    
    return result;
}

Tensor tensor_relu(Tensor input) {
    Tensor result = tensor_create(input.rows, input.cols, input.backend);
    
    if (result.on_cuda && input.on_cuda) {
#ifdef CUDA_AVAILABLE
        extern void cuda_relu(const float* input, float* output, int size);
        cuda_relu(input.cuda_data, result.cuda_data, input.rows * input.cols);
#endif
    } else {
        if (input.on_cuda) tensor_sync_to_cpu((Tensor*)&input);
        if (result.on_cuda) tensor_to_cpu(&result);
        
        for (int i = 0; i < input.rows * input.cols; i++) {
            result.data[i] = fmaxf(0.0f, input.data[i]);
        }
    }
    
    return result;
}

// === UTILITY FUNCTIONS ===

void tensor_sync_to_cpu(Tensor* tensor) {
    if (!tensor->on_cuda) return;
    
#ifdef CUDA_AVAILABLE
    extern void cuda_memcpy_to_host(float* dst, const float* src, size_t size);
    
    if (!tensor->data) {
        tensor->data = (float*)malloc(tensor->rows * tensor->cols * sizeof(float));
    }
    
    cuda_memcpy_to_host(tensor->data, tensor->cuda_data, 
                       tensor->rows * tensor->cols * sizeof(float));
#endif
}

void tensor_print(Tensor tensor, const char* name) {
    tensor_sync_to_cpu(&tensor);
    
    printf("\n=== Tensor: %s ===\n", name ? name : "Unknown");
    printf("Shape: %dx%d, Backend: %s\n", 
           tensor.rows, tensor.cols,
           tensor.on_cuda ? "CUDA" : "CPU");
    
    if (tensor.rows <= 10 && tensor.cols <= 10) {
        for (int i = 0; i < tensor.rows; i++) {
            printf("[ ");
            for (int j = 0; j < tensor.cols; j++) {
                printf("%8.4f ", tensor.data[i * tensor.cols + j]);
            }
            printf("]\n");
        }
    } else {
        printf("Tensor troppo grande per visualizzazione completa\n");
        printf("Primi elementi: ");
        for (int i = 0; i < fmin(5, tensor.rows * tensor.cols); i++) {
            printf("%.4f ", tensor.data[i]);
        }
        printf("...\n");
    }
    printf("========================\n\n");
}

void tensor_print_info(Tensor tensor) {
    printf("Tensor Info: %dx%d, %s, Memoria: %.2f KB\n",
           tensor.rows, tensor.cols,
           tensor.on_cuda ? "GPU" : "CPU",
           tensor_memory_usage(tensor) / 1024.0);
}

size_t tensor_memory_usage(Tensor tensor) {
    return tensor.rows * tensor.cols * sizeof(float);
}

size_t get_gpu_memory_usage(void) {
    if (!g_device_info.cuda_available) {
        return 0;
    }
    
#ifdef CUDA_AVAILABLE
    extern size_t cuda_get_memory_usage(void);
    return cuda_get_memory_usage();
#else
    return 0;
#endif
}

void tensor_set(Tensor tensor, int row, int col, float value) {
    if (row < 0 || row >= tensor.rows || col < 0 || col >= tensor.cols) {
        aicraft_log(LOG_ERROR, "[AiCraft] tensor_set: Indici fuori limite (%d,%d) per tensor (%d,%d)", 
                   row, col, tensor.rows, tensor.cols);
        return;
    }
    
    int index = row * tensor.cols + col;
    
    if (tensor.on_cuda) {
#ifdef CUDA_AVAILABLE
        extern void aicraft_cuda_set_value(float* data, int index, float value);
        aicraft_cuda_set_value(tensor.cuda_data, index, value);
#endif
    } else {
        tensor.data[index] = value;
    }
}

float tensor_get(Tensor tensor, int row, int col) {
    if (row < 0 || row >= tensor.rows || col < 0 || col >= tensor.cols) {
        aicraft_log(LOG_ERROR, "[AiCraft] tensor_get: Indici fuori limite (%d,%d) per tensor (%d,%d)",
                   row, col, tensor.rows, tensor.cols);
        return 0.0f;
    }
    
    int index = row * tensor.cols + col;
    
    if (tensor.on_cuda) {
        // Sync to CPU temporarily for single value access
        tensor_sync_to_cpu((Tensor*)&tensor);
    }
    
    return tensor.data[index];
}

void tensor_fill(Tensor* tensor, float value) {
    if (!tensor) return;
    
    if (tensor->on_cuda) {
#ifdef CUDA_AVAILABLE
        extern void aicraft_cuda_fill(float* data, float value, int size);
        aicraft_cuda_fill(tensor->cuda_data, value, tensor->rows * tensor->cols);
#endif
    } else {
        int size = tensor->rows * tensor->cols;
        for (int i = 0; i < size; i++) {
            tensor->data[i] = value;
        }
    }
}

int tensor_argmax(Tensor tensor) {
    tensor_sync_to_cpu(&tensor);
    
    int max_idx = 0;
    float max_val = tensor.data[0];
    int size = tensor.rows * tensor.cols;
    
    for (int i = 1; i < size; i++) {
        if (tensor.data[i] > max_val) {
            max_val = tensor.data[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

Tensor tensor_sum_axis(Tensor tensor, int axis) {
    Tensor result;
    
    if (axis == 0) {
        // Sum over rows (result is 1 x cols)
        result = tensor_create(1, tensor.cols, tensor.on_cuda ? BACKEND_CUDA : BACKEND_CPU);
    } else {
        // Sum over columns (result is rows x 1)
        result = tensor_create(tensor.rows, 1, tensor.on_cuda ? BACKEND_CUDA : BACKEND_CPU);
    }
    
    if (tensor.on_cuda && result.on_cuda) {
#ifdef CUDA_AVAILABLE
        extern void aicraft_cuda_sum_axis(float* output, const float* input, 
                                         int rows, int cols, int axis);
        aicraft_cuda_sum_axis(result.cuda_data, tensor.cuda_data, 
                             tensor.rows, tensor.cols, axis);
#endif
    } else {
        tensor_sync_to_cpu(&tensor);
        if (result.on_cuda) tensor_to_cpu(&result);
        
        if (axis == 0) {
            // Sum over rows
            for (int j = 0; j < tensor.cols; j++) {
                float sum = 0.0f;
                for (int i = 0; i < tensor.rows; i++) {
                    sum += tensor.data[i * tensor.cols + j];
                }
                result.data[j] = sum;
            }
        } else {
            // Sum over columns
            for (int i = 0; i < tensor.rows; i++) {
                float sum = 0.0f;
                for (int j = 0; j < tensor.cols; j++) {
                    sum += tensor.data[i * tensor.cols + j];
                }
                result.data[i] = sum;
            }
        }
        
        if (tensor.on_cuda) tensor_to_cuda(&result);
    }
    
    return result;
}

void tensor_elementwise_mul(Tensor* a, Tensor b) {
    if (!a || a->rows != b.rows || a->cols != b.cols) {
        aicraft_log(LOG_ERROR, "[AiCraft] tensor_elementwise_mul: Dimensioni incompatibili");
        return;
    }
    
    if (a->on_cuda && b.on_cuda) {
#ifdef CUDA_AVAILABLE
        extern void aicraft_cuda_elementwise_mul(float* a, const float* b, int size);
        aicraft_cuda_elementwise_mul(a->cuda_data, b.cuda_data, a->rows * a->cols);
#endif
    } else {
        tensor_sync_to_cpu(a);
        tensor_sync_to_cpu((Tensor*)&b);
        
        int size = a->rows * a->cols;
        for (int i = 0; i < size; i++) {
            a->data[i] *= b.data[i];
        }
        
        if (a->on_cuda) tensor_sync_to_cuda(a);
    }
}

void tensor_copy_row(Tensor* dst, int dst_row, Tensor* src, int src_row) {
    if (!dst || !src || dst_row >= dst->rows || src_row >= src->rows || dst->cols != src->cols) {
        aicraft_log(LOG_ERROR, "[AiCraft] tensor_copy_row: Parametri non validi");
        return;
    }
    
    if (dst->on_cuda && src->on_cuda) {
#ifdef CUDA_AVAILABLE
        extern void aicraft_cuda_memcpy_d2d(float* dst, const float* src, size_t size);
        aicraft_cuda_memcpy_d2d(&dst->cuda_data[dst_row * dst->cols],
                               &src->cuda_data[src_row * src->cols],
                               dst->cols * sizeof(float));
#endif
    } else {
        tensor_sync_to_cpu(dst);
        tensor_sync_to_cpu(src);
        
        memcpy(&dst->data[dst_row * dst->cols],
               &src->data[src_row * src->cols],
               dst->cols * sizeof(float));
        
        if (dst->on_cuda) tensor_sync_to_cuda(dst);
    }
}

Tensor tensor_zeros(int rows, int cols, BackendType backend) {
    Tensor tensor = tensor_create(rows, cols, backend);
    tensor_fill(&tensor, 0.0f);
    return tensor;
}

Tensor tensor_ones(int rows, int cols, BackendType backend) {
    Tensor tensor = tensor_create(rows, cols, backend);
    tensor_fill(&tensor, 1.0f);
    return tensor;
}

Tensor tensor_copy(Tensor tensor) {
    Tensor copy = tensor_create(tensor.rows, tensor.cols, 
                               tensor.on_cuda ? BACKEND_CUDA : BACKEND_CPU);
    
    if (tensor.on_cuda && copy.on_cuda) {
#ifdef CUDA_AVAILABLE
        extern void aicraft_cuda_memcpy_d2d(float* dst, const float* src, size_t size);
        aicraft_cuda_memcpy_d2d(copy.cuda_data, tensor.cuda_data,
                               tensor.rows * tensor.cols * sizeof(float));
#endif
    } else {
        tensor_sync_to_cpu(&tensor);
        if (copy.on_cuda) tensor_to_cpu(&copy);
        
        memcpy(copy.data, tensor.data, tensor.rows * tensor.cols * sizeof(float));
        
        if (tensor.on_cuda) tensor_to_cuda(&copy);
    }
    
    return copy;
}

// === GLOBAL FUNCTIONS ===

void aicraft_log(LogLevel level, const char* format, ...) {
    if (level < g_log_level) return;
    
    const char* level_names[] = {"DEBUG", "INFO", "WARNING", "ERROR"};
    const char* level_colors[] = {"\033[0;37m", "\033[0;36m", "\033[0;33m", "\033[0;31m"};
    
    printf("%s[%s]\033[0m ", level_colors[level], level_names[level]);
    
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    
    printf("\n");
    fflush(stdout);
}

void aicraft_set_log_level(LogLevel level) {
    g_log_level = level;
}

// === PERFORMANCE TIMING ===

void start_timer(void) {
    g_timer_start = clock();
}

double get_elapsed_time(void) {
    clock_t end = clock();
    return ((double)(end - g_timer_start)) / CLOCKS_PER_SEC * 1000.0; // milliseconds
}

const char* tensor_error_string(TensorError error) {
    switch (error) {
        case TENSOR_SUCCESS: return "Success";
        case TENSOR_ERROR_INVALID_SHAPE: return "Invalid tensor shape";
        case TENSOR_ERROR_MEMORY_ALLOCATION: return "Memory allocation failed";
        case TENSOR_ERROR_CUDA_ERROR: return "CUDA error";
        case TENSOR_ERROR_BACKEND_MISMATCH: return "Backend mismatch";
        default: return "Unknown error";
    }
}
