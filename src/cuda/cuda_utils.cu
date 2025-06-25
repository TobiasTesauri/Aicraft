// CUDA utilities and memory management for AiCraft

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

// Memory pool for efficient GPU memory management
#define MAX_MEMORY_BLOCKS 1024
#define MEMORY_POOL_SIZE (512 * 1024 * 1024)  // 512MB limit

typedef struct {
    void* ptr;
    size_t size;
    bool is_free;
    int alignment;
} MemoryBlock;

typedef struct {
    void* pool_memory;
    MemoryBlock blocks[MAX_MEMORY_BLOCKS];
    int num_blocks;
    size_t total_allocated;
    size_t total_size;
    cublasHandle_t cublas_handle;
} GPUMemoryPool;

static GPUMemoryPool g_memory_pool = {0};
static bool g_cuda_initialized = false;

// === CUDA INITIALIZATION ===

extern "C" {

bool cuda_init_device(void* device_info_ptr) {
    typedef struct {
        bool cuda_available;
        int cuda_device_count;
        int current_device;
        size_t total_memory;
        size_t free_memory;
        char device_name[256];
        int compute_capability_major;
        int compute_capability_minor;
    } DeviceInfo;
    
    DeviceInfo* info = (DeviceInfo*)device_info_ptr;
    
    // Check if CUDA is available
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess || device_count == 0) {
        printf("[AiCraft] CUDA non disponibile: %s\n", cudaGetErrorString(error));
        return false;
    }
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // Fill device info
    info->cuda_available = true;
    info->cuda_device_count = device_count;
    info->current_device = 0;
    strncpy(info->device_name, prop.name, sizeof(info->device_name) - 1);
    info->compute_capability_major = prop.major;
    info->compute_capability_minor = prop.minor;
    
    // Get memory info
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    info->free_memory = free_mem;
    info->total_memory = total_mem;
    
    // Initialize memory pool
    if (cuda_init_memory_pool()) {
        printf("[AiCraft] Pool di memoria GPU inizializzato: %.1f MB\n", 
               MEMORY_POOL_SIZE / 1024.0 / 1024.0);
    } else {
        printf("[AiCraft] Errore inizializzazione pool memoria GPU\n");
        return false;
    }
    
    // Initialize cuBLAS
    cublasStatus_t stat = cublasCreate(&g_memory_pool.cublas_handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("[AiCraft] Errore inizializzazione cuBLAS\n");
        return false;
    }
    
    // Set math mode for Tensor Cores
    cublasSetMathMode(g_memory_pool.cublas_handle, CUBLAS_TENSOR_OP_MATH);
    
    g_cuda_initialized = true;
    printf("[AiCraft] CUDA inizializzato con successo\n");
    printf("[AiCraft] GPU: %s (Compute %d.%d)\n", 
           prop.name, prop.major, prop.minor);
    printf("[AiCraft] Memoria: %.1f MB totale, %.1f MB libera\n",
           total_mem / 1024.0 / 1024.0, free_mem / 1024.0 / 1024.0);
    
    if (prop.major >= 7) {
        printf("[AiCraft] Tensor Cores disponibili - Performance ottimale\n");
    }
    
    return true;
}

bool cuda_init_memory_pool() {
    if (g_memory_pool.pool_memory != nullptr) {
        return true; // Already initialized
    }
    
    // Allocate main memory pool
    cudaError_t error = cudaMalloc(&g_memory_pool.pool_memory, MEMORY_POOL_SIZE);
    if (error != cudaSuccess) {
        printf("[AiCraft] Errore allocazione pool memoria: %s\n", cudaGetErrorString(error));
        return false;
    }
    
    // Initialize first block as entirely free
    g_memory_pool.blocks[0].ptr = g_memory_pool.pool_memory;
    g_memory_pool.blocks[0].size = MEMORY_POOL_SIZE;
    g_memory_pool.blocks[0].is_free = true;
    g_memory_pool.blocks[0].alignment = 256;
    
    g_memory_pool.num_blocks = 1;
    g_memory_pool.total_allocated = 0;
    g_memory_pool.total_size = MEMORY_POOL_SIZE;
    
    return true;
}

void cuda_cleanup() {
    if (!g_cuda_initialized) return;
    
    printf("[AiCraft] Pulizia risorse CUDA...\n");
    
    // Destroy cuBLAS handle
    if (g_memory_pool.cublas_handle) {
        cublasDestroy(g_memory_pool.cublas_handle);
    }
    
    // Free memory pool
    if (g_memory_pool.pool_memory) {
        cudaFree(g_memory_pool.pool_memory);
        g_memory_pool.pool_memory = nullptr;
    }
    
    // Reset CUDA device
    cudaDeviceReset();
    
    g_cuda_initialized = false;
}

// === MEMORY MANAGEMENT ===

float* cuda_malloc(size_t size) {
    if (!g_cuda_initialized) {
        printf("[AiCraft] CUDA non inizializzato\n");
        return nullptr;
    }
    
    // Align to 256 bytes for optimal memory access
    size = (size + 255) & ~255;
    
    // Find free block
    for (int i = 0; i < g_memory_pool.num_blocks; ++i) {
        MemoryBlock* block = &g_memory_pool.blocks[i];
        
        if (block->is_free && block->size >= size) {
            block->is_free = false;
            
            // Split block if necessary
            if (block->size > size + 256 && g_memory_pool.num_blocks < MAX_MEMORY_BLOCKS - 1) {
                MemoryBlock* new_block = &g_memory_pool.blocks[g_memory_pool.num_blocks];
                new_block->ptr = (char*)block->ptr + size;
                new_block->size = block->size - size;
                new_block->is_free = true;
                new_block->alignment = 256;
                
                block->size = size;
                g_memory_pool.num_blocks++;
            }
            
            g_memory_pool.total_allocated += size;
            return (float*)block->ptr;
        }
    }
    
    printf("[AiCraft] Pool memoria GPU esaurito (richiesti %zu bytes)\n", size);
    return nullptr;
}

void cuda_free(float* ptr) {
    if (!ptr || !g_cuda_initialized) return;
    
    // Find block and mark as free
    for (int i = 0; i < g_memory_pool.num_blocks; ++i) {
        if (g_memory_pool.blocks[i].ptr == ptr) {
            g_memory_pool.blocks[i].is_free = true;
            g_memory_pool.total_allocated -= g_memory_pool.blocks[i].size;
            
            // Coalesce adjacent free blocks
            cuda_coalesce_memory();
            return;
        }
    }
    
    printf("[AiCraft] Tentativo di liberare puntatore non valido\n");
}

void cuda_coalesce_memory() {
    for (int i = 0; i < g_memory_pool.num_blocks - 1; ++i) {
        MemoryBlock* current = &g_memory_pool.blocks[i];
        MemoryBlock* next = &g_memory_pool.blocks[i + 1];
        
        if (current->is_free && next->is_free &&
            (char*)current->ptr + current->size == next->ptr) {
            
            // Merge blocks
            current->size += next->size;
            
            // Remove next block
            for (int j = i + 1; j < g_memory_pool.num_blocks - 1; ++j) {
                g_memory_pool.blocks[j] = g_memory_pool.blocks[j + 1];
            }
            g_memory_pool.num_blocks--;
            i--; // Re-check this position
        }
    }
}

size_t cuda_get_memory_usage() {
    return g_memory_pool.total_allocated;
}

size_t cuda_get_memory_available() {
    return g_memory_pool.total_size - g_memory_pool.total_allocated;
}

void cuda_print_memory_stats() {
    printf("[AiCraft] === Statistiche Memoria GPU ===\n");
    printf("[AiCraft] Allocata: %.1f MB / %.1f MB (%.1f%%)\n",
           g_memory_pool.total_allocated / 1024.0 / 1024.0,
           g_memory_pool.total_size / 1024.0 / 1024.0,
           (g_memory_pool.total_allocated * 100.0) / g_memory_pool.total_size);
    printf("[AiCraft] Blocchi: %d, Liberi: %d\n", 
           g_memory_pool.num_blocks, cuda_count_free_blocks());
    printf("[AiCraft] ===================================\n");
}

int cuda_count_free_blocks() {
    int free_count = 0;
    for (int i = 0; i < g_memory_pool.num_blocks; ++i) {
        if (g_memory_pool.blocks[i].is_free) {
            free_count++;
        }
    }
    return free_count;
}

// === MEMORY COPY OPERATIONS ===

void cuda_memcpy_to_device(float* dst, const float* src, size_t size) {
    cudaError_t error = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("[AiCraft] Errore copia memoria H->D: %s\n", cudaGetErrorString(error));
    }
}

void cuda_memcpy_to_host(float* dst, const float* src, size_t size) {
    cudaError_t error = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        printf("[AiCraft] Errore copia memoria D->H: %s\n", cudaGetErrorString(error));
    }
}

void cuda_memcpy_device_to_device(float* dst, const float* src, size_t size) {
    cudaError_t error = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    if (error != cudaSuccess) {
        printf("[AiCraft] Errore copia memoria D->D: %s\n", cudaGetErrorString(error));
    }
}

// === OPTIMIZED BLAS OPERATIONS ===

void cuda_gemm_blas(const float* A, const float* B, float* C, 
                    int M, int N, int K, float alpha, float beta) {
    if (!g_cuda_initialized || !g_memory_pool.cublas_handle) {
        printf("[AiCraft] cuBLAS non inizializzato\n");
        return;
    }
    
    cublasStatus_t stat = cublasSgemm(g_memory_pool.cublas_handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, M, K,
                                     &alpha,
                                     B, N,
                                     A, K,
                                     &beta,
                                     C, N);
    
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("[AiCraft] Errore cuBLAS SGEMM: %d\n", stat);
    }
}

// === UTILITY FUNCTIONS ===

bool cuda_is_available() {
    return g_cuda_initialized;
}

void cuda_synchronize() {
    if (g_cuda_initialized) {
        cudaDeviceSynchronize();
    }
}

void cuda_check_error(const char* operation) {
    if (!g_cuda_initialized) return;
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("[AiCraft] Errore CUDA in %s: %s\n", operation, cudaGetErrorString(error));
    }
}

void cuda_set_device(int device_id) {
    if (g_cuda_initialized) {
        cudaSetDevice(device_id);
    }
}

int cuda_get_device() {
    if (!g_cuda_initialized) return -1;
    
    int device;
    cudaGetDevice(&device);
    return device;
}

void cuda_device_info() {
    if (!g_cuda_initialized) {
        printf("[AiCraft] CUDA non inizializzato\n");
        return;
    }
    
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    printf("[AiCraft] === Informazioni GPU ===\n");
    printf("[AiCraft] Device: %d - %s\n", device, prop.name);
    printf("[AiCraft] Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("[AiCraft] Memory Clock: %d MHz\n", prop.memoryClockRate / 1000);
    printf("[AiCraft] Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("[AiCraft] L2 Cache: %d KB\n", prop.l2CacheSize / 1024);
    printf("[AiCraft] Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("[AiCraft] Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("[AiCraft] Memoria Globale: %.1f GB\n", total_mem / 1024.0 / 1024.0 / 1024.0);
    printf("[AiCraft] Memoria Libera: %.1f GB\n", free_mem / 1024.0 / 1024.0 / 1024.0);
    
    if (prop.major >= 7) {
        printf("[AiCraft] Tensor Cores: Supportati\n");
    } else {
        printf("[AiCraft] Tensor Cores: Non supportati\n");
    }
    
    printf("[AiCraft] ==========================\n");
}

// === PERFORMANCE MONITORING ===

void cuda_start_profiling() {
    if (g_cuda_initialized) {
        cudaProfilerStart();
    }
}

void cuda_stop_profiling() {
    if (g_cuda_initialized) {
        cudaProfilerStop();
    }
}

float cuda_get_memory_bandwidth() {
    if (!g_cuda_initialized) return 0.0f;
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // Calculate theoretical memory bandwidth
    float memory_clock_ghz = prop.memoryClockRate / 1000000.0f;
    float bus_width_bytes = prop.memoryBusWidth / 8.0f;
    float bandwidth_gb_s = memory_clock_ghz * bus_width_bytes * 2.0f; // DDR
    
    return bandwidth_gb_s;
}

} // extern "C"
