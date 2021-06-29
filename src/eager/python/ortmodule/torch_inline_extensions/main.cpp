#include <torch/extension.h>

        #include <torch/extension.h>
        #include <c10/cuda/CUDACachingAllocator.h>

        size_t gpu_caching_allocator_raw_alloc_address() {
            return reinterpret_cast<size_t>(&c10::cuda::CUDACachingAllocator::raw_alloc);
        }

        size_t gpu_caching_allocator_raw_delete_address() {
            return reinterpret_cast<size_t>(&c10::cuda::CUDACachingAllocator::raw_delete);
        }
    
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("gpu_caching_allocator_raw_alloc_address", torch::wrap_pybind_function(gpu_caching_allocator_raw_alloc_address), "gpu_caching_allocator_raw_alloc_address");
m.def("gpu_caching_allocator_raw_delete_address", torch::wrap_pybind_function(gpu_caching_allocator_raw_delete_address), "gpu_caching_allocator_raw_delete_address");
}