// CUDA box operations for LOFOP: the optional third ops tier.
//
// Backend order is cuda > cpp > python. These kernels accelerate the
// embarrassingly parallel ops (pairwise IoU, dense decode) on NVIDIA GPUs for
// the torch-free deployment path; greedy/Soft-NMS stays on the C++ tier
// because its suppression loop is inherently sequential.
//
// Same contract as box_ops.cpp: plain C ABI over host pointers (device
// transfers are handled inside), float32 xyxy boxes, identical results to the
// Python reference. Built only when nvcc is available -- see
// lofop.ops.native.build_native(cuda=True); everything falls back cleanly
// when it is not.

#include <cstdint>
#include <cuda_runtime.h>

namespace {

__device__ inline float box_area(const float* box) {
    const float w = box[2] - box[0];
    const float h = box[3] - box[1];
    return (w > 0.0f && h > 0.0f) ? w * h : 0.0f;
}

__global__ void iou_matrix_kernel(const float* a, int32_t n, const float* b, int32_t m,
                                  float* out) {
    const int32_t i = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= m) {
        return;
    }
    const float* box_a = a + 4 * i;
    const float* box_b = b + 4 * j;
    const float ix1 = fmaxf(box_a[0], box_b[0]);
    const float iy1 = fmaxf(box_a[1], box_b[1]);
    const float ix2 = fminf(box_a[2], box_b[2]);
    const float iy2 = fminf(box_a[3], box_b[3]);
    const float iw = ix2 - ix1;
    const float ih = iy2 - iy1;
    float iou = 0.0f;
    if (iw > 0.0f && ih > 0.0f) {
        const float inter = iw * ih;
        const float uni = box_area(box_a) + box_area(box_b) - inter;
        iou = uni > 0.0f ? inter / uni : 0.0f;
    }
    out[static_cast<int64_t>(i) * m + j] = iou;
}

__global__ void decode_dense_kernel(const float* scores, int32_t n, int32_t num_classes,
                                    float threshold, int32_t* label_out, float* score_out) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float* row = scores + static_cast<int64_t>(i) * num_classes;
    int32_t best = 0;
    float best_score = row[0];
    for (int32_t c = 1; c < num_classes; ++c) {
        if (row[c] > best_score) {
            best = c;
            best_score = row[c];
        }
    }
    // A negative label marks a below-threshold row; the host side compacts.
    label_out[i] = best_score > threshold ? best : -1;
    score_out[i] = best_score;
}

bool alloc_copy(void** device, const void* host, size_t bytes) {
    if (cudaMalloc(device, bytes) != cudaSuccess) {
        return false;
    }
    if (host != nullptr && cudaMemcpy(*device, host, bytes, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        cudaFree(*device);
        *device = nullptr;
        return false;
    }
    return true;
}

}  // namespace

extern "C" {

// Returns 1 when at least one CUDA device is usable, else 0. Called once at
// load time by the Python side to decide whether this tier is active.
int32_t lofop_cuda_available() {
    int count = 0;
    return (cudaGetDeviceCount(&count) == cudaSuccess && count > 0) ? 1 : 0;
}

// GPU pairwise IoU over host arrays. Returns 0 on success, nonzero on any
// CUDA failure (caller falls back to the C++/Python tiers).
int32_t lofop_cuda_iou_matrix(const float* a, int32_t n, const float* b, int32_t m, float* out) {
    float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
    const size_t bytes_a = static_cast<size_t>(n) * 4 * sizeof(float);
    const size_t bytes_b = static_cast<size_t>(m) * 4 * sizeof(float);
    const size_t bytes_out = static_cast<size_t>(n) * m * sizeof(float);
    int32_t status = 1;
    if (alloc_copy(reinterpret_cast<void**>(&d_a), a, bytes_a) &&
        alloc_copy(reinterpret_cast<void**>(&d_b), b, bytes_b) &&
        alloc_copy(reinterpret_cast<void**>(&d_out), nullptr, bytes_out)) {
        const dim3 block(16, 16);
        const dim3 grid((m + 15) / 16, (n + 15) / 16);
        iou_matrix_kernel<<<grid, block>>>(d_a, n, d_b, m, d_out);
        if (cudaGetLastError() == cudaSuccess &&
            cudaMemcpy(out, d_out, bytes_out, cudaMemcpyDeviceToHost) == cudaSuccess) {
            status = 0;
        }
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    return status;
}

// GPU dense decode over a host (n, num_classes) score matrix. Emits the same
// compacted candidate arrays as the C++ lofop_decode_dense; returns the
// candidate count, or -1 on any CUDA failure (caller falls back).
int32_t lofop_cuda_decode_dense(const float* scores, int32_t n, int32_t num_classes,
                                float threshold, int32_t* index_out, int32_t* label_out,
                                float* score_out) {
    float *d_scores = nullptr, *d_best = nullptr;
    int32_t* d_labels = nullptr;
    const size_t bytes_scores = static_cast<size_t>(n) * num_classes * sizeof(float);
    const size_t bytes_row = static_cast<size_t>(n) * sizeof(float);
    int32_t count = -1;
    if (alloc_copy(reinterpret_cast<void**>(&d_scores), scores, bytes_scores) &&
        alloc_copy(reinterpret_cast<void**>(&d_labels), nullptr,
                   static_cast<size_t>(n) * sizeof(int32_t)) &&
        alloc_copy(reinterpret_cast<void**>(&d_best), nullptr, bytes_row)) {
        const int32_t block = 256;
        const int32_t grid = (n + block - 1) / block;
        decode_dense_kernel<<<grid, block>>>(d_scores, n, num_classes, threshold, d_labels,
                                             d_best);
        // Host-side compaction keeps candidate order deterministic (row
        // ascending), matching the C++ tier exactly.
        float* best = new float[static_cast<size_t>(n)];
        int32_t* labels = new int32_t[static_cast<size_t>(n)];
        if (cudaGetLastError() == cudaSuccess &&
            cudaMemcpy(labels, d_labels, static_cast<size_t>(n) * sizeof(int32_t),
                       cudaMemcpyDeviceToHost) == cudaSuccess &&
            cudaMemcpy(best, d_best, bytes_row, cudaMemcpyDeviceToHost) == cudaSuccess) {
            count = 0;
            for (int32_t i = 0; i < n; ++i) {
                if (labels[i] >= 0) {
                    index_out[count] = i;
                    label_out[count] = labels[i];
                    score_out[count] = best[i];
                    ++count;
                }
            }
        }
        delete[] best;
        delete[] labels;
    }
    cudaFree(d_scores);
    cudaFree(d_labels);
    cudaFree(d_best);
    return count;
}

}  // extern "C"
