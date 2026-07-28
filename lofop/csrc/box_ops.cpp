// Native box operations for LOFOP detection post-processing.
//
// A multi-scale detector emits thousands of candidate boxes per image, from
// tiny objects on high-resolution pyramid levels to large ones on coarse
// levels; pairwise IoU and greedy NMS over those candidates dominate CPU
// post-processing time. These kernels exist so that cost stays flat as the
// candidate count grows.
//
// Exposed through a plain C ABI (loaded via ctypes) so the library builds
// with any C++17 compiler and needs no Python headers, no pybind11, and no
// framework runtime. Boxes are float32 xyxy, row-major.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace {

inline float box_area(const float* box) {
    const float w = box[2] - box[0];
    const float h = box[3] - box[1];
    return (w > 0.0f && h > 0.0f) ? w * h : 0.0f;
}

inline float pair_iou(const float* a, const float* b, float area_a, float area_b) {
    const float ix1 = std::max(a[0], b[0]);
    const float iy1 = std::max(a[1], b[1]);
    const float ix2 = std::min(a[2], b[2]);
    const float iy2 = std::min(a[3], b[3]);
    const float iw = ix2 - ix1;
    const float ih = iy2 - iy1;
    if (iw <= 0.0f || ih <= 0.0f) {
        return 0.0f;
    }
    const float inter = iw * ih;
    const float uni = area_a + area_b - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

}  // namespace

extern "C" {

// Fill `out` (n*m, row-major) with IoU of every box in `a` against every box
// in `b`.
void lofop_iou_matrix(const float* a, int32_t n, const float* b, int32_t m, float* out) {
    std::vector<float> areas_b(static_cast<size_t>(m));
    for (int32_t j = 0; j < m; ++j) {
        areas_b[static_cast<size_t>(j)] = box_area(b + 4 * j);
    }
    for (int32_t i = 0; i < n; ++i) {
        const float* box_a = a + 4 * i;
        const float area_a = box_area(box_a);
        float* row = out + static_cast<int64_t>(i) * m;
        for (int32_t j = 0; j < m; ++j) {
            row[j] = pair_iou(box_a, b + 4 * j, area_a, areas_b[static_cast<size_t>(j)]);
        }
    }
}

// Greedy NMS. Writes kept indices (score-descending) into `keep_out`, which
// must hold at least n int32 values; returns how many were kept.
// `max_keep` <= 0 means no limit.
int32_t lofop_nms(const float* boxes, const float* scores, int32_t n, float iou_threshold,
                  int32_t max_keep, int32_t* keep_out) {
    std::vector<int32_t> order(static_cast<size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [scores](int32_t l, int32_t r) { return scores[l] > scores[r]; });

    std::vector<float> areas(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) {
        areas[static_cast<size_t>(i)] = box_area(boxes + 4 * i);
    }

    std::vector<char> suppressed(static_cast<size_t>(n), 0);
    int32_t kept = 0;
    for (size_t oi = 0; oi < order.size(); ++oi) {
        const int32_t i = order[oi];
        if (suppressed[static_cast<size_t>(i)]) {
            continue;
        }
        keep_out[kept++] = i;
        if (max_keep > 0 && kept >= max_keep) {
            break;
        }
        const float* box_i = boxes + 4 * i;
        const float area_i = areas[static_cast<size_t>(i)];
        for (size_t oj = oi + 1; oj < order.size(); ++oj) {
            const int32_t j = order[oj];
            if (suppressed[static_cast<size_t>(j)]) {
                continue;
            }
            if (pair_iou(box_i, boxes + 4 * j, area_i, areas[static_cast<size_t>(j)]) >
                iou_threshold) {
                suppressed[static_cast<size_t>(j)] = 1;
            }
        }
    }
    return kept;
}

// Soft-NMS. Instead of hard-dropping boxes that overlap a kept box, decay
// their scores -- linear (method 0): s *= 1 - iou when iou > iou_threshold;
// gaussian (method 1): s *= exp(-iou^2 / sigma). Boxes whose decayed score
// falls below `score_threshold` are discarded. Writes kept indices into
// `keep_out` and their final scores into `scores_out` (both sized >= n, kept
// in selection order, score-descending); returns how many were kept.
int32_t lofop_soft_nms(const float* boxes, const float* scores, int32_t n, float iou_threshold,
                       float sigma, float score_threshold, int32_t method, int32_t max_keep,
                       int32_t* keep_out, float* scores_out) {
    std::vector<float> live_scores(scores, scores + n);
    std::vector<float> areas(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) {
        areas[static_cast<size_t>(i)] = box_area(boxes + 4 * i);
    }
    std::vector<char> done(static_cast<size_t>(n), 0);
    int32_t kept = 0;
    for (;;) {
        int32_t best = -1;
        float best_score = score_threshold;
        for (int32_t i = 0; i < n; ++i) {
            if (!done[static_cast<size_t>(i)] && live_scores[static_cast<size_t>(i)] > best_score) {
                best = i;
                best_score = live_scores[static_cast<size_t>(i)];
            }
        }
        if (best < 0) {
            break;
        }
        done[static_cast<size_t>(best)] = 1;
        keep_out[kept] = best;
        scores_out[kept] = best_score;
        ++kept;
        if (max_keep > 0 && kept >= max_keep) {
            break;
        }
        const float* box_b = boxes + 4 * best;
        const float area_b = areas[static_cast<size_t>(best)];
        for (int32_t j = 0; j < n; ++j) {
            if (done[static_cast<size_t>(j)]) {
                continue;
            }
            const float iou =
                pair_iou(box_b, boxes + 4 * j, area_b, areas[static_cast<size_t>(j)]);
            if (iou <= 0.0f) {
                continue;
            }
            float weight = 1.0f;
            if (method == 1) {
                weight = std::exp(-(iou * iou) / sigma);
            } else if (iou > iou_threshold) {
                weight = 1.0f - iou;
            }
            live_scores[static_cast<size_t>(j)] *= weight;
        }
    }
    return kept;
}

// Dense decode: per candidate row of an (n, num_classes) score matrix, find
// the best class; emit the candidate when that score clears `threshold`.
// Writes candidate row indices, class ids, and scores into the out arrays
// (sized >= n); returns how many candidates were emitted. This replaces the
// O(n * c) Python loop in torch-free deployment post-processing.
int32_t lofop_decode_dense(const float* scores, int32_t n, int32_t num_classes, float threshold,
                           int32_t* index_out, int32_t* label_out, float* score_out) {
    int32_t count = 0;
    for (int32_t i = 0; i < n; ++i) {
        const float* row = scores + static_cast<int64_t>(i) * num_classes;
        int32_t best = 0;
        float best_score = row[0];
        for (int32_t c = 1; c < num_classes; ++c) {
            if (row[c] > best_score) {
                best = c;
                best_score = row[c];
            }
        }
        if (best_score > threshold) {
            index_out[count] = i;
            label_out[count] = best;
            score_out[count] = best_score;
            ++count;
        }
    }
    return count;
}

}  // extern "C"
