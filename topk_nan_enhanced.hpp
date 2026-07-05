// Enhanced TopK with NaN handling modes - Proof of Concept
// This is a standalone implementation to demonstrate the concept
#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <tuple>

namespace enhanced_topk {

// NaN handling modes as suggested by mitruska
enum class NaNMode {
    NONE,           // Keep backward compatibility (default) - current behavior
    NAN_AS_SMALLEST,  // Treat NaN as smallest value
    NAN_AS_LARGEST    // Treat NaN as largest value
};

// Enhanced comparison function with NaN handling
template <bool IsMaxMode, typename T, typename U>
class NaNAwareComparator {
private:
    NaNMode nan_mode;

public:
    explicit NaNAwareComparator(NaNMode mode) : nan_mode(mode) {}

    bool operator()(const std::tuple<T, U>& a, const std::tuple<T, U>& b) const {
        T val_a = std::get<0>(a);
        T val_b = std::get<0>(b);
        U idx_a = std::get<1>(a);
        U idx_b = std::get<1>(b);

        bool a_is_nan = std::isnan(val_a);
        bool b_is_nan = std::isnan(val_b);

        // Handle NaN comparison based on mode
        if (a_is_nan || b_is_nan) {
            if (nan_mode == NaNMode::NONE) {
                // Current behavior - let standard comparison handle it (undefined)
                if (val_a != val_b) {
                    return IsMaxMode ? val_a > val_b : val_a < val_b;
                } else {
                    return idx_a < idx_b;
                }
            } else if (nan_mode == NaNMode::NAN_AS_SMALLEST) {
                // NaN is treated as smallest value
                if (a_is_nan && b_is_nan) {
                    return idx_a < idx_b;  // Both NaN, use index for stability
                } else if (a_is_nan) {
                    return !IsMaxMode;  // For MAX: NaN < any, for MIN: NaN is first
                } else {  // b_is_nan
                    return IsMaxMode;   // For MAX: any > NaN, for MIN: any is not first
                }
            } else {  // NaNMode::NAN_AS_LARGEST
                // NaN is treated as largest value
                if (a_is_nan && b_is_nan) {
                    return idx_a < idx_b;  // Both NaN, use index for stability
                } else if (a_is_nan) {
                    return IsMaxMode;   // For MAX: NaN > any, for MIN: NaN is last
                } else {  // b_is_nan
                    return !IsMaxMode;  // For MAX: any < NaN, for MIN: any is first
                }
            }
        }

        // Neither is NaN - normal comparison
        if (val_a != val_b) {
            return IsMaxMode ? val_a > val_b : val_a < val_b;
        } else {
            return idx_a < idx_b;  // Equal values, use index for stable sort
        }
    }
};

// Enhanced TopK implementation
template <typename T>
void topk_with_nan_handling(
    const T* input,
    int64_t* out_indices,
    T* out_values,
    size_t input_size,
    size_t k,
    bool compute_max,
    NaNMode nan_mode,
    bool stable_sort = false
) {
    // Create workspace with value-index pairs
    std::vector<std::tuple<T, int64_t>> workspace;
    workspace.reserve(input_size);

    for (size_t i = 0; i < input_size; ++i) {
        workspace.emplace_back(input[i], static_cast<int64_t>(i));
    }

    // Select appropriate comparator
    if (compute_max) {
        NaNAwareComparator<true, T, int64_t> cmp(nan_mode);

        if (stable_sort) {
            // Use stable_sort for entire array then take first k
            std::stable_sort(workspace.begin(), workspace.end(), cmp);
        } else {
            // Use nth_element for efficiency
            std::nth_element(workspace.begin(), workspace.begin() + k, workspace.end(), cmp);
            std::sort(workspace.begin(), workspace.begin() + k, cmp);
        }
    } else {
        NaNAwareComparator<false, T, int64_t> cmp(nan_mode);

        if (stable_sort) {
            std::stable_sort(workspace.begin(), workspace.end(), cmp);
        } else {
            std::nth_element(workspace.begin(), workspace.begin() + k, workspace.end(), cmp);
            std::sort(workspace.begin(), workspace.begin() + k, cmp);
        }
    }

    // Extract results
    for (size_t j = 0; j < k; ++j) {
        out_values[j] = std::get<0>(workspace[j]);
        out_indices[j] = std::get<1>(workspace[j]);
    }
}

} // namespace enhanced_topk