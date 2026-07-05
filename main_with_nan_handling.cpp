// OvaSearch with TopK NaN handling
// Add this to your existing main.cpp

#include "topk_nan_enhanced.hpp"

// Modified search function with NaN protection
std::vector<size_t> search_with_nan_protection(
    const std::vector<float>& similarities,
    size_t k,
    bool debug_mode = false
) {
    std::vector<int64_t> indices(k);
    std::vector<float> values(k);

    // Use NAN_AS_SMALLEST for production (exclude corrupted)
    // Use NAN_AS_LARGEST for debug (see corrupted first)
    auto nan_mode = debug_mode ?
                    enhanced_topk::NaNMode::NAN_AS_LARGEST :
                    enhanced_topk::NaNMode::NAN_AS_SMALLEST;

    enhanced_topk::topk_with_nan_handling(
        similarities.data(),
        indices.data(),
        values.data(),
        similarities.size(),
        k,
        true,  // MAX mode for similarity
        nan_mode
    );

    // Check for NaN in results and log warnings
    for (size_t i = 0; i < k; ++i) {
        if (std::isnan(values[i])) {
            std::cerr << "WARNING: NaN detected in retrieval at position "
                     << i << " (document index: " << indices[i] << ")\n";
        }
    }

    return std::vector<size_t>(indices.begin(), indices.end());
}

// Add this to your existing retrieval code:
// Replace: auto top_indices = find_top_k(similarities, 5);
// With:    auto top_indices = search_with_nan_protection(similarities, 5);
