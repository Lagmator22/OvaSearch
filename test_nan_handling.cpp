#include <iostream>
#include <vector>
#include <cmath>
#include "topk_nan_enhanced.hpp"

int main() {
    // Simulate corrupted embeddings scenario
    std::vector<float> similarities = {
        0.92f,                              // Good document
        std::nanf(""),                     // Corrupted embedding
        0.87f,                              // Good document
        std::nanf(""),                     // Another corruption
        0.75f,                              // Good document
        0.88f,                              // Good document
        0.91f,                              // Good document
        std::nanf("")                      // More corruption
    };

    std::cout << "Testing OvaSearch with NaN handling\n";
    std::cout << "===================================\n\n";

    // Test production mode
    std::cout << "Production Mode (exclude corrupted):\n";
    std::vector<int64_t> indices(5);
    std::vector<float> values(5);

    enhanced_topk::topk_with_nan_handling(
        similarities.data(), indices.data(), values.data(),
        similarities.size(), 5, true,
        enhanced_topk::NaNMode::NAN_AS_SMALLEST
    );

    for (int i = 0; i < 5; ++i) {
        std::cout << "  " << (i+1) << ". Document " << indices[i];
        if (std::isnan(values[i])) {
            std::cout << " (NaN - ERROR!)";
        } else {
            std::cout << " (score: " << values[i] << ")";
        }
        std::cout << "\n";
    }

    std::cout << "\nDebug Mode (see corrupted first):\n";
    enhanced_topk::topk_with_nan_handling(
        similarities.data(), indices.data(), values.data(),
        similarities.size(), 5, true,
        enhanced_topk::NaNMode::NAN_AS_LARGEST
    );

    for (int i = 0; i < 5; ++i) {
        std::cout << "  " << (i+1) << ". Document " << indices[i];
        if (std::isnan(values[i])) {
            std::cout << " (NaN - needs reindexing)";
        } else {
            std::cout << " (score: " << values[i] << ")";
        }
        std::cout << "\n";
    }

    return 0;
}
