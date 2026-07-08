// Copyright (c) 2026 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).
//
// Test-suite entry point. Defines the globals that src/main.cu normally
// provides via the "#define MAIN" pattern (random.h, timing.h), then runs
// all GoogleTest cases.

#define MAIN
#include "random.h"
#include "timing.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <string>

// Normally defined in src/main.cu; referenced by die() and PS_Box::NVT().
std::string giveQuote() {
    return "matilda.ft test suite";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Deterministic default seed for the host RNG; individual tests may
    // re-seed by assigning a negative value to idum before calling ran2().
    idum = -911;

    int ret = RUN_ALL_TESTS();

    cudaDeviceReset();
    return ret;
}
