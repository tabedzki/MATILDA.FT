// Copyright (c) 2026 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).
//
// Unit tests for the host RNG (src/random.cu, Numerical Recipes ran2).

#include <gtest/gtest.h>

double ran2(void);
extern long idum;

// Re-seeding with the same negative idum must reproduce the same sequence.
TEST(Ran2, DeterministicForFixedSeed) {
    idum = -42;
    double first[8];
    for (double& v : first) v = ran2();

    idum = -42;
    for (double v : first) EXPECT_DOUBLE_EQ(v, ran2());
}

// Different seeds must give different sequences.
TEST(Ran2, SeedChangesSequence) {
    idum = -1;
    double a = ran2();
    idum = -2;
    double b = ran2();
    EXPECT_NE(a, b);
}

// Values stay in [0,1) and the sample mean/variance match a uniform
// distribution to within loose statistical bounds.
TEST(Ran2, UniformOnUnitInterval) {
    idum = -1234;
    const int N = 100000;
    double sum = 0.0, sumSq = 0.0;
    for (int i = 0; i < N; i++) {
        double r = ran2();
        ASSERT_GE(r, 0.0);
        ASSERT_LT(r, 1.0);
        sum += r;
        sumSq += r * r;
    }
    double mean = sum / N;
    double var = sumSq / N - mean * mean;
    EXPECT_NEAR(mean, 0.5, 0.01);
    EXPECT_NEAR(var, 1.0 / 12.0, 0.01);
}
