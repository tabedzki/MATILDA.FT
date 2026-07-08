// Copyright (c) 2026 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).
//
// Integration-level tests for FTS_Box built from tests/fixtures/fts2d.input
// (2D SCFT, 32x32 grid, two homopolymer species, Helfand + Flory potentials
// — a shrunken examples/ft/input rewritten in the current input syntax; the
// shipped example is stale: FTS_Molec now requires "phi/nmolecs/activity"
// before the value, and FTS_Box::readInput() die()s on its "maxSteps" line).
//
// Exercises the double-precision FFT wrapper (the FTS hot path), the
// k-space convolution helper, and the trapezoid field integrators.
// Construction is done once per suite (device allocations, propagator
// solves) and the box is shared read-mostly, same as test_ps_box.cu.
//
// Issue #1 notes: cufftWrapperDouble/convolveTComplexDouble take their
// inputs by value (a per-call device copy) — a performance finding only,
// results are still correct, so these tests exercise current signatures.

#include <gtest/gtest.h>
#include "FTS_Box.h"
#include "fts_species.h"   // FTS_Box.h only forward-declares FTS_Species

#include <cuda_runtime.h>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>

Box* BoxFactory(std::istringstream&);

namespace {

std::string fixturesDir() {
    const char* env = getenv("MATILDA_FIXTURES_DIR");
    char resolved[4096];
    if (realpath(env != nullptr ? env : "fixtures", resolved) == nullptr)
        return "";
    return resolved;
}

using TCpxVec = thrust::host_vector<thrust::complex<double>>;
using TCpxDevVec = thrust::device_vector<thrust::complex<double>>;

} // namespace

class FtsBoxTest : public ::testing::Test {
protected:
    static FTS_Box* box;
    static std::string origDir;
    static bool built;

    static void SetUpTestSuite() {
        std::string fixtures = fixturesDir();
        if (fixtures.empty()) return;

        // Construction writes fts_data.dat (and potential/density field
        // files) into the CWD; contain them in a private temp dir.
        char tmpl[] = "/tmp/matilda_ftsbox_XXXXXX";
        char* d = mkdtemp(tmpl);
        if (d == nullptr) return;
        char cwd[4096];
        if (getcwd(cwd, sizeof(cwd)) != nullptr) origDir = cwd;
        if (chdir(d) != 0) return;

        std::ifstream in2(fixtures + "/fts2d.input");
        if (!in2.is_open()) return;

        std::string line, word;
        Box* b = nullptr;
        while (std::getline(in2, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            iss >> word;
            if (word == "box") {
                b = BoxFactory(iss);
                b->readInput(in2);
            }
        }
        box = static_cast<FTS_Box*>(b);
        built = (box != nullptr);
    }

    static void TearDownTestSuite() {
        if (!origDir.empty()) { (void)!chdir(origDir.c_str()); }
        // Leaked deliberately; see the note in test_ps_box.cu.
        box = nullptr;
    }
};
FTS_Box* FtsBoxTest::box = nullptr;
std::string FtsBoxTest::origDir;
bool FtsBoxTest::built = false;

// The input commands must land in the right members, and the derived grid
// quantities must be mutually consistent (gvol * M == V).
TEST_F(FtsBoxTest, ConstructsFromFixture) {
    ASSERT_TRUE(built) << "FTS_Box failed to construct from fixture";

    EXPECT_EQ(box->returnBoxStyle(), "fts");
    EXPECT_EQ(box->returnFTSstyle(), "scft");
    EXPECT_EQ(box->returnDimension(), 2);

    EXPECT_EQ(box->Nx[0], 32);
    EXPECT_EQ(box->Nx[1], 32);
    EXPECT_EQ(box->M, 32 * 32);
    EXPECT_FLOAT_EQ(box->L[0], 16.0f);
    EXPECT_FLOAT_EQ(box->L[1], 16.0f);
    EXPECT_DOUBLE_EQ(box->V, 256.0);
    EXPECT_NEAR(box->dx[0], 0.5, 1e-12);
    EXPECT_NEAR(box->gvol, 0.25, 1e-12);
    EXPECT_NEAR(box->gvol * box->M, box->V, 1e-9);

    EXPECT_DOUBLE_EQ(box->rho0, 50.0);
    EXPECT_EQ(box->Species.size(), 2u);
    EXPECT_EQ(box->Molecs.size(), 2u);
    EXPECT_EQ(box->Potentials.size(), 2u);
}

// Forward (normalized by 1/M) then inverse (un-normalized) Z2Z transform
// must reproduce the original field to near machine precision.
TEST_F(FtsBoxTest, CufftDoubleForwardInverseRoundtrip) {
    ASSERT_TRUE(built);
    const int M = box->M;

    TCpxVec h(M);
    for (int i = 0; i < M; i++)
        h[i] = thrust::complex<double>(std::sin(0.3 * i) + 0.5,
                                       std::cos(0.11 * i) - 0.25);

    TCpxDevVec d_in = h, d_k(M), d_back(M);
    box->cufftWrapperDouble(d_in, d_k, 1);
    box->cufftWrapperDouble(d_k, d_back, -1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    TCpxVec out = d_back;
    for (int i = 0; i < M; i++) {
        EXPECT_NEAR(out[i].real(), h[i].real(), 1e-10) << "real at " << i;
        EXPECT_NEAR(out[i].imag(), h[i].imag(), 1e-10) << "imag at " << i;
    }
}

// The forward transform of a constant field is DC-only: bin 0 holds the
// constant (thanks to the 1/M normalization), every other bin is ~0.
TEST_F(FtsBoxTest, CufftDoubleForwardOfConstantIsDC) {
    ASSERT_TRUE(built);
    const int M = box->M;
    const thrust::complex<double> c(3.25, -1.5);

    TCpxVec h(M, c);
    TCpxDevVec d_in = h, d_k(M);
    box->cufftWrapperDouble(d_in, d_k, 1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    TCpxVec out = d_k;
    EXPECT_NEAR(out[0].real(), c.real(), 1e-10);
    EXPECT_NEAR(out[0].imag(), c.imag(), 1e-10);
    for (int i = 1; i < M; i++) {
        EXPECT_NEAR(out[i].real(), 0.0, 1e-10) << "DC leakage at bin " << i;
        EXPECT_NEAR(out[i].imag(), 0.0, 1e-10) << "DC leakage at bin " << i;
    }
}

// convolveTComplexDouble multiplies FFT_w(input) by the supplied k-space
// transfer function and inverse-transforms. A transfer function of all ones
// is the k-space representation of a unit delta, so the convolution must be
// the identity: dest == input.
TEST_F(FtsBoxTest, ConvolveWithUnitTransferFunctionIsIdentity) {
    ASSERT_TRUE(built);
    const int M = box->M;

    TCpxVec h(M);
    for (int i = 0; i < M; i++)
        h[i] = thrust::complex<double>(std::cos(0.07 * i), 0.02 * (i % 17));

    TCpxDevVec d_in = h;
    TCpxDevVec d_ones(M, thrust::complex<double>(1.0, 0.0));
    TCpxDevVec d_dest(M);

    box->convolveTComplexDouble(d_in, d_dest, d_ones);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    TCpxVec out = d_dest;
    for (int i = 0; i < M; i++) {
        EXPECT_NEAR(out[i].real(), h[i].real(), 1e-10) << "real at " << i;
        EXPECT_NEAR(out[i].imag(), h[i].imag(), 1e-10) << "imag at " << i;
    }
}

// Integrating a constant complex field over the grid must give
// constant * V for both the thrust and std::complex integrators
// (trapezoid rule == gvol * sum under PBC).
TEST_F(FtsBoxTest, IntegrateConstantFieldGivesVolume) {
    ASSERT_TRUE(built);
    const int M = box->M;
    const double cr = 1.75, ci = -0.4;

    TCpxVec h(M, thrust::complex<double>(cr, ci));
    thrust::complex<double> tsum = box->integTComplexD(h);
    EXPECT_NEAR(tsum.real(), cr * box->V, 1e-9 * std::abs(cr) * box->V);
    EXPECT_NEAR(tsum.imag(), ci * box->V, 1e-9 * std::abs(ci) * box->V);

    std::vector<std::complex<double>> hs(M, std::complex<double>(cr, ci));
    std::complex<double> ssum = box->integComplexD(hs.data());
    EXPECT_NEAR(ssum.real(), cr * box->V, 1e-9 * std::abs(cr) * box->V);
    EXPECT_NEAR(ssum.imag(), ci * box->V, 1e-9 * std::abs(ci) * box->V);
}
