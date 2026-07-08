// Copyright (c) 2026 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).
//
// Integration-level tests for PS_Box built from a small 2D fixture
// (tests/fixtures/ps2d.input + ps2d.data): 200 particles, two types, a VV
// integrator, an "all" neighbor list (rcut=1.5) and a Gaussian A-B potential.
//
// The box is expensive to construct (file I/O + device allocation), so it is
// built ONCE per suite in SetUpTestSuite and shared read-mostly across tests.
// The suite chdir's into a temp dir first because finishInitialization()
// writes several output files (init data, traj.gsd, density-*.bin, ...) into
// the current working directory.

#include <gtest/gtest.h>
#include "PS_Box.h"

#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <cmath>
#include <unistd.h>

Box* BoxFactory(std::istringstream&);

// Element-wise complex multiply from src/device_utils.cu, used to exercise
// convolution through the single-precision FFT wrapper.
__global__ void d_multiplyCpxByCpx(cuComplex*, const cuComplex*, const cuComplex*, const int);

namespace {

// Locate the fixtures directory without assuming where the repo is checked
// out: $MATILDA_FIXTURES_DIR wins, otherwise "fixtures" relative to the
// directory the test binary was launched from (make test runs in tests/).
// Resolved to an absolute path because the suite chdir's into a temp dir.
std::string fixturesDir() {
    const char* env = getenv("MATILDA_FIXTURES_DIR");
    char resolved[4096];
    if (realpath(env != nullptr ? env : "fixtures", resolved) == nullptr)
        return "";
    return resolved;
}

// Byte-for-byte file copy; returns false if either side fails to open.
bool copyFile(const std::string& from, const std::string& to) {
    std::ifstream src(from, std::ios::binary);
    std::ofstream dst(to, std::ios::binary);
    if (!src.is_open() || !dst.is_open()) return false;
    dst << src.rdbuf();
    return src.good() || src.eof();
}

} // namespace

class PSBoxTest : public ::testing::Test {
protected:
    static PS_Box* box;
    static std::string origDir;
    static bool built;

    static void SetUpTestSuite() {
        // Resolve fixture paths BEFORE leaving the launch directory.
        std::string fixtures = fixturesDir();
        if (fixtures.empty()) return;

        // Contain generated output files in a private temp dir. The data file
        // is copied in because ps2d.input references it relative to the CWD.
        char tmpl[] = "/tmp/matilda_psbox_XXXXXX";
        char* d = mkdtemp(tmpl);
        if (d == nullptr) return;
        if (!copyFile(fixtures + "/ps2d.data", std::string(d) + "/ps2d.data"))
            return;
        char cwd[4096];
        if (getcwd(cwd, sizeof(cwd)) != nullptr) origDir = cwd;
        if (chdir(d) != 0) return;

        // Drive the same parse sequence main.cu uses: find the "box" line,
        // construct via the factory, then let readInput() consume to endBox.
        std::ifstream in2(fixtures + "/ps2d.input");
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
        box = static_cast<PS_Box*>(b);
        built = (box != nullptr);
    }

    static void TearDownTestSuite() {
        if (!origDir.empty()) { (void)!chdir(origDir.c_str()); }
        // Deliberately leak the box: its destructor frees device memory that
        // other global CUDA teardown may also touch; leaking is safe for a
        // one-shot test process and avoids double-free noise.
        box = nullptr;
    }

    // Single-image minimum-image squared distance, matching device
    // d_pbc_dr2f() exactly (used to brute-force the neighbor list).
    static float pbcDr2(const float* ri, const float* rj,
                        const float* L, const float* Lh, int Dim) {
        float dr2 = 0.0f;
        for (int n = 0; n < Dim; n++) {
            float dr = ri[n] - rj[n];
            if (dr > Lh[n]) dr -= L[n];
            else if (dr < -Lh[n]) dr += L[n];
            dr2 += dr * dr;
        }
        return dr2;
    }
};
PS_Box* PSBoxTest::box = nullptr;
std::string PSBoxTest::origDir;
bool PSBoxTest::built = false;

// The data file's header and particle records must be parsed into the right
// counts, box lengths, types and grid.
TEST_F(PSBoxTest, ParsesDataFile) {
    ASSERT_TRUE(built) << "PS_Box failed to construct from fixture";

    EXPECT_EQ(box->returnDimension(), 2);
    EXPECT_EQ(box->nstot, 200);
    EXPECT_EQ(box->species.size(), 2u);
    EXPECT_EQ(box->nTypes, 2);

    // Box lengths come from "0 12 xlo xhi" style lines in the data file.
    EXPECT_FLOAT_EQ(box->L[0], 12.0f);
    EXPECT_FLOAT_EQ(box->L[1], 12.0f);
    EXPECT_FLOAT_EQ(box->Lh[0], 6.0f);

    // grid 24 24  ->  M = 576.
    EXPECT_EQ(box->Nx[0], 24);
    EXPECT_EQ(box->Nx[1], 24);
    EXPECT_EQ(box->M, 24 * 24);

    // No bonds/angles in the fixture.
    EXPECT_EQ(box->nBondsTot, 0);
    EXPECT_EQ(box->nAnglesTot, 0);

    // Types alternate 1,2 in the file -> 100 of each species (0-indexed).
    int n0 = 0, n1 = 0;
    for (int i = 0; i < box->nstot; i++) {
        if (box->intSpecies[i] == 0) n0++;
        else if (box->intSpecies[i] == 1) n1++;
    }
    EXPECT_EQ(n0, 100);
    EXPECT_EQ(n1, 100);
}

// The auto-created "all" group holds every particle; each type group holds
// exactly the particles of that species.
TEST_F(PSBoxTest, DefaultGroupsPartitionByType) {
    ASSERT_TRUE(built);

    int gAll = box->findGroupInteger("all");
    EXPECT_EQ(box->psGroup[gAll].nsites, box->nstot);

    // Species labels A (type 0) and B (type 1) become type groups.
    int gA = box->findGroupInteger("A");
    int gB = box->findGroupInteger("B");
    EXPECT_EQ(box->psGroup[gA].nsites, 100);
    EXPECT_EQ(box->psGroup[gB].nsites, 100);

    // Every site listed in group "A" must actually be a type-0 particle,
    // and the group must contain ALL type-0 particles (no dups, no misses).
    std::set<int> inA;
    for (int k = 0; k < box->psGroup[gA].nsites; k++) {
        int p = box->psGroup[gA].siteList[k];
        ASSERT_GE(p, 0);
        ASSERT_LT(p, box->nstot);
        EXPECT_EQ(box->intSpecies[p], 0) << "site " << p << " in group A is not type 0";
        inA.insert(p);
    }
    EXPECT_EQ((int)inA.size(), 100);
    for (int i = 0; i < box->nstot; i++) {
        if (box->intSpecies[i] == 0)
            EXPECT_TRUE(inA.count(i)) << "type-0 particle " << i << " missing from group A";
    }
}

// Build the cell-list neighbor list on the device, copy it back, and check it
// against a brute-force O(N^2) pair search with the same cutoff and PBC.
TEST_F(PSBoxTest, NeighborListMatchesBruteForce) {
    ASSERT_TRUE(built);
    ASSERT_FALSE(box->neighborLists.empty());

    PS_NeighborList* nl = box->neighborLists[0];
    nl->build();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const int nsites = nl->nsites;           // "all" -> 200
    const int maxN   = nl->maxNeighbors;
    const int Dim    = box->returnDimension();
    ASSERT_EQ(nsites, box->nstot);

    // Pull the device neighbor list and current positions back to the host.
    std::vector<int> nNeigh(nsites), flat(nsites * maxN);
    cudaMemcpy(nNeigh.data(), nl->d_nNeighbors, nsites * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(flat.data(), nl->d_neighborList, nsites * maxN * sizeof(int),
               cudaMemcpyDeviceToHost);

    std::vector<float> x(box->nstot * Dim);
    cudaMemcpy(x.data(), box->d_x, box->nstot * Dim * sizeof(float),
               cudaMemcpyDeviceToHost);

    const float rcut2 = nl->rcut2;
    // For the "all" group the slot index t equals the global particle id.
    int totalDuplicates = 0;
    int overflow = 0;

    for (int t = 0; t < nsites; t++) {
        int cnt = nNeigh[t];
        if (cnt > maxN) { overflow++; cnt = maxN; }

        std::set<int> devSet;
        for (int k = 0; k < cnt; k++) devSet.insert(flat[t * maxN + k]);
        totalDuplicates += (cnt - (int)devSet.size());

        // Brute-force neighbors of particle t.
        std::set<int> bruteSet;
        for (int j = 0; j < box->nstot; j++) {
            if (j == t) continue;
            float dr2 = pbcDr2(&x[t * Dim], &x[j * Dim], box->L, box->Lh, Dim);
            if (dr2 < rcut2) bruteSet.insert(j);
        }

        // The SET of neighbors found must match brute force exactly. (Any
        // multiplicity/double-count is checked separately below so this
        // assertion validates the cell-search pair-finding logic itself.)
        EXPECT_EQ(devSet, bruteSet) << "neighbor mismatch for particle " << t;
    }

    EXPECT_EQ(overflow, 0) << overflow << " particles exceeded maxNeighbors";

    // Document (not fail on) any double counting in the raw list: with an 8x8
    // cell grid and rcut < cell width there should be none.
    std::cout << "[ neighbor list ] total duplicate entries across all particles: "
              << totalDuplicates << std::endl;
    EXPECT_EQ(totalDuplicates, 0)
        << "raw neighbor list contains duplicate entries (double counting)";
}

// Advance the system several VV steps (zero-noise, deterministic) with the
// Gaussian potential providing forces, then assert nothing diverged: no NaN/Inf
// in positions/velocities/forces, positions stay wrapped in-box, and the total
// (unit-mass) momentum and potential energy remain finite.
TEST_F(PSBoxTest, VVStepsStayFinite) {
    ASSERT_TRUE(built);

    const int Dim = box->returnDimension();
    const int N = box->nstot;
    const int nSteps = 20;

    for (int s = 0; s < nSteps; s++) box->doTimeStep(s);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<float> x(N * Dim), v(N * Dim), f(N * Dim);
    cudaMemcpy(x.data(), box->d_x, N * Dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v.data(), box->d_v, N * Dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(f.data(), box->d_f, N * Dim * sizeof(float), cudaMemcpyDeviceToHost);

    double mom[2] = {0.0, 0.0};
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < Dim; j++) {
            float xv = x[i * Dim + j], vv = v[i * Dim + j], fv = f[i * Dim + j];
            ASSERT_TRUE(std::isfinite(xv)) << "x NaN/Inf at particle " << i;
            ASSERT_TRUE(std::isfinite(vv)) << "v NaN/Inf at particle " << i;
            ASSERT_TRUE(std::isfinite(fv)) << "f NaN/Inf at particle " << i;
            // VV wraps positions into [0, L) each step.
            EXPECT_GE(xv, 0.0f);
            EXPECT_LT(xv, box->L[j] + 1e-3f);
            mom[j] += vv;  // unit masses
        }
    }

    for (int j = 0; j < Dim; j++)
        EXPECT_TRUE(std::isfinite(mom[j])) << "total momentum not finite, dim " << j;

    // Potential energy after the run must be a finite number.
    box->computeThermoProps();
    EXPECT_TRUE(std::isfinite(box->Upe)) << "Upe not finite";
}

// --------------------------------------------------------------------------
// FFT wrapper tests. PS_Box builds fftplanSingle (2D C2C, 24x24) during
// finishInitialization, so Box::cufftWrapperSingle can be driven directly.
// The wrapper normalizes the FORWARD transform by 1/M and leaves the inverse
// un-normalized, so forward-then-inverse is the identity.
// --------------------------------------------------------------------------

// Forward then inverse must reproduce the original field (up to FP round-off).
TEST_F(PSBoxTest, CufftSingleForwardInverseRoundtrip) {
    ASSERT_TRUE(built);
    const int M = box->M;

    std::vector<cuComplex> h(M);
    for (int i = 0; i < M; i++)
        h[i] = make_cuComplex(std::sin(0.3f * i) + 0.5f, std::cos(0.11f * i));

    cuComplex *d_in, *d_k, *d_back;
    cudaMalloc(&d_in,   M * sizeof(cuComplex));
    cudaMalloc(&d_k,    M * sizeof(cuComplex));
    cudaMalloc(&d_back, M * sizeof(cuComplex));
    cudaMemcpy(d_in, h.data(), M * sizeof(cuComplex), cudaMemcpyHostToDevice);

    box->cufftWrapperSingle(d_in, d_k, 1);    // forward (normalized by 1/M)
    box->cufftWrapperSingle(d_k, d_back, -1); // inverse
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<cuComplex> out(M);
    cudaMemcpy(out.data(), d_back, M * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    for (int i = 0; i < M; i++) {
        EXPECT_NEAR(out[i].x, h[i].x, 1e-3f) << "real at " << i;
        EXPECT_NEAR(out[i].y, h[i].y, 1e-3f) << "imag at " << i;
    }
    cudaFree(d_in); cudaFree(d_k); cudaFree(d_back);
}

// Forward transform of a constant field has all spectral weight in the DC bin:
// with the 1/M normalization the k=0 component equals the constant and all
// other bins are ~0.
TEST_F(PSBoxTest, CufftSingleForwardOfConstantIsDC) {
    ASSERT_TRUE(built);
    const int M = box->M;
    const float c = 2.75f;

    std::vector<cuComplex> h(M, make_cuComplex(c, 0.0f));
    cuComplex *d_in, *d_k;
    cudaMalloc(&d_in, M * sizeof(cuComplex));
    cudaMalloc(&d_k,  M * sizeof(cuComplex));
    cudaMemcpy(d_in, h.data(), M * sizeof(cuComplex), cudaMemcpyHostToDevice);

    box->cufftWrapperSingle(d_in, d_k, 1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<cuComplex> out(M);
    cudaMemcpy(out.data(), d_k, M * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    EXPECT_NEAR(out[0].x, c, 1e-3f);
    EXPECT_NEAR(out[0].y, 0.0f, 1e-3f);
    for (int i = 1; i < M; i++) {
        EXPECT_NEAR(out[i].x, 0.0f, 1e-3f) << "nonzero DC leakage at bin " << i;
        EXPECT_NEAR(out[i].y, 0.0f, 1e-3f);
    }
    cudaFree(d_in); cudaFree(d_k);
}

// Circular convolution of a discrete delta with an arbitrary kernel returns the
// kernel. With this wrapper's normalization the convolution is
//   (a (*) b) = M * IFFT_w( FFT_w(a) .* FFT_w(b) ),
// so for a = delta at the origin the result must equal b.
TEST_F(PSBoxTest, CufftSingleDeltaConvolutionReturnsKernel) {
    ASSERT_TRUE(built);
    const int M = box->M;

    std::vector<cuComplex> delta(M, make_cuComplex(0.0f, 0.0f));
    delta[0] = make_cuComplex(1.0f, 0.0f);

    std::vector<cuComplex> kernel(M);
    for (int i = 0; i < M; i++)
        kernel[i] = make_cuComplex(std::exp(-0.01f * ((i - M / 2) * (i - M / 2))) , 0.0f);

    cuComplex *d_delta, *d_kernel, *d_dk, *d_kk, *d_prod, *d_conv;
    for (cuComplex** p : {&d_delta, &d_kernel, &d_dk, &d_kk, &d_prod, &d_conv})
        cudaMalloc(p, M * sizeof(cuComplex));
    cudaMemcpy(d_delta,  delta.data(),  M * sizeof(cuComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.data(), M * sizeof(cuComplex), cudaMemcpyHostToDevice);

    box->cufftWrapperSingle(d_delta,  d_dk, 1);   // FFT_w(delta)
    box->cufftWrapperSingle(d_kernel, d_kk, 1);   // FFT_w(kernel)
    d_multiplyCpxByCpx<<<box->M_Grid, box->M_Block>>>(d_prod, d_dk, d_kk, M);
    box->cufftWrapperSingle(d_prod, d_conv, -1);  // IFFT_w(product)
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<cuComplex> conv(M);
    cudaMemcpy(conv.data(), d_conv, M * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    for (int i = 0; i < M; i++) {
        float got = conv[i].x * float(M);   // undo the 1/M normalization
        EXPECT_NEAR(got, kernel[i].x, 1e-3f) << "kernel mismatch at " << i;
    }
    cudaFree(d_delta); cudaFree(d_kernel); cudaFree(d_dk);
    cudaFree(d_kk); cudaFree(d_prod); cudaFree(d_conv);
}
