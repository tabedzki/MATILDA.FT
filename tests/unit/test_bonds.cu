// Copyright (c) 2026 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).
//
// Analytic tests for the bonded interactions (src/ps_deviceBonds.cu and
// src/ps_deviceAngles.cu) driven through a real PS_Box built from
// tests/fixtures/psbond.input + psbond.data: 7 particles in a 12x12 2D box,
// no nonbonded potential, so forces() produces pure bond+angle forces.
//
//   - bond 1: particles 0-1, r = 1.5, harmonic k = 3, req = 1 (stretched)
//   - bond 2: particles 2-3, r = 1.0 (at equilibrium)
//   - angle 1: particles 4-5-6 (5 is the vertex), theta = 90 deg,
//              harmonic k = 2, theta_eq = 120 deg
//
// Conventions verified against the kernels:
//   harmonic bond   E = k (r - req)^2,        |F| = 2 k |r - req|
//   harmonic angle  E = k (theta - theta_eq)^2 per angle
//                   (each member particle accumulates the full energy;
//                    computeThermoProps divides the sum by 3),
//                   end-particle force magnitude 2 k |dtheta| / r_arm,
//                   perpendicular to its arm; vertex gets -(F_i + F_k).

#include <gtest/gtest.h>
#include "PS_Box.h"

#include <cuda_runtime.h>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

Box* BoxFactory(std::istringstream&);

namespace {

const float PI_F = 3.14159265358979f;

std::string bondFixturesDir() {
    const char* env = getenv("MATILDA_FIXTURES_DIR");
    char resolved[4096];
    if (realpath(env != nullptr ? env : "fixtures", resolved) == nullptr)
        return "";
    return resolved;
}

bool bondCopyFile(const std::string& from, const std::string& to) {
    std::ifstream src(from, std::ios::binary);
    std::ofstream dst(to, std::ios::binary);
    if (!src.is_open() || !dst.is_open()) return false;
    dst << src.rdbuf();
    return src.good() || src.eof();
}

} // namespace

class BondAngleTest : public ::testing::Test {
protected:
    static PS_Box* box;
    static std::string origDir;
    static bool built;

    // Same construction pattern as PSBoxTest (see test_ps_box.cu): chdir into
    // a private temp dir because finishInitialization() writes output files,
    // then drive the main.cu parse sequence on the fixture input.
    static void SetUpTestSuite() {
        std::string fixtures = bondFixturesDir();
        if (fixtures.empty()) return;

        char tmpl[] = "/tmp/matilda_bond_XXXXXX";
        char* d = mkdtemp(tmpl);
        if (d == nullptr) return;
        if (!bondCopyFile(fixtures + "/psbond.data", std::string(d) + "/psbond.data"))
            return;
        char cwd[4096];
        if (getcwd(cwd, sizeof(cwd)) != nullptr) origDir = cwd;
        if (chdir(d) != 0) return;

        std::ifstream in2(fixtures + "/psbond.input");
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
        box = nullptr;  // deliberately leaked, see test_ps_box.cu
    }

    // Zero d_f and evaluate bond+angle forces (no potentials are configured,
    // and group force-mapping is a no-op unless a potential enabled it).
    static std::vector<float> evalForces() {
        const int n = box->nstot * box->returnDimension();
        cudaMemset(box->d_f, 0, n * sizeof(float));
        box->forces();
        cudaDeviceSynchronize();
        std::vector<float> f(n);
        cudaMemcpy(f.data(), box->d_f, n * sizeof(float), cudaMemcpyDeviceToHost);
        return f;
    }
};
PS_Box* BondAngleTest::box = nullptr;
std::string BondAngleTest::origDir;
bool BondAngleTest::built = false;

// The bonded fixture must parse into the expected topology.
TEST_F(BondAngleTest, ParsesBondedFixture) {
    ASSERT_TRUE(built) << "PS_Box failed to construct from psbond fixture";

    EXPECT_EQ(box->returnDimension(), 2);
    EXPECT_EQ(box->nstot, 7);
    // SUSPECTED BUG (not in issue #1): readDataConfig reads nBondsTot/
    // nAnglesTot from the data-file header (PS_Box_initialization.cu:809-810),
    // then finishInitialization re-counts bonds/angles from the per-particle
    // tables *without zeroing first* (PS_Box_initialization.cu:430-443), so on
    // the data-file init path both totals come out exactly doubled. The
    // doubled counts propagate into writeDataConfig headers and the GSD
    // topology chunk sizes. The fixture has 2 bonds and 1 angle; these
    // characterization assertions pin the current (buggy) doubled values and
    // should be tightened to 2 and 1 when the bug is fixed.
    EXPECT_EQ(box->nBondsTot, 4);
    EXPECT_EQ(box->nAnglesTot, 2);
    EXPECT_EQ(box->nBondTypes, 1);
    EXPECT_EQ(box->nAngleTypes, 1);

    // Per-particle bond bookkeeping: 0-1 and 2-3 bonded, angle members 4,5,6.
    EXPECT_EQ(box->nBonds[0], 1);
    EXPECT_EQ(box->nBonds[1], 1);
    EXPECT_EQ(box->bondedTo[0 * box->MAXBONDS], 1);
    EXPECT_EQ(box->bondedTo[1 * box->MAXBONDS], 0);
    EXPECT_EQ(box->nBonds[4], 0);
    EXPECT_EQ(box->nAngles[4], 1);
    EXPECT_EQ(box->nAngles[5], 1);
    EXPECT_EQ(box->nAngles[6], 1);
}

// Ubond = k (r - req)^2 summed over bonds: only the stretched bond
// contributes, 3 * 0.5^2 = 0.75. Uangle = k (theta - theta_eq)^2 =
// 2 * (pi/2 - 2pi/3)^2 = 2 (pi/6)^2.
TEST_F(BondAngleTest, BondAndAngleEnergiesMatchAnalytic) {
    ASSERT_TRUE(built);

    box->computeThermoProps();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_NEAR(box->Ubond, 0.75f, 1e-4f);

    const float dth = PI_F / 2.0f - 2.0f * PI_F / 3.0f;   // -pi/6
    EXPECT_NEAR(box->Uangle, 2.0f * dth * dth, 1e-4f);

    // With no nonbonded potentials, Upe is exactly the bonded total.
    EXPECT_NEAR(box->Upe, 0.75f + 2.0f * dth * dth, 1e-4f);
}

// The stretched harmonic bond (k=3, req=1, r=1.5 along +x) must pull the two
// particles together with |F| = 2 k (r - req) = 3, and the equilibrium-length
// bond must be force-free.
TEST_F(BondAngleTest, HarmonicBondForcesMatchAnalytic) {
    ASSERT_TRUE(built);

    std::vector<float> f = evalForces();

    // Particle 0 at x=2, particle 1 at x=3.5: 0 is pulled +x, 1 is pulled -x.
    EXPECT_NEAR(f[0 * 2 + 0],  3.0f, 1e-4f);
    EXPECT_NEAR(f[0 * 2 + 1],  0.0f, 1e-4f);
    EXPECT_NEAR(f[1 * 2 + 0], -3.0f, 1e-4f);
    EXPECT_NEAR(f[1 * 2 + 1],  0.0f, 1e-4f);

    // Newton's third law, exactly antisymmetric to float round-off.
    EXPECT_NEAR(f[0 * 2 + 0] + f[1 * 2 + 0], 0.0f, 1e-5f);

    // Bond 2-3 sits at its equilibrium length: no force on either particle.
    for (int p : {2, 3})
        for (int j = 0; j < 2; j++)
            EXPECT_NEAR(f[p * 2 + j], 0.0f, 1e-5f)
                << "equilibrium bond leaks force onto particle " << p;
}

// Harmonic angle at 90 deg with theta_eq = 120 deg: each end particle feels
// |F| = 2 k |dtheta| / r_arm perpendicular to its arm, directed to open the
// angle; the vertex takes -(F_i + F_k); the triplet's net force is zero.
TEST_F(BondAngleTest, HarmonicAngleForcesMatchAnalytic) {
    ASSERT_TRUE(built);

    std::vector<float> f = evalForces();

    // Geometry: vertex p5 at (6,9), end p4 at (7,9), end p6 at (6,10).
    // dtheta = -pi/6, arms have unit length -> |F_end| = 2*2*(pi/6) = 2pi/3.
    const float fmag = 2.0f * 2.0f * (PI_F / 6.0f);

    // End p4 (arm +x): force is -y (rotates away from p6, opening the angle).
    EXPECT_NEAR(f[4 * 2 + 0], 0.0f,  1e-4f);
    EXPECT_NEAR(f[4 * 2 + 1], -fmag, 1e-4f);

    // End p6 (arm +y): force is -x.
    EXPECT_NEAR(f[6 * 2 + 0], -fmag, 1e-4f);
    EXPECT_NEAR(f[6 * 2 + 1], 0.0f,  1e-4f);

    // Vertex p5 balances both ends.
    EXPECT_NEAR(f[5 * 2 + 0], fmag, 1e-4f);
    EXPECT_NEAR(f[5 * 2 + 1], fmag, 1e-4f);

    // End forces are perpendicular to their arms (arm p4: +x, arm p6: +y).
    EXPECT_NEAR(f[4 * 2 + 0] * 1.0f + f[4 * 2 + 1] * 0.0f, 0.0f, 1e-4f);
    EXPECT_NEAR(f[6 * 2 + 0] * 0.0f + f[6 * 2 + 1] * 1.0f, 0.0f, 1e-4f);

    // Net force over the whole system is zero (bond pairs cancel too).
    float net[2] = {0.0f, 0.0f};
    for (int p = 0; p < box->nstot; p++)
        for (int j = 0; j < 2; j++) net[j] += f[p * 2 + j];
    EXPECT_NEAR(net[0], 0.0f, 1e-4f);
    EXPECT_NEAR(net[1], 0.0f, 1e-4f);
}
