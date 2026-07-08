// Copyright (c) 2026 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).
//
// Host-side Box index/geometry math: unstack2 grid-index decomposition,
// get_r/get_rf grid positions, get_kD FFT wavevectors, and the pbc_dr2
// minimum-image displacement (float and double overloads).
//
// These are member functions, so a real box is constructed from the same
// ps2d fixture used by test_ps_box.cu (12x12 box, 24x24 grid, dx = 0.5).

#include <gtest/gtest.h>
#include "PS_Box.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

Box* BoxFactory(std::istringstream&);

namespace {

const double kTwoPi = 8.0 * std::atan(1.0);

std::string fixturesDir() {
    const char* env = getenv("MATILDA_FIXTURES_DIR");
    char resolved[4096];
    if (realpath(env != nullptr ? env : "fixtures", resolved) == nullptr)
        return "";
    return resolved;
}

bool copyFile(const std::string& from, const std::string& to) {
    std::ifstream src(from, std::ios::binary);
    std::ofstream dst(to, std::ios::binary);
    if (!src.is_open() || !dst.is_open()) return false;
    dst << src.rdbuf();
    return src.good() || src.eof();
}

} // namespace

class BoxUtilsTest : public ::testing::Test {
protected:
    static PS_Box* box;
    static std::string origDir;
    static bool built;

    static void SetUpTestSuite() {
        std::string fixtures = fixturesDir();
        if (fixtures.empty()) return;

        // finishInitialization() writes output files into the CWD; contain
        // them in a private temp dir (the data file is referenced relative
        // to the CWD by the input file, so copy it in).
        char tmpl[] = "/tmp/matilda_boxutils_XXXXXX";
        char* d = mkdtemp(tmpl);
        if (d == nullptr) return;
        if (!copyFile(fixtures + "/ps2d.data", std::string(d) + "/ps2d.data"))
            return;
        char cwd[4096];
        if (getcwd(cwd, sizeof(cwd)) != nullptr) origDir = cwd;
        if (chdir(d) != 0) return;

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
        // Leaked deliberately; see the note in test_ps_box.cu.
        box = nullptr;
    }
};
PS_Box* BoxUtilsTest::box = nullptr;
std::string BoxUtilsTest::origDir;
bool BoxUtilsTest::built = false;

// unstack2 must decompose every flat grid index into per-axis indices that
// are in range and that re-stack (row-major, x fastest) to the same index.
TEST_F(BoxUtilsTest, UnstackStackRoundtrip) {
    ASSERT_TRUE(built) << "PS_Box failed to construct from fixture";
    const int Dim = box->returnDimension();
    ASSERT_EQ(Dim, 2);

    for (int id = 0; id < box->M; id++) {
        int nn[3] = {0, 0, 0};
        box->unstack2(id, nn);
        for (int j = 0; j < Dim; j++) {
            ASSERT_GE(nn[j], 0) << "axis " << j << " at id " << id;
            ASSERT_LT(nn[j], box->Nx[j]) << "axis " << j << " at id " << id;
        }
        int restacked = nn[0] + box->Nx[0] * nn[1];
        ASSERT_EQ(restacked, id);
    }
}

// get_r (float and double) and get_rf must all return n[j] * dx[j] for the
// unstacked index — i.e. grid positions start at 0 and step by the spacing.
TEST_F(BoxUtilsTest, GetRMatchesGridArithmetic) {
    ASSERT_TRUE(built);
    const int Dim = box->returnDimension();

    for (int id = 0; id < box->M; id++) {
        int nn[3] = {0, 0, 0};
        box->unstack2(id, nn);

        float rf[3], rff[3];
        double rd[3];
        box->get_r(id, rf);
        box->get_rf(id, rff);
        box->get_r(id, rd);

        for (int j = 0; j < Dim; j++) {
            double expect = double(nn[j]) * box->dx[j];
            EXPECT_NEAR(rf[j], expect, 1e-6) << "get_r(float) id " << id;
            EXPECT_NEAR(rd[j], expect, 1e-12) << "get_r(double) id " << id;
            // get_rf duplicates the float get_r and must agree exactly.
            EXPECT_EQ(rff[j], rf[j]) << "get_rf vs get_r id " << id;
        }
    }
}

// get_kD must produce FFT-convention wavevectors: k = 2*pi*n/L for
// n < Nx/2 and the wrapped negative frequency 2*pi*(n - Nx)/L otherwise
// (the Nyquist bin n == Nx/2 maps to the negative branch), and return |k|^2.
//
// Tolerances are float-level for BOTH overloads: the double get_kD computes
// each component as PI2 * float(n) / L[i] with float L, so its extra
// precision is nominal only (related to the issue #1 simplification item
// about collapsing the duplicated float/double overloads).
TEST_F(BoxUtilsTest, GetKdFollowsFFTWraparound) {
    ASSERT_TRUE(built);
    const int Dim = box->returnDimension();

    // DC bin: zero vector, zero magnitude.
    {
        float k0[3];
        EXPECT_EQ(box->get_kD(0, k0), 0.0f);
        for (int j = 0; j < Dim; j++) EXPECT_EQ(k0[j], 0.0f);
    }

    for (int id = 0; id < box->M; id++) {
        int nn[3] = {0, 0, 0};
        box->unstack2(id, nn);

        float kf[3];
        double kd[3];
        float kmagf = box->get_kD(id, kf);
        double kmagd = box->get_kD(id, kd);

        double expectMag = 0.0;
        for (int j = 0; j < Dim; j++) {
            int n = nn[j];
            double expect = (n < box->Nx[j] / 2.0)
                                ? kTwoPi * n / box->L[j]
                                : kTwoPi * (n - box->Nx[j]) / box->L[j];
            expectMag += expect * expect;
            double compTol = 1e-5 * (1.0 + std::abs(expect));
            EXPECT_NEAR(kf[j], expect, compTol) << "float k, id " << id << " axis " << j;
            EXPECT_NEAR(kd[j], expect, compTol) << "double k, id " << id << " axis " << j;
            // Wrapped frequencies stay within (-pi/dx, pi/dx].
            double kNyq = kTwoPi * (box->Nx[j] / 2.0) / box->L[j];
            EXPECT_LE(std::abs(kd[j]), kNyq + 1e-4);
        }
        double magTol = 1e-5 * (1.0 + expectMag);
        EXPECT_NEAR(kmagf, expectMag, magTol) << "float |k|^2, id " << id;
        EXPECT_NEAR(kmagd, expectMag, magTol) << "double |k|^2, id " << id;
    }
}

// Minimum-image displacement, float overload. The fixture box is 12x12 with
// half-lengths 6x6.
TEST_F(BoxUtilsTest, PbcDr2Float) {
    ASSERT_TRUE(built);
    ASSERT_EQ(box->returnDimension(), 2);
    ASSERT_FLOAT_EQ(box->L[0], 12.0f);
    ASSERT_FLOAT_EQ(box->Lh[0], 6.0f);

    float dr[2];

    // Zero displacement.
    {
        const float a[2] = {3.25f, 7.5f};
        EXPECT_EQ(box->pbc_dr2(dr, a, a), 0.0f);
        EXPECT_EQ(dr[0], 0.0f);
        EXPECT_EQ(dr[1], 0.0f);
    }

    // In-range separation: no wrapping.
    {
        const float a[2] = {4.0f, 5.5f}, b[2] = {1.0f, 2.0f};
        float mdr2 = box->pbc_dr2(dr, a, b);
        EXPECT_FLOAT_EQ(dr[0], 3.0f);
        EXPECT_FLOAT_EQ(dr[1], 3.5f);
        EXPECT_FLOAT_EQ(mdr2, 3.0f * 3.0f + 3.5f * 3.5f);
    }

    // Wrap across the boundary: raw dr = +/-11.5 must become -/+0.5.
    {
        const float a[2] = {11.75f, 0.25f}, b[2] = {0.25f, 11.75f};
        float mdr2 = box->pbc_dr2(dr, a, b);
        EXPECT_FLOAT_EQ(dr[0], -0.5f);
        EXPECT_FLOAT_EQ(dr[1], 0.5f);
        EXPECT_FLOAT_EQ(mdr2, 0.5f);
    }

    // Exactly at the half-box boundary: dr == Lh is NOT wrapped (the test
    // is strict inequality), so the displacement stays at +/-6.
    {
        const float a[2] = {9.0f, 0.0f}, b[2] = {3.0f, 6.0f};
        float mdr2 = box->pbc_dr2(dr, a, b);
        EXPECT_FLOAT_EQ(dr[0], 6.0f);
        EXPECT_FLOAT_EQ(dr[1], -6.0f);
        EXPECT_FLOAT_EQ(mdr2, 72.0f);
    }

    // Antisymmetry: swapping the points negates dr, |dr|^2 unchanged.
    {
        const float a[2] = {10.5f, 2.0f}, b[2] = {1.25f, 8.75f};
        float dr2[2];
        float m1 = box->pbc_dr2(dr, a, b);
        float m2 = box->pbc_dr2(dr2, b, a);
        EXPECT_FLOAT_EQ(m1, m2);
        EXPECT_FLOAT_EQ(dr[0], -dr2[0]);
        EXPECT_FLOAT_EQ(dr[1], -dr2[1]);
    }
}

// Same contract for the double overload (which reads the same float L/Lh
// members; all test values are exactly representable in both precisions).
TEST_F(BoxUtilsTest, PbcDr2Double) {
    ASSERT_TRUE(built);
    ASSERT_EQ(box->returnDimension(), 2);

    double dr[2];

    {
        const double a[2] = {3.25, 7.5};
        EXPECT_EQ(box->pbc_dr2(dr, a, a), 0.0);
    }

    {
        const double a[2] = {11.75, 0.25}, b[2] = {0.25, 11.75};
        double mdr2 = box->pbc_dr2(dr, a, b);
        EXPECT_DOUBLE_EQ(dr[0], -0.5);
        EXPECT_DOUBLE_EQ(dr[1], 0.5);
        EXPECT_DOUBLE_EQ(mdr2, 0.5);
    }

    {
        const double a[2] = {10.5, 2.0}, b[2] = {1.25, 8.75};
        double dr2[2];
        double m1 = box->pbc_dr2(dr, a, b);
        double m2 = box->pbc_dr2(dr2, b, a);
        EXPECT_DOUBLE_EQ(m1, m2);
        EXPECT_DOUBLE_EQ(dr[0], -dr2[0]);
        EXPECT_DOUBLE_EQ(dr[1], -dr2[1]);
    }
}
