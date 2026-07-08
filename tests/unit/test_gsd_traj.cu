// Copyright (c) 2026 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).
//
// Tests for PS_Box GSD trajectory output (src/PS_Box_gsdUtils.cu,
// writeGSDtraj) using the ps2d fixture. Frames are verified by reading the
// file back with the raw GSD C API rather than PS_Box::readGSDtraj, because
// the read path is unusable for 2D boxes on current main:
//
//   * readGSDtraj mallocs h_ns_float as nstot*Dim floats but the
//     "particles/position" chunk is always nstot*3 floats, so for Dim == 2
//     gsd_read_chunk overruns the buffer, and the i*Dim+j unpacking stride
//     mismatches the file's i*3+j layout (src/PS_Box_gsdUtils.cu:544-559).
//
// Known-bug caveats from issue #1 finding 9 (unfixed on this branch):
//   * "configuration/step" is written as GSD_TYPE_UINT64 from a 4-byte
//     unsigned int (src/PS_Box_gsdUtils.cu:38-42,113-118), so the stored
//     64-bit value has garbage high bits -- tests must not assert on it.
//   * readGSDtraj's frame bound check is off by one (frame_num > tmp_frame
//     instead of >=), so out-of-range behavior is also not asserted here.

#include <gtest/gtest.h>
#include "PS_Box.h"

extern "C" {
#include "gsd.h"
}

#include <cuda_runtime.h>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

Box* BoxFactory(std::istringstream&);

namespace {

std::string trajFixturesDir() {
    const char* env = getenv("MATILDA_FIXTURES_DIR");
    char resolved[4096];
    if (realpath(env != nullptr ? env : "fixtures", resolved) == nullptr)
        return "";
    return resolved;
}

bool trajCopyFile(const std::string& from, const std::string& to) {
    std::ifstream src(from, std::ios::binary);
    std::ofstream dst(to, std::ios::binary);
    if (!src.is_open() || !dst.is_open()) return false;
    dst << src.rdbuf();
    return src.good() || src.eof();
}

// Read an entire chunk of frame `frame` into `out` (sized N*M elements by the
// caller). Returns false if the chunk is missing or the read fails.
template <typename T>
bool readChunk(gsd_handle* h, uint64_t frame, const char* name,
               std::vector<T>& out) {
    const gsd_index_entry* e = gsd_find_chunk(h, frame, name);
    if (e == nullptr) return false;
    out.resize(size_t(e->N) * e->M);
    return gsd_read_chunk(h, out.data(), e) == GSD_SUCCESS;
}

} // namespace

class GsdTrajTest : public ::testing::Test {
protected:
    static PS_Box* box;
    static std::string origDir;
    static bool built;

    // Same construction pattern as PSBoxTest (test_ps_box.cu): private temp
    // dir + the main.cu parse sequence over the ps2d fixture.
    static void SetUpTestSuite() {
        std::string fixtures = trajFixturesDir();
        if (fixtures.empty()) return;

        char tmpl[] = "/tmp/matilda_gsdtraj_XXXXXX";
        char* d = mkdtemp(tmpl);
        if (d == nullptr) return;
        if (!trajCopyFile(fixtures + "/ps2d.data", std::string(d) + "/ps2d.data"))
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
        box = nullptr;  // deliberately leaked, see test_ps_box.cu
    }
};
PS_Box* GsdTrajTest::box = nullptr;
std::string GsdTrajTest::origDir;
bool GsdTrajTest::built = false;

// A fresh writeGSDtraj at totSteps == 0 creates the file from scratch
// (gsd_create_and_open uses O_TRUNC) with one frame holding the full set of
// configuration chunks.
TEST_F(GsdTrajTest, WriteCreatesSingleFrameWithMetadata) {
    ASSERT_TRUE(built) << "PS_Box failed to construct from fixture";

    box->gsd_name = "roundtrip.gsd";
    box->totSteps = 0;
    box->writeGSDtraj();

    gsd_handle h;
    ASSERT_EQ(gsd_open(&h, "roundtrip.gsd", GSD_OPEN_READONLY), GSD_SUCCESS);
    EXPECT_EQ(gsd_get_nframes(&h), 1u);

    std::vector<unsigned int> n;
    ASSERT_TRUE(readChunk(&h, 0, "particles/N", n));
    ASSERT_EQ(n.size(), 1u);
    EXPECT_EQ(n[0], (unsigned int)box->nstot);

    // The box chunk carries L (z slot is a placeholder 5.0 for 2D runs).
    std::vector<float> gsdBox;
    ASSERT_TRUE(readChunk(&h, 0, "configuration/box", gsdBox));
    ASSERT_EQ(gsdBox.size(), 6u);
    EXPECT_FLOAT_EQ(gsdBox[0], box->L[0]);
    EXPECT_FLOAT_EQ(gsdBox[1], box->L[1]);

    // Type ids are written 1-based from intSpecies.
    std::vector<unsigned int> typeids;
    ASSERT_TRUE(readChunk(&h, 0, "particles/typeid", typeids));
    ASSERT_EQ((int)typeids.size(), box->nstot);
    for (int i = 0; i < box->nstot; i++)
        EXPECT_EQ(typeids[i], (unsigned int)(box->intSpecies[i] + 1))
            << "typeid mismatch at particle " << i;

    // Unit masses from the fixture's Masses section.
    std::vector<float> masses;
    ASSERT_TRUE(readChunk(&h, 0, "particles/mass", masses));
    ASSERT_EQ((int)masses.size(), box->nstot);
    for (int i = 0; i < box->nstot; i++) EXPECT_FLOAT_EQ(masses[i], 1.0f);

    // Topology chunks exist even for an unbonded system.
    std::vector<unsigned int> nb, na;
    ASSERT_TRUE(readChunk(&h, 0, "bonds/N", nb));
    EXPECT_EQ(nb[0], 0u);
    ASSERT_TRUE(readChunk(&h, 0, "angles/N", na));
    EXPECT_EQ(na[0], 0u);

    EXPECT_EQ(gsd_close(&h), GSD_SUCCESS);
}

// Positions are stored HOOMD-style: shifted by -Lh into a centered box, N x 3
// with z = 0 for 2D. Reading them back and re-shifting must reproduce the
// device positions exactly (same floats, no arithmetic beyond the shift).
TEST_F(GsdTrajTest, PositionsRoundTripThroughShift) {
    ASSERT_TRUE(built);

    box->gsd_name = "roundtrip.gsd";
    box->totSteps = 0;
    box->writeGSDtraj();

    const int Dim = box->returnDimension();
    std::vector<float> x(box->nstot * Dim);
    cudaMemcpy(x.data(), box->d_x, x.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    gsd_handle h;
    ASSERT_EQ(gsd_open(&h, "roundtrip.gsd", GSD_OPEN_READONLY), GSD_SUCCESS);
    std::vector<float> pos;
    ASSERT_TRUE(readChunk(&h, 0, "particles/position", pos));
    ASSERT_EQ(pos.size(), size_t(box->nstot) * 3);

    for (int i = 0; i < box->nstot; i++) {
        for (int j = 0; j < Dim; j++)
            EXPECT_FLOAT_EQ(pos[i * 3 + j], x[i * Dim + j] - box->Lh[j])
                << "position mismatch, particle " << i << " dim " << j;
        if (Dim == 2)
            EXPECT_FLOAT_EQ(pos[i * 3 + 2], 0.0f) << "2D z not zeroed, particle " << i;
    }
    EXPECT_EQ(gsd_close(&h), GSD_SUCCESS);
}

// At totSteps != 0 writeGSDtraj appends a frame containing just the step and
// positions; earlier frames and their metadata stay intact and readable.
TEST_F(GsdTrajTest, AppendsSecondFrame) {
    ASSERT_TRUE(built);

    box->gsd_name = "append.gsd";
    box->totSteps = 0;
    box->writeGSDtraj();
    box->totSteps = 100;
    box->writeGSDtraj();
    box->totSteps = 0;  // restore for any later test

    gsd_handle h;
    ASSERT_EQ(gsd_open(&h, "append.gsd", GSD_OPEN_READONLY), GSD_SUCCESS);
    EXPECT_EQ(gsd_get_nframes(&h), 2u);

    // Frame 1 has its own positions; metadata chunks live only in frame 0.
    EXPECT_NE(gsd_find_chunk(&h, 1, "particles/position"), nullptr);
    EXPECT_NE(gsd_find_chunk(&h, 1, "configuration/step"), nullptr);
    EXPECT_EQ(gsd_find_chunk(&h, 1, "particles/typeid"), nullptr);
    EXPECT_NE(gsd_find_chunk(&h, 0, "particles/typeid"), nullptr);

    // The box hasn't moved between the writes, so both frames' positions match.
    std::vector<float> p0, p1;
    ASSERT_TRUE(readChunk(&h, 0, "particles/position", p0));
    ASSERT_TRUE(readChunk(&h, 1, "particles/position", p1));
    ASSERT_EQ(p0.size(), p1.size());
    for (size_t i = 0; i < p0.size(); i++)
        EXPECT_FLOAT_EQ(p0[i], p1[i]) << "position drift at flat index " << i;

    // NOTE: the step value itself is NOT checked -- it is written as
    // GSD_TYPE_UINT64 from a 4-byte variable (issue #1 finding 9), so its
    // high 32 bits are unspecified garbage on current main.

    EXPECT_EQ(gsd_close(&h), GSD_SUCCESS);
}
