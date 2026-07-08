// Copyright (c) 2026 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).
//
// Unit tests for the vendored GSD library (src/gsd.c / src/gsd.h). Pure C,
// no GPU: exercise the create -> write -> close -> reopen -> read roundtrip,
// metadata, multi-frame indexing, and error handling on bad opens.

#include <gtest/gtest.h>

extern "C" {
#include "gsd.h"
}

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

namespace {

// Build a unique scratch path under /tmp for one test's GSD file. Uses
// mkdtemp so tests are hermetic and never collide with each other.
std::string makeTempGsdPath() {
    char dirTemplate[] = "/tmp/matilda_gsd_XXXXXX";
    char* dir = mkdtemp(dirTemplate);
    EXPECT_NE(dir, nullptr) << "mkdtemp failed";
    return std::string(dir) + "/test.gsd";
}

} // namespace

// A freshly created file must open cleanly and report zero frames and the
// application/schema metadata we wrote.
TEST(Gsd, CreateOpenMetadataRoundtrip) {
    std::string path = makeTempGsdPath();

    ASSERT_EQ(gsd_create(path.c_str(), "matilda-test", "hoomd",
                         gsd_make_version(1, 4)),
              GSD_SUCCESS);

    gsd_handle handle;
    ASSERT_EQ(gsd_open(&handle, path.c_str(), GSD_OPEN_READONLY), GSD_SUCCESS);

    EXPECT_STREQ(handle.header.application, "matilda-test");
    EXPECT_STREQ(handle.header.schema, "hoomd");
    EXPECT_EQ(handle.header.schema_version, gsd_make_version(1, 4));
    EXPECT_EQ(gsd_get_nframes(&handle), 0u);

    EXPECT_EQ(gsd_close(&handle), GSD_SUCCESS);
    std::remove(path.c_str());
}

// Write a float chunk, end the frame, close, reopen and verify the data and
// its N/M/type metadata survive the roundtrip exactly.
TEST(Gsd, WriteReadFloatChunk) {
    std::string path = makeTempGsdPath();

    gsd_handle handle;
    ASSERT_EQ(gsd_create_and_open(&handle, path.c_str(), "matilda-test",
                                  "hoomd", gsd_make_version(1, 4),
                                  GSD_OPEN_READWRITE, 0),
              GSD_SUCCESS);

    const uint64_t N = 5;
    const uint32_t M = 3;
    float data[N * M];
    for (uint64_t i = 0; i < N * M; i++) data[i] = 1.5f * float(i) - 2.0f;

    ASSERT_EQ(gsd_write_chunk(&handle, "particles/position", GSD_TYPE_FLOAT,
                              N, M, 0, data),
              GSD_SUCCESS);
    ASSERT_EQ(gsd_end_frame(&handle), GSD_SUCCESS);
    ASSERT_EQ(gsd_close(&handle), GSD_SUCCESS);

    // Reopen read-only and pull the chunk back.
    ASSERT_EQ(gsd_open(&handle, path.c_str(), GSD_OPEN_READONLY), GSD_SUCCESS);
    EXPECT_EQ(gsd_get_nframes(&handle), 1u);

    const gsd_index_entry* entry =
        gsd_find_chunk(&handle, 0, "particles/position");
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->N, N);
    EXPECT_EQ(entry->M, M);
    EXPECT_EQ(entry->type, (uint8_t)GSD_TYPE_FLOAT);

    float readback[N * M];
    ASSERT_EQ(gsd_read_chunk(&handle, readback, entry), GSD_SUCCESS);
    for (uint64_t i = 0; i < N * M; i++)
        EXPECT_FLOAT_EQ(readback[i], data[i]) << "at index " << i;

    EXPECT_EQ(gsd_close(&handle), GSD_SUCCESS);
    std::remove(path.c_str());
}

// Multiple frames each with their own chunk must be indexed independently
// and read back per-frame.
TEST(Gsd, MultipleFramesIndependent) {
    std::string path = makeTempGsdPath();

    gsd_handle handle;
    ASSERT_EQ(gsd_create_and_open(&handle, path.c_str(), "matilda-test",
                                  "hoomd", gsd_make_version(1, 4),
                                  GSD_OPEN_READWRITE, 0),
              GSD_SUCCESS);

    const int nFrames = 4;
    for (int fr = 0; fr < nFrames; fr++) {
        uint32_t step = (uint32_t)(fr * 100);
        ASSERT_EQ(gsd_write_chunk(&handle, "configuration/step",
                                  GSD_TYPE_UINT32, 1, 1, 0, &step),
                  GSD_SUCCESS);
        ASSERT_EQ(gsd_end_frame(&handle), GSD_SUCCESS);
    }
    ASSERT_EQ(gsd_close(&handle), GSD_SUCCESS);

    ASSERT_EQ(gsd_open(&handle, path.c_str(), GSD_OPEN_READONLY), GSD_SUCCESS);
    EXPECT_EQ(gsd_get_nframes(&handle), (uint64_t)nFrames);

    for (int fr = 0; fr < nFrames; fr++) {
        const gsd_index_entry* entry =
            gsd_find_chunk(&handle, fr, "configuration/step");
        ASSERT_NE(entry, nullptr) << "frame " << fr;
        uint32_t step = 0;
        ASSERT_EQ(gsd_read_chunk(&handle, &step, entry), GSD_SUCCESS);
        EXPECT_EQ(step, (uint32_t)(fr * 100));
    }

    EXPECT_EQ(gsd_close(&handle), GSD_SUCCESS);
    std::remove(path.c_str());
}

// gsd_sizeof_type must report correct widths and 0 for an invalid type.
TEST(Gsd, SizeofType) {
    EXPECT_EQ(gsd_sizeof_type(GSD_TYPE_UINT8), 1u);
    EXPECT_EQ(gsd_sizeof_type(GSD_TYPE_UINT32), 4u);
    EXPECT_EQ(gsd_sizeof_type(GSD_TYPE_FLOAT), 4u);
    EXPECT_EQ(gsd_sizeof_type(GSD_TYPE_DOUBLE), 8u);
    EXPECT_EQ(gsd_sizeof_type(GSD_TYPE_INT64), 8u);
    EXPECT_EQ(gsd_sizeof_type((gsd_type)0), 0u);
}

// gsd_make_version packs major/minor into aaaa.bbbb halves.
TEST(Gsd, MakeVersionPacking) {
    uint32_t v = gsd_make_version(1, 4);
    EXPECT_EQ(v >> 16, 1u);
    EXPECT_EQ(v & 0xffff, 4u);
}

// Opening a path that does not exist must fail (not crash) with an IO error.
TEST(Gsd, OpenNonexistentFails) {
    gsd_handle handle;
    int rc = gsd_open(&handle,
                      "/tmp/matilda_gsd_does_not_exist_zzz/nope.gsd",
                      GSD_OPEN_READONLY);
    EXPECT_LT(rc, 0);
    EXPECT_EQ(rc, GSD_ERROR_IO);
}

// Opening an existing file whose contents are not a GSD file must be rejected.
TEST(Gsd, OpenNonGsdFileFails) {
    char dirTemplate[] = "/tmp/matilda_gsd_XXXXXX";
    char* dir = mkdtemp(dirTemplate);
    ASSERT_NE(dir, nullptr);
    std::string path = std::string(dir) + "/garbage.gsd";

    FILE* fp = std::fopen(path.c_str(), "wb");
    ASSERT_NE(fp, nullptr);
    const char* junk = "this is definitely not a GSD file header payload";
    std::fwrite(junk, 1, std::strlen(junk), fp);
    std::fclose(fp);

    gsd_handle handle;
    int rc = gsd_open(&handle, path.c_str(), GSD_OPEN_READONLY);
    EXPECT_EQ(rc, GSD_ERROR_NOT_A_GSD_FILE);

    std::remove(path.c_str());
}
