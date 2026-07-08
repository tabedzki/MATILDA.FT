// Copyright (c) 2026 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).
//
// Unit tests for the standalone device kernels in src/device_utils.cu:
// element-wise float/complex math and the parallel-reduction sum kernel.
// Device reductions use tolerances (EXPECT_NEAR) since FP accumulation order
// on the GPU is not bit-reproducible.

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <vector>
#include <cmath>

// Kernels defined in src/device_utils.cu (object linked into the test binary).
__global__ void d_multiplyByFloat(float*, const float, const int);
__global__ void d_floatPlusEqFloat(float*, const float*, const int);
__global__ void d_multiplyCpxByCpx(cuComplex*, const cuComplex*, const cuComplex*, const int);
__global__ void d_multiplyCpxByCpxConj(float*, const cuComplex*, const cuComplex*, const int);
__global__ void d_cpxToFloat(float*, const cuComplex*, const int);
__global__ void d_floatToCpx(cuComplex*, const float*, const int);
__global__ void d_assignFloatVal(float*, const float, const int);
__global__ void sumArrayKernel(float*, float*, int);

namespace {
// Small helper: launch config for N elements with 256-thread blocks.
inline int nBlocks(int N, int block) { return (N + block - 1) / block; }
} // namespace

// d_multiplyByFloat scales every element by the scalar.
TEST(DeviceUtils, MultiplyByFloat) {
    const int N = 1000;
    std::vector<float> h(N);
    for (int i = 0; i < N; i++) h[i] = float(i) - 500.0f;

    float* d;
    cudaMalloc(&d, N * sizeof(float));
    cudaMemcpy(d, h.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    d_multiplyByFloat<<<nBlocks(N, 256), 256>>>(d, 2.5f, N);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<float> out(N);
    cudaMemcpy(out.data(), d, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) EXPECT_FLOAT_EQ(out[i], (float(i) - 500.0f) * 2.5f);
    cudaFree(d);
}

// d_floatPlusEqFloat computes out += in element-wise.
TEST(DeviceUtils, FloatPlusEqFloat) {
    const int N = 777;
    std::vector<float> a(N), b(N);
    for (int i = 0; i < N; i++) { a[i] = float(i); b[i] = float(2 * i + 1); }

    float *da, *db;
    cudaMalloc(&da, N * sizeof(float));
    cudaMalloc(&db, N * sizeof(float));
    cudaMemcpy(da, a.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    d_floatPlusEqFloat<<<nBlocks(N, 256), 256>>>(da, db, N);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<float> out(N);
    cudaMemcpy(out.data(), da, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) EXPECT_FLOAT_EQ(out[i], a[i] + b[i]);
    cudaFree(da);
    cudaFree(db);
}

// d_assignFloatVal fills the array with a constant.
TEST(DeviceUtils, AssignFloatVal) {
    const int N = 513;
    float* d;
    cudaMalloc(&d, N * sizeof(float));
    d_assignFloatVal<<<nBlocks(N, 128), 128>>>(d, -3.14f, N);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<float> out(N);
    cudaMemcpy(out.data(), d, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) EXPECT_FLOAT_EQ(out[i], -3.14f);
    cudaFree(d);
}

// d_multiplyCpxByCpx performs (a)(b) complex multiplication element-wise.
TEST(DeviceUtils, MultiplyCpxByCpx) {
    const int N = 256;
    std::vector<cuComplex> a(N), b(N);
    for (int i = 0; i < N; i++) {
        a[i] = make_cuComplex(float(i) * 0.1f, float(i) * -0.2f + 1.0f);
        b[i] = make_cuComplex(float(N - i) * 0.05f, float(i) * 0.3f);
    }

    cuComplex *da, *db, *dout;
    cudaMalloc(&da, N * sizeof(cuComplex));
    cudaMalloc(&db, N * sizeof(cuComplex));
    cudaMalloc(&dout, N * sizeof(cuComplex));
    cudaMemcpy(da, a.data(), N * sizeof(cuComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), N * sizeof(cuComplex), cudaMemcpyHostToDevice);

    d_multiplyCpxByCpx<<<nBlocks(N, 128), 128>>>(dout, da, db, N);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<cuComplex> out(N);
    cudaMemcpy(out.data(), dout, N * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        float re = a[i].x * b[i].x - a[i].y * b[i].y;
        float im = a[i].y * b[i].x + a[i].x * b[i].y;
        EXPECT_NEAR(out[i].x, re, std::abs(re) * 1e-5f + 1e-4f);
        EXPECT_NEAR(out[i].y, im, std::abs(im) * 1e-5f + 1e-4f);
    }
    cudaFree(da);
    cudaFree(db);
    cudaFree(dout);
}

// d_multiplyCpxByCpxConj computes Re(c1 * conj(c2)) = c1.x*c2.x + c1.y*c2.y.
TEST(DeviceUtils, MultiplyCpxByCpxConj) {
    const int N = 200;
    std::vector<cuComplex> c1(N), c2(N);
    for (int i = 0; i < N; i++) {
        c1[i] = make_cuComplex(float(i) * 0.01f, float(i) * 0.02f);
        c2[i] = make_cuComplex(float(i) * 0.03f - 1.0f, float(i) * -0.04f);
    }

    cuComplex *d1, *d2;
    float* dout;
    cudaMalloc(&d1, N * sizeof(cuComplex));
    cudaMalloc(&d2, N * sizeof(cuComplex));
    cudaMalloc(&dout, N * sizeof(float));
    cudaMemcpy(d1, c1.data(), N * sizeof(cuComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d2, c2.data(), N * sizeof(cuComplex), cudaMemcpyHostToDevice);

    d_multiplyCpxByCpxConj<<<nBlocks(N, 128), 128>>>(dout, d1, d2, N);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<float> out(N);
    cudaMemcpy(out.data(), dout, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++)
        EXPECT_NEAR(out[i], c1[i].x * c2[i].x + c1[i].y * c2[i].y, 1e-4f);
    cudaFree(d1);
    cudaFree(d2);
    cudaFree(dout);
}

// float -> complex -> float roundtrip preserves the real part and zeros imag.
TEST(DeviceUtils, FloatCpxRoundtrip) {
    const int N = 321;
    std::vector<float> h(N);
    for (int i = 0; i < N; i++) h[i] = std::sin(0.1f * float(i));

    float* dfin;
    cuComplex* dcpx;
    float* dfout;
    cudaMalloc(&dfin, N * sizeof(float));
    cudaMalloc(&dcpx, N * sizeof(cuComplex));
    cudaMalloc(&dfout, N * sizeof(float));
    cudaMemcpy(dfin, h.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    d_floatToCpx<<<nBlocks(N, 128), 128>>>(dcpx, dfin, N);
    d_cpxToFloat<<<nBlocks(N, 128), 128>>>(dfout, dcpx, N);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<cuComplex> cpx(N);
    std::vector<float> out(N);
    cudaMemcpy(cpx.data(), dcpx, N * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(out.data(), dfout, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(cpx[i].y, 0.0f);
        EXPECT_FLOAT_EQ(out[i], h[i]);
    }
    cudaFree(dfin);
    cudaFree(dcpx);
    cudaFree(dfout);
}

// sumArrayKernel with a single power-of-two block reduces the whole array;
// its one output element equals the host sum within FP tolerance.
TEST(DeviceUtils, SumArraySingleBlock) {
    const int block = 256;
    const int N = block;  // one full block, exact tree reduction
    std::vector<float> h(N);
    double hostSum = 0.0;
    for (int i = 0; i < N; i++) { h[i] = 0.5f * float(i) - 30.0f; hostSum += h[i]; }

    float *d, *dout;
    cudaMalloc(&d, N * sizeof(float));
    cudaMalloc(&dout, sizeof(float));
    cudaMemcpy(d, h.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    sumArrayKernel<<<1, block, block * sizeof(float)>>>(d, dout, N);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    float out = 0.0f;
    cudaMemcpy(&out, dout, sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_NEAR(out, float(hostSum), std::abs(float(hostSum)) * 1e-4f + 1e-3f);
    cudaFree(d);
    cudaFree(dout);
}

// Multi-block reduction: each block writes a partial sum; summing the partials
// on the host must recover the full array sum within tolerance.
TEST(DeviceUtils, SumArrayMultiBlock) {
    const int block = 256;
    const int nb = 8;
    const int N = block * nb;
    std::vector<float> h(N);
    double hostSum = 0.0;
    for (int i = 0; i < N; i++) { h[i] = std::cos(0.01f * float(i)); hostSum += h[i]; }

    float *d, *dpart;
    cudaMalloc(&d, N * sizeof(float));
    cudaMalloc(&dpart, nb * sizeof(float));
    cudaMemcpy(d, h.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    sumArrayKernel<<<nb, block, block * sizeof(float)>>>(d, dpart, N);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<float> part(nb);
    cudaMemcpy(part.data(), dpart, nb * sizeof(float), cudaMemcpyDeviceToHost);
    double devSum = 0.0;
    for (int i = 0; i < nb; i++) devSum += part[i];

    EXPECT_NEAR(devSum, hostSum, std::abs(hostSum) * 1e-4 + 1e-3);
    cudaFree(d);
    cudaFree(dpart);
}
