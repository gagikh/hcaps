// Tuned launch parameters per GPU model (include/tuning.hpp). Written by the tuner:
//     ./bin/test_diagnostics tune   (build with -DLOOPS_COUNT_AS_TEMPLATE_ARGS=OFF, then rebuild ON)
// which replaces the running GPU's entries and keeps the others. "default" covers GPUs without
// their own entries. No include guard: included with different macro definitions
// (src/tuning.cpp, src/bfs_caps.cu).
//
//   GEMM_TUNED_BFS(gpu, N, R, encode_loops, bilinear_loops, reconstruct_loops, decode_loops)

#ifdef GEMM_TUNED_BFS
GEMM_TUNED_BFS("default", 512, 1, 1, 1, 1, 1)
GEMM_TUNED_BFS("default", 1024, 1, 1, 1, 1, 1)
GEMM_TUNED_BFS("default", 1024, 2, 1, 1, 1, 1)
GEMM_TUNED_BFS("default", 2048, 1, 1, 1, 1, 1)
GEMM_TUNED_BFS("default", 2048, 2, 1, 2, 2, 1)
GEMM_TUNED_BFS("default", 2048, 3, 1, 1, 1, 1)
GEMM_TUNED_BFS("default", 4096, 1, 1, 1, 1, 1)
GEMM_TUNED_BFS("default", 4096, 2, 1, 1, 4, 1)
GEMM_TUNED_BFS("default", 4096, 3, 1, 16, 4, 1)
GEMM_TUNED_BFS("default", 4096, 4, 1, 1, 4, 1)
GEMM_TUNED_BFS("default", 8192, 1, 1, 1, 1, 1)
GEMM_TUNED_BFS("default", 8192, 2, 8, 64, 32, 8)
GEMM_TUNED_BFS("default", 8192, 3, 8, 64, 64, 8)
GEMM_TUNED_BFS("default", 8192, 4, 8, 128, 32, 8)
GEMM_TUNED_BFS("default", 12288, 1, 1, 1, 1, 1)  // N = 3 * 2^12: loop counts of 8192, not tuned
GEMM_TUNED_BFS("default", 12288, 2, 8, 64, 32, 8)  // N = 3 * 2^12: loop counts of 8192, not tuned
GEMM_TUNED_BFS("default", 16384, 1, 1, 1, 1, 1)
GEMM_TUNED_BFS("default", 16384, 2, 32, 256, 128, 32)
GEMM_TUNED_BFS("default", 16384, 3, 32, 256, 256, 32)
GEMM_TUNED_BFS("default", 16384, 4, 32, 256, 256, 32)
GEMM_TUNED_BFS("default", 24576, 1, 1, 1, 1, 1)  // N = 3 * 2^13: loop counts of 16384, not tuned
GEMM_TUNED_BFS("default", 24576, 2, 32, 256, 128, 32)  // N = 3 * 2^13: loop counts of 16384, not tuned
GEMM_TUNED_BFS("default", 24576, 3, 32, 256, 256, 32)  // N = 3 * 2^13: loop counts of 16384, not tuned
GEMM_TUNED_BFS("default", 32768, 1, 1, 1, 1, 1)
GEMM_TUNED_BFS("default", 32768, 2, 128, 1024, 512, 128)
GEMM_TUNED_BFS("default", 32768, 3, 128, 2048, 512, 128)
GEMM_TUNED_BFS("default", 32768, 4, 128, 2048, 512, 128)  // not tuned yet: the R = 3 counts (as 16384, R = 4 uses its R = 3 counts)
GEMM_TUNED_BFS("default", 49152, 1, 1, 1, 1, 1)  // N = 3 * 2^14: loop counts of 32768, not tuned
GEMM_TUNED_BFS("default", 49152, 2, 128, 1024, 512, 128)  // N = 3 * 2^14: loop counts of 32768, not tuned
GEMM_TUNED_BFS("default", 65536, 1, 1, 1, 1, 1)
GEMM_TUNED_BFS("default", 65536, 2, 512, 4096, 2048, 512)  // not tuned yet: the 32768, R = 2 counts x 4 (the R = 2 rows scale x 4 per doubling of N)
GEMM_TUNED_BFS("NVIDIA GeForce RTX 4090", 1024, 1, 1, 1, 1, 1)  // CAPS 0.03891 ms with 1,1,1,1, 0.04238 ms tuned 1,2,1,1
GEMM_TUNED_BFS("NVIDIA GeForce RTX 4090", 2048, 1, 1, 1, 1, 1)  // CAPS 0.08734 ms with 1,1,1,1, 0.09265 ms tuned 2,1,2,2
GEMM_TUNED_BFS("NVIDIA GeForce RTX 4090", 2048, 2, 1, 1, 1, 1)  // CAPS 0.1247 ms with 1,2,2,1, 0.1207 ms tuned 1,1,1,1
GEMM_TUNED_BFS("NVIDIA GeForce RTX 4090", 4096, 1, 1, 1, 1, 1)  // CAPS 0.7234 ms with 1,1,1,1, 0.7217 ms tuned 1,4,1,2
GEMM_TUNED_BFS("NVIDIA GeForce RTX 4090", 4096, 2, 1, 8, 8, 1)  // CAPS 1.02 ms with 1,1,4,1, 0.9663 ms tuned 1,8,8,1
GEMM_TUNED_BFS("NVIDIA GeForce RTX 4090", 4096, 3, 1, 16, 4, 1)  // CAPS 1.556 ms with 1,16,4,1, 1.558 ms tuned 1,32,16,1
GEMM_TUNED_BFS("NVIDIA GeForce RTX 4090", 8192, 1, 1, 1, 1, 1)  // CAPS 5.086 ms with 1,1,1,1, 5.087 ms tuned 1,1,1,1
GEMM_TUNED_BFS("NVIDIA GeForce RTX 4090", 8192, 2, 8, 64, 32, 8)  // CAPS 5.802 ms with 8,64,32,8, 5.996 ms tuned 8,32,32,8
GEMM_TUNED_BFS("NVIDIA GeForce RTX 4090", 8192, 3, 8, 64, 64, 8)  // CAPS 7.071 ms with 8,64,64,8, 7.066 ms tuned 4,128,64,4
GEMM_TUNED_BFS("NVIDIA GeForce RTX 4090", 8192, 4, 8, 128, 32, 8)  // CAPS 10.81 ms with 8,128,32,8, 10.83 ms tuned 8,64,64,8
GEMM_TUNED_BFS("NVIDIA GeForce RTX 4090", 16384, 1, 1, 1, 1, 1)  // CAPS 28.92 ms with 1,1,1,1, 29.1 ms tuned 2,4,1,1; grid divisor 8: 29.07 ms (4: 29.09 ms)
GEMM_TUNED_BFS("NVIDIA GeForce RTX 4090", 16384, 2, 32, 256, 128, 32)  // CAPS 31.06 ms with 32,256,128,32, 30.85 ms tuned 16,256,128,32; grid divisor 3: 30.8 ms (4: 30.86 ms)
GEMM_TUNED_BFS("NVIDIA GeForce RTX 4090", 16384, 3, 32, 256, 256, 32)  // CAPS 35.71 ms with 32,256,256,32, 35.71 ms tuned 16,256,256,16; grid divisor 4: 35.56 ms (4: 35.56 ms)
GEMM_TUNED_BFS("NVIDIA GeForce RTX 4090", 16384, 4, 32, 256, 256, 32)  // CAPS 45.99 ms with 32,256,256,32, 45.92 ms tuned 32,512,256,32; grid divisor 4: 45.73 ms (4: 45.73 ms)
GEMM_TUNED_BFS("NVIDIA GeForce RTX 4090", 32768, 1, 1, 1, 1, 1)  // CAPS 212.5 ms with 1,1,1,1, 212.1 ms tuned 1,32,8,8; grid divisor 8: 212.3 ms (4: 212.4 ms)
GEMM_TUNED_BFS("NVIDIA GeForce RTX 4090", 32768, 2, 128, 1024, 512, 128)  // CAPS 207 ms with 128,1024,512,128, 206.6 ms tuned 128,1024,128,128; grid divisor 4: 206.4 ms (4: 206.4 ms)
GEMM_TUNED_BFS("NVIDIA H100 80GB HBM3", 1024, 1, 1, 1, 1, 1)  // CAPS 0.03252 ms with 1,1,1,1, 0.03314 ms tuned 1,1,1,1; grid divisor 6: 0.03224 ms (4: 0.03298 ms)
GEMM_TUNED_BFS("NVIDIA H100 80GB HBM3", 2048, 1, 1, 1, 1, 1)  // CAPS 0.08557 ms with 1,1,1,1, 0.08563 ms tuned 1,1,1,1; grid divisor 4: 0.08546 ms (4: 0.08546 ms)
GEMM_TUNED_BFS("NVIDIA H100 80GB HBM3", 2048, 2, 1, 1, 1, 1)  // CAPS 0.1316 ms with 1,2,2,1, 0.1282 ms tuned 1,1,1,1; grid divisor 2: 0.1261 ms (4: 0.1277 ms)
GEMM_TUNED_BFS("NVIDIA H100 80GB HBM3", 4096, 1, 1, 1, 1, 1)  // CAPS 0.4151 ms with 1,1,1,1, 0.4203 ms tuned 1,1,1,1; grid divisor 2: 0.4132 ms (4: 0.4204 ms)
GEMM_TUNED_BFS("NVIDIA H100 80GB HBM3", 4096, 2, 1, 1, 4, 1)  // CAPS 0.5901 ms with 1,1,4,1, 0.5886 ms tuned 2,8,2,1; grid divisor 2: 0.5821 ms (4: 0.597 ms)
GEMM_TUNED_BFS("NVIDIA H100 80GB HBM3", 4096, 3, 1, 16, 4, 1)  // CAPS 0.8577 ms with 1,16,4,1, 0.8501 ms tuned 2,16,4,1; grid divisor 2: 0.8531 ms (4: 0.8663 ms)
GEMM_TUNED_BFS("NVIDIA H100 80GB HBM3", 8192, 1, 1, 1, 1, 1)  // CAPS 2.556 ms with 1,1,1,1, 2.606 ms tuned 1,1,1,1; grid divisor 3: 2.539 ms (4: 2.555 ms)
GEMM_TUNED_BFS("NVIDIA H100 80GB HBM3", 8192, 2, 8, 32, 8, 4)  // CAPS 3.208 ms with 8,64,32,8, 3.123 ms tuned 8,32,8,4; grid divisor 2: 3.041 ms (4: 3.075 ms)
GEMM_TUNED_BFS("NVIDIA H100 80GB HBM3", 8192, 3, 8, 64, 16, 4)  // CAPS 4.149 ms with 8,64,64,8, 4.039 ms tuned 8,64,16,4; grid divisor 3: 3.998 ms (4: 4.036 ms)
GEMM_TUNED_BFS("NVIDIA H100 80GB HBM3", 8192, 4, 8, 64, 32, 4)  // CAPS 5.975 ms with 8,128,32,8, 5.746 ms tuned 8,64,32,4; grid divisor 2: 5.617 ms (4: 5.748 ms)
GEMM_TUNED_BFS("NVIDIA H100 80GB HBM3", 16384, 1, 1, 1, 1, 1)  // CAPS 17.5 ms with 1,1,1,1, 17.5 ms tuned 1,1,1,1; grid divisor 2: 17.4 ms (4: 17.5 ms)
GEMM_TUNED_BFS("NVIDIA H100 80GB HBM3", 16384, 2, 32, 1, 32, 16)  // CAPS 19.24 ms with 32,256,128,32, 18.7 ms tuned 32,1,32,16; grid divisor 2: 18.49 ms (4: 18.7 ms)
GEMM_TUNED_BFS("NVIDIA H100 80GB HBM3", 16384, 3, 32, 256, 64, 16)  // CAPS 22.19 ms with 32,256,256,32, 21.64 ms tuned 32,256,64,16; grid divisor 3: 21.49 ms (4: 21.65 ms)
GEMM_TUNED_BFS("NVIDIA H100 80GB HBM3", 16384, 4, 32, 256, 128, 16)  // CAPS 28.05 ms with 32,256,256,32, 27.62 ms tuned 32,256,128,16; grid divisor 2: 27.09 ms (4: 27.6 ms)
GEMM_TUNED_BFS("NVIDIA H100 80GB HBM3", 32768, 1, 1, 1, 1, 1)  // CAPS 129.6 ms with 1,1,1,1, 129.6 ms tuned 1,1,2,1; grid divisor 2: 129.2 ms (4: 129.6 ms)
GEMM_TUNED_BFS("NVIDIA H100 80GB HBM3", 32768, 2, 1, 2, 1, 64)  // CAPS 129.3 ms with 128,1024,512,128, 127.1 ms tuned 1,2,1,64; grid divisor 2: 126.3 ms (4: 127.1 ms)
GEMM_TUNED_BFS("NVIDIA H100 80GB HBM3", 32768, 3, 128, 1024, 256, 64)  // CAPS 137.1 ms with 128,2048,512,128, 132.1 ms tuned 128,1024,256,64; grid divisor 2: 131.4 ms (4: 132.1 ms)
GEMM_TUNED_BFS("NVIDIA H100 80GB HBM3", 65536, 2, 1, 2048, 512, 256)  // CAPS 640.8 ms with 512,4096,2048,512, 625.8 ms tuned 1,2048,512,256; grid divisor 3: 611.3 ms (4: 631.6 ms)
#endif
