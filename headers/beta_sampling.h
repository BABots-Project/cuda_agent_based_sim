//
// Created by nema on 11/02/25.
//

#ifndef BETA_SAMPLING_H
#define BETA_SAMPLING_H
/*
  Copyright (c) 2023, Norbert Juffa

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

  1. Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/* Compute scaled gamma function, Γ*(a) = sqrt(a/2*pi)*exp(a)*pow(a,-a)*Γ(a) */
__device__ float gammastarf (float x)
{
    const float MY_NAN_F = __int_as_float (0x7fffffff);
    float r, s;

    if (x <= 0.0f) {
        r = MY_NAN_F;
    } else if (x < 1.0f) {
        /* (0, 1): maximum error 4.17 ulp */
        if (x < 3.5e-9f) {
            const float oosqrt2pi = 0.39894228040143267794f; // 1/sqrt(2pi)
            r = oosqrt2pi / sqrtf (x);
        } else {
            r = 1.0f / x;
            r = gammastarf (x + 1.0f) * expf (fmaf (x, log1pf (r), -1.0f)) *
                sqrtf (1.0f + r);
        }
    } else {
        /* [1, INF]: maximum error 0.56 ulp */
        r = 1.0f / x;
        s =              1.24335289e-4f;  //  0x1.04c000p-13
        s = fmaf (s, r, -5.94899990e-4f); // -0x1.37e620p-11
        s = fmaf (s, r,  1.07218279e-3f); //  0x1.1910f8p-10
        s = fmaf (s, r, -2.95283855e-4f); // -0x1.35a0a8p-12
        s = fmaf (s, r, -2.67404946e-3f); // -0x1.5e7e36p-9
        s = fmaf (s, r,  3.47193284e-3f); //  0x1.c712bcp-9
        s = fmaf (s, r,  8.33333358e-2f); //  0x1.555556p-4
        r = fmaf (s, r,  1.00000000e+0f); //  0x1.000000p+0
    }
    return r;
}

__device__ float my_betaf (float a, float b)
{
    const float MY_NAN_F = __int_as_float (0x7fffffff);
    const float MY_INF_F = __int_as_float (0x7f800000);
    const float sqrt_2pi = 2.506628274631000502416f;
    float sum, mn, mx, mn_over_sum, lms, phi, plo, g, p, q, r;
    if ((a < 0) || (b < 0)) return MY_NAN_F;
    sum = a + b;
    if (sum == 0) return MY_INF_F;
    mn = fminf (a, b);
    mx = fmaxf (a, b);
    if (mn < 1) {
        r = my_betaf (mn + 1.0f, mx);
        r = sum * r / mn;
        return r;
    }
    mn_over_sum = mn / sum;
    // (mx / sum) ** mx
    lms = log1pf (-mn_over_sum);
    phi = lms * mx;
    plo = fmaf (lms, mx, -phi);
    p = expf (phi);
    p = fmaf (plo, p, p);
    // (mn / sum) ** mn
    q = powf (mn_over_sum, mn);
    g = gammastarf (mn) * gammastarf (mx) / gammastarf (sum);
    r = g * sqrt_2pi * sqrtf (1.0f / mx + 1.0f / mn) * p * q;
    return r;
}


#include <curand_kernel.h>

__device__ float sample_gamma_device(curandState *state, float alpha) {
    if (alpha < 1.0f) {
        float u = curand_uniform(state);
        return sample_gamma_device(state, alpha + 1) * powf(u, 1.0f / alpha);
    }

    float d = alpha - 1.0f / 3.0f;
    float c = 1.0f / sqrtf(9.0f * d);

    while (true) {
        float z, u;
        do {
            z = curand_normal(state);
            u = curand_uniform(state);
        } while (z <= -1.0f / c);

        float v = powf(1.0f + c * z, 3.0f);
        if (u < 1.0f - 0.0331f * (z * z) * (z * z)) return d * v;
        if (logf(u) < 0.5f * z * z + d * (1.0f - v + logf(v))) return d * v;
    }
}

__device__ float sample_beta_device(curandState *state, float alpha, float beta) {
    float G1 = sample_gamma_device(state, alpha);
    float G2 = sample_gamma_device(state, beta);
    return G1 / (G1 + G2);
}
#endif //BETA_SAMPLING_H
