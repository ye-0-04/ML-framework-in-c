#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>
#include <windows.h>


#include <immintrin.h>

#include "../include/neural.h"
#include "../include/tensors.h"
#include "../include/models.h"

static double get_time_seconds()
{
    static LARGE_INTEGER freq;
    static int init = 0;

    if (!init)
    {
        QueryPerformanceFrequency(&freq);
        init = 1;
    }

    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);

    return (double)counter.QuadPart / (double)freq.QuadPart;
}

/* ---------------- benchmark struct ---------------- */

typedef struct {
    const char* name;
    void (*fn)(tensor*, tensor*, tensor*);
    int wins;
} bench_fn;

/* ---------------- main ---------------- */

int main()
{
    int runs = 30;
    int x = 512;
    int dims = 4;
    int shape[] = {1, 3, x, x};

    tensor* a = create_tensor(shape, dims);
    tensor* b = create_tensor(shape, dims);
    tensor* output = create_tensor(shape, dims);

    fill(a, 1.0f);
    fill(b, 2.0f);
    fill(output, 0.0f);

    bench_fn funcs[] = {
        {"bare_hands",       matmul_bare_hands,      0},
        {"bare_hands_v2",    matmul_bare_hands_v2,   0},
        {"simd_avx2",        matmul_simd_avx2,       0},
        {"simd_omp_batch",   matmul_simd_omp_batch,  0}
    };

    int nfuncs = sizeof(funcs) / sizeof(funcs[0]);

    /* warm-up */
    for (int i = 0; i < nfuncs; i++)
    {
        fill(output, 0.0f);
        funcs[i].fn(output, a, b);
    }

    printf("Warm-up done\n\n");

    /* benchmark */
    for (int r = 0; r < runs; r++)
    {
        printf("===== RUN %d =====\n", r + 1);

        double best_time = 1e100;
        int winner = -1;

        for (int f = 0; f < nfuncs; f++)
        {
            fill(output, 0.0f);

            double start = get_time_seconds();
            funcs[f].fn(output, a, b);
            double end = get_time_seconds();

            double t = end - start;

            printf("%-18s : %.6f sec\n", funcs[f].name, t);

            if (t < best_time)
            {
                best_time = t;
                winner = f;
            }
        }

        funcs[winner].wins++;
        printf("Winner: %s\n\n", funcs[winner].name);
    }

    printf("\n========================\n");
    printf("FINAL RESULTS\n");
    printf("========================\n");

    for (int i = 0; i < nfuncs; i++)
    {
        printf("%-18s won %d/%d runs\n",
               funcs[i].name,
               funcs[i].wins,
               runs);
    }

    free_tensor(a);
    free_tensor(b);
    free_tensor(output);

    return 0;
}