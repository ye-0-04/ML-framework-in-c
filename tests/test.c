#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <windows.h>


#include <immintrin.h>

#include "../include/neural.h"
#include "../include/tensors.h"
#include "../include/models.h"


 // Assuming this exists
static double now_seconds()
{
    static LARGE_INTEGER freq;
    static int init = 0;

    if (!init)
    {
        QueryPerformanceFrequency(&freq);
        init = 1;
    }

    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);

    return (double)t.QuadPart / (double)freq.QuadPart;
}
void reset_network(neural_network* nt)
{
    for (int i = 0; i < nt->size; i++)
    {
        fill(&nt->layers[i].output, 0.0f);
        fill(&nt->layers[i].input, 0.0f);
        fill(&nt->layers[i].gradient, 0.0f);
    }
}
void benchmark_forward(const char* name,
                       void (*fn)(neural_network*, tensor*),
                       neural_network* net,
                       tensor* input,
                       int runs)
{
    double total = 0.0;

    /* warmup */
    fn(net, input);

    for (int i = 0; i < runs; i++)
    {
        reset_network(net);

        double start = now_seconds();

        fn(net, input);

        double end = now_seconds();

        double dt = end - start;
        total += dt;

        printf("%s run %d: %.6f sec\n", name, i, dt);
    }

    printf("%s average: %.6f sec\n\n", name, total / runs);
}
int main()
{
    int shape[] = {3, 3, 3, 3};
    int size = 4;

    int shape_tensor[] = {1, 1, 3};
    int size_tensor = 3;

    tensor* in1 = create_tensor(shape_tensor, size_tensor);
    tensor* in2 = create_tensor(shape_tensor, size_tensor);

    fill(in1, 4.0f);
    fill(in2, 4.0f);

    neural_network nt1 = create_net(shape, 3, 4);
    neural_network nt2 = create_net(shape, 3, 4);

    int runs = 10;

    printf("Benchmarking forward_pass...\n");
    benchmark_forward("forward_pass",
                      forward_pass,
                      &nt1,
                      in1,
                      runs);

    printf("Benchmarking forward_pass_v2...\n");
    benchmark_forward("forward_pass_v2",
                      forward_pass_v2,
                      &nt2,
                      in2,
                      runs);

    return 0;
}
