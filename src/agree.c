/* agree.c -- do all four versions agree? Compares results/<graph>-{omp,cilk,pth}.mat
 * with results/<graph>-seq.mat (max relative difference of each output).
 *   ./agree <graph> <n>      e.g. ./agree k49_norm_10NN 38547                */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tsne.h"
int main(int argc, char **argv) {           /* argv[1] = graph name, argv[2] = n */
  int64_t n = atoll(argv[2]); const char *v[] = {"seq", "omp", "cilk", "pth"};
  double *b[4], *fa[4], *fr[4], *g[4], Z[4]; char path[512];
  for (int k = 0; k < 4; k++) {
    b[k] = malloc(n * 8); fa[k] = malloc(16 * n); fr[k] = malloc(16 * n); g[k] = malloc(16 * n);
    snprintf(path, sizeof path, "results/%s-%s.mat", argv[1], v[k]);
    if (read_results(path, n, b[k], fa[k], fr[k], &Z[k], g[k])) { printf("cannot read %s\n", path); return 1; }
  }
  for (int k = 1; k < 4; k++)
    printf("%s %-4s vs seq: beta %.1e | Fattr %.1e | Frep %.1e | Z %.1e | grad %.1e\n", argv[1], v[k],
           rel_err(b[k], b[0], n), rel_err(fa[k], fa[0], 2*n), rel_err(fr[k], fr[0], 2*n),
           fabs(Z[k] - Z[0]) / Z[0], rel_err(g[k], g[0], 2*n));
}
