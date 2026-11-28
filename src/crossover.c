/* crossover.c -- below what n is the direct O(n^2) repulsion faster than
 * Barnes-Hut? Times both on the first m points of a graph's Y0, for growing m,
 * with the same OpenMP loops as omp.c (OMP_NUM_THREADS=1: sequential).
 * Barnes-Hut = build + centers of mass + walk + Z + normalize (depth tree_depth(m));
 * direct     = all pairs + Z + normalize.
 * Each time is the median of REPS runs; also prints how far Barnes-Hut is
 * from the direct sum (max relative error of Frep, relative error of Z).
 *
 *   OMP_NUM_THREADS=1  ./crossover data/k49_norm_10NN
 *   OMP_NUM_THREADS=12 ./crossover data/k49_norm_10NN                      */
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "tsne.h"

enum { REPS = 7 };

static int cmp(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}
static double median(double *x) { qsort(x, REPS, sizeof *x, cmp); return x[REPS / 2]; }

/* Barnes-Hut repulsion of the first m points; returns Z, writes Frep */
static double barnes_hut(const double *Y, int64_t m, double theta, double *Frep, double *zi) {
  quadtree_t T;
  int L = tree_depth(m);
  int64_t leaves = (int64_t)1 << (2 * L);
  tree_init(&T, Y, m, L);
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < m; i++) {
    T.leaf[i] = point_leaf(&T, Y, i);
    #pragma omp atomic
    T.start[T.leaf[i] + 1]++;
  }
  for (int64_t c = 0; c < leaves; c++) T.start[c + 1] += T.start[c];
  #pragma omp parallel for schedule(static)
  for (int64_t c = 0; c < leaves; c++) T.next[c] = T.start[c];
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < m; i++) {
    int64_t slot;
    #pragma omp atomic capture
    slot = T.next[T.leaf[i]]++;
    T.perm[slot] = i;
  }
  int64_t cells = leaves;
  for (int l = L; l >= 0; l--, cells /= 4) {
    #pragma omp parallel for schedule(dynamic, 256)
    for (int64_t c = 0; c < cells; c++) cell_mass(&T, Y, l, c);
  }
  #pragma omp parallel for schedule(dynamic, 256)
  for (int64_t i = 0; i < m; i++) zi[i] = rep_point(&T, Y, i, theta, Frep);
  double Z = 0;
  for (int64_t i = 0; i < m; i++) Z += zi[i];
  #pragma omp parallel for schedule(static)
  for (int64_t k = 0; k < 2 * m; k++) Frep[k] /= Z;
  tree_free(&T);
  return Z;
}

/* direct O(m^2) repulsion of the first m points; returns Z, writes Frep */
static double direct(const double *Y, int64_t m, double *Frep, double *zi) {
  #pragma omp parallel for schedule(dynamic, 256)
  for (int64_t i = 0; i < m; i++) zi[i] = rep_direct_point(Y, m, i, Frep);
  double Z = 0;
  for (int64_t i = 0; i < m; i++) Z += zi[i];
  #pragma omp parallel for schedule(static)
  for (int64_t k = 0; k < 2 * m; k++) Frep[k] /= Z;
  return Z;
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s data/<graph>\n", argv[0]); return 1; }
  state_t S;
  if (read_state(argv[1], &S)) return 1;
  int64_t n = S.n;
  double *Fb = malloc(2 * n * sizeof *Fb), *Fd = malloc(2 * n * sizeof *Fd);
  double *zi = malloc(n * sizeof *zi);

  int64_t sizes[] = {100, 200, 500, 1000, 2000, 5000, 10000, 20000, n};
  printf("# %s, first m points of Y0, p = %d, theta = %.2f, median of %d runs\n",
         argv[1], omp_get_max_threads(), S.theta, REPS);
  for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; s++) {
    int64_t m = sizes[s] < n ? sizes[s] : n;
    double tb[REPS], td[REPS], Zb = 0, Zd = 0;
    for (int r = 0; r < REPS; r++) {                  /* alternate the two methods */
      double t0 = now();
      Zb = barnes_hut(S.Y0, m, S.theta, Fb, zi);
      tb[r] = now() - t0;
      t0 = now();
      Zd = direct(S.Y0, m, Fd, zi);
      td[r] = now() - t0;
    }
    double b = median(tb), d = median(td);
    printf("m=%ld L=%d barnes_hut=%.6f direct=%.6f direct/bh=%.2f Frep_err=%.1e Z_err=%.1e\n",
           (long)m, tree_depth(m), b, d, d / b, rel_err(Fb, Fd, 2 * m), fabs(Zb - Zd) / Zd);
  }
  return 0;
}
