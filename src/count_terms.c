/* count_terms.c -- how much work the Barnes-Hut walk does per point.
 * Same walk as visit() in tsne.c, but instead of forces it counts, for each point:
 *   far   = cells accepted as one body,
 *   near  = points added one by one at near leaves,
 *   cover = points covered by those terms (must be n - 1: every other point once).
 * usage: ./count_terms data/<graph> [L]                                     */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "tsne.h"

static int64_t far, near, cover;

static void count_visit(const quadtree_t *T, const double *Y, int64_t i, double theta,
                        int l, int64_t cx, int64_t cy) {
  int64_t side = (int64_t)1 << l, c = cy * side + cx;
  int64_t N = T->cnt[l][c];
  if (N == 0) return;
  double dx = Y[2 * i] - T->sumx[l][c] / N, dy = Y[2 * i + 1] - T->sumy[l][c] / N;
  double r = sqrt(2.0) * T->side / side;
  if (r < theta * sqrt(dx * dx + dy * dy)) { far++; cover += N; }
  else if (l == T->L) {
    for (int64_t k = T->start[c]; k < T->start[c + 1]; k++)
      if (T->perm[k] != i) { near++; cover++; }
  } else
    for (int b = 0; b < 2; b++)
      for (int a = 0; a < 2; a++) count_visit(T, Y, i, theta, l + 1, 2 * cx + a, 2 * cy + b);
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s data/<graph> [L]\n", argv[0]); return 1; }
  state_t S;
  if (read_state(argv[1], &S)) return 1;
  int64_t n = S.n;
  int L = argc > 2 ? atoi(argv[2]) : tree_depth(n);

  quadtree_t T;
  tree_build(&T, S.Y0, n, L);
  for (int l = L; l >= 0; l--)
    for (int64_t c = 0; c < ((int64_t)1 << (2 * l)); c++) cell_mass(&T, S.Y0, l, c);

  int64_t sum_far = 0, sum_near = 0, max_terms = 0, bad = 0;
  for (int64_t i = 0; i < n; i++) {
    far = near = cover = 0;
    count_visit(&T, S.Y0, i, S.theta, 0, 0, 0);
    if (cover != n - 1) bad++;
    sum_far += far; sum_near += near;
    if (far + near > max_terms) max_terms = far + near;
  }
  printf("%s n=%ld L=%d theta=%.2f | per point: %.0f far cells + %.0f near points = %.0f terms"
         " (max %ld), direct sum n-1 = %ld | points not covered exactly once: %ld\n",
         argv[1], (long)n, L, S.theta, (double)sum_far / n, (double)sum_near / n,
         (double)(sum_far + sum_near) / n, (long)max_terms, (long)(n - 1), (long)bad);
  tree_free(&T);
  return 0;
}
