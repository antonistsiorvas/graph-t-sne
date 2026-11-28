/* embed.c -- optional part: "see the picture". Iterates the gradient from Y0
 * with the optimizer of the README (momentum 0.8, step 200, one gain per point):
 *   gamma_i <- gamma_i + 0.2  if g_i . v_i < 0,  else 0.8 gamma_i  (never < 0.01)
 *   v_i     <- 0.8 v_i - 200 gamma_i g_i
 *   y_i     <- y_i + v_i
 * P is computed once; each iteration recomputes Fattr and, with a new quadtree,
 * Frep and Z at the current positions (the same OpenMP loops as omp.c).
 * Writes results/embed_<graph>.txt: one line "x y label" per point.
 *
 *   OMP_NUM_THREADS=12 ./embed data/<graph> [iterations]   (default 300)   */
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tsne.h"

/* gradient 4 (Fattr - Frep) at Y; returns Z */
static double gradient(const graph_t *W, const double *P, const double *Y, int64_t n,
                       double theta, int L, double *Fattr, double *Frep, double *zi,
                       double *grad) {
  #pragma omp parallel for schedule(dynamic, 256)
  for (int64_t i = 0; i < n; i++) attr_row(W, P, Y, i, Fattr);

  quadtree_t T;
  int64_t leaves = (int64_t)1 << (2 * L);
  tree_init(&T, Y, n, L);
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; i++) {
    T.leaf[i] = point_leaf(&T, Y, i);
    #pragma omp atomic
    T.start[T.leaf[i] + 1]++;
  }
  for (int64_t c = 0; c < leaves; c++) T.start[c + 1] += T.start[c];
  #pragma omp parallel for schedule(static)
  for (int64_t c = 0; c < leaves; c++) T.next[c] = T.start[c];
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; i++) {
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
  for (int64_t i = 0; i < n; i++) zi[i] = rep_point(&T, Y, i, theta, Frep);
  tree_free(&T);

  double Z = 0;
  for (int64_t i = 0; i < n; i++) Z += zi[i];
  #pragma omp parallel for schedule(static)
  for (int64_t k = 0; k < 2 * n; k++) grad[k] = 4 * (Fattr[k] - Frep[k] / Z);
  return Z;
}

/* KL(P||Q) = sum p log p - sum p log q over the edges, q = (1+d^2)^-1 / Z */
static double kl(const graph_t *W, const double *P, const double *Y, double Z) {
  double C = 0;
  #pragma omp parallel for reduction(+:C) schedule(dynamic, 256)
  for (int64_t j = 0; j < W->n; j++)
    for (int64_t k = W->colptr[j]; k < W->colptr[j + 1]; k++) {
      int64_t i = W->row[k];
      double dx = Y[2 * i] - Y[2 * j], dy = Y[2 * i + 1] - Y[2 * j + 1];
      if (P[k] > 0) C += P[k] * log(P[k] * (1 + dx * dx + dy * dy) * Z);
    }
  return C;
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s data/<graph> [iterations]\n", argv[0]); return 1; }
  int iters = argc > 2 ? atoi(argv[2]) : 300;
  graph_t W; state_t S;
  if (read_graph(argv[1], &W) || read_state(argv[1], &S)) return 1;
  int64_t n = W.n, nnz = W.nnz;
  int L = tree_depth(n);

  /* P, once (as in omp.c) */
  double *beta = malloc(n * sizeof *beta), *Pc = malloc(nnz * sizeof *Pc);
  double *P = malloc(nnz * sizeof *P);
  int64_t *mirror = malloc(nnz * sizeof *mirror);
  #pragma omp parallel for schedule(dynamic, 256)
  for (int64_t i = 0; i < n; i++) {
    int64_t s = W.colptr[i];
    beta[i] = cond_row(W.val + s, W.colptr[i + 1] - s, S.u, Pc + s);
  }
  build_mirror(&W, mirror);
  #pragma omp parallel for schedule(static)
  for (int64_t k = 0; k < nnz; k++) P[k] = (Pc[k] + Pc[mirror[k]]) / (2.0 * n);

  /* optimizer state */
  double *Y = malloc(2 * n * sizeof *Y), *v = calloc(2 * n, sizeof *v);
  double *gain = malloc(n * sizeof *gain);
  double *Fattr = malloc(2 * n * sizeof *Fattr), *Frep = malloc(2 * n * sizeof *Frep);
  double *grad = malloc(2 * n * sizeof *grad), *zi = malloc(n * sizeof *zi);
  memcpy(Y, S.Y0, 2 * n * sizeof *Y);
  for (int64_t i = 0; i < n; i++) gain[i] = 1;

  double t0 = now();
  for (int it = 0; it <= iters; it++) {
    double Z = gradient(&W, P, Y, n, S.theta, L, Fattr, Frep, zi, grad);
    if (it % 50 == 0)
      printf("iteration %4d  KL = %.4f  (%.1f s)\n", it, kl(&W, P, Y, Z), now() - t0);
    if (it == iters) break;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i++) {
      double gx = grad[2 * i], gy = grad[2 * i + 1];
      double dot = gx * v[2 * i] + gy * v[2 * i + 1];           /* g_i . v_i */
      gain[i] = dot < 0 ? gain[i] + 0.2 : 0.8 * gain[i];
      if (gain[i] < 0.01) gain[i] = 0.01;
      v[2 * i]     = 0.8 * v[2 * i]     - 200 * gain[i] * gx;
      v[2 * i + 1] = 0.8 * v[2 * i + 1] - 200 * gain[i] * gy;
      Y[2 * i] += v[2 * i];
      Y[2 * i + 1] += v[2 * i + 1];
    }
  }

  const char *slash = strrchr(argv[1], '/');
  char path[4096];
  snprintf(path, sizeof path, "results/embed_%s.txt", slash ? slash + 1 : argv[1]);
  FILE *f = fopen(path, "w");
  for (int64_t i = 0; i < n; i++)
    fprintf(f, "%.6g %.6g %ld\n", Y[2 * i], Y[2 * i + 1], (long)W.label[i]);
  fclose(f);
  printf("wrote %s (%d iterations, p = %d)\n", path, iters, omp_get_max_threads());
  return 0;
}
