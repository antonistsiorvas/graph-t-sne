/* count_bisection.c -- how much work does the bisection for beta do?
 * Repeats the steps of cond_row (tsne.c) for every row and counts the rows that
 * need the bisection and the entropy evaluations per row (the doubling of the
 * upper bound, the bisection steps, and the final evaluation at beta).
 * A copy, because row_entropy is private to tsne.c; keep the two in step.
 *
 *   ./count_bisection data/<graph>                                         */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "tsne.h"

static long evals;                       /* entropy evaluations so far */

static double row_entropy(const double *w, int64_t d, double wmax, double beta, double *p) {
  evals++;
  double s = 0, H = 0;
  for (int64_t j = 0; j < d; j++) { p[j] = pow(w[j] / wmax, beta); s += p[j]; }
  for (int64_t j = 0; j < d; j++) {
    p[j] /= s;
    if (p[j] > 0) H -= p[j] * log(p[j]);
  }
  return H;
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s data/<graph>\n", argv[0]); return 1; }
  graph_t W;
  state_t S;
  if (read_graph(argv[1], &W) || read_state(argv[1], &S)) return 1;
  double u = S.u;
  double *p = malloc(W.nnz * sizeof *p);
  long rows = 0, max_evals = 0;

  for (int64_t i = 0; i < W.n; i++) {
    const double *w = W.val + W.colptr[i];
    int64_t d = W.colptr[i + 1] - W.colptr[i];
    double wmin = w[0], wmax = w[0];
    for (int64_t j = 1; j < d; j++) { wmin = fmin(wmin, w[j]); wmax = fmax(wmax, w[j]); }
    if (u >= d || wmin == wmax) continue;                  /* uniform row: no bisection */

    long e0 = evals;
    double target = log(u), lo = 0, hi = 1, beta = 1;      /* as in cond_row */
    for (int t = 0; t < 60 && row_entropy(w, d, wmax, hi, p) > target; t++) hi = 2 * hi;
    for (int t = 0; t < 100; t++) {
      beta = (lo + hi) / 2;
      double H = row_entropy(w, d, wmax, beta, p);
      if (fabs(H - target) < 1e-5) break;
      if (H > target) lo = beta; else hi = beta;
    }
    row_entropy(w, d, wmax, beta, p);
    rows++;
    if (evals - e0 > max_evals) max_evals = evals - e0;
  }
  printf("%s: %ld of %ld rows need the bisection (u = %g); entropy evaluations per such row: "
         "mean %.1f, max %ld\n", argv[1], rows, (long)W.n, u,
         rows ? (double)evals / rows : 0.0, max_evals);
  return 0;
}
