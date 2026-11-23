#define _POSIX_C_SOURCE 199309L   /* needed for clock_gettime; must come before includes */
#include <math.h>
#include <time.h>
#include "tsne.h"

double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* For one row, at a given beta:
 *   p_j = (w_j/wmax)^beta / sum_k (w_k/wmax)^beta
 *   returns the entropy H = -sum_j p_j log p_j               */
static double row_entropy(const double *w, int64_t d, double wmax,
                          double beta, double *p) {
  double s = 0;
  for (int64_t j = 0; j < d; j++) { p[j] = pow(w[j] / wmax, beta); s += p[j]; }
  double H = 0;
  for (int64_t j = 0; j < d; j++) {
    p[j] /= s;
    if (p[j] > 0) H -= p[j] * log(p[j]);   /* 0·log 0 = 0 */
  }
  return H;
}

double cond_row(const double *w, int64_t d, double u, double *p) {
  double wmin = w[0], wmax = w[0];
  for (int64_t j = 1; j < d; j++) { wmin = fmin(wmin, w[j]); wmax = fmax(wmax, w[j]); }

  /* uniform row: p = 1/d, beta = 0 */
  if (u >= d || wmin == wmax) {
    for (int64_t j = 0; j < d; j++) p[j] = 1.0 / d;
    return 0;
  }

  double target = log(u);          /* here u < d, so min(u, d) = u */
  double lo = 0, hi = 1, beta = 1;

  /* (a) upper bound: H decreases as beta grows, so double hi
   *     until the row at hi is sharp enough (H <= target)      */
  for (int t = 0; t < 60 && row_entropy(w, d, wmax, hi, p) > target; t++)
    hi = 2 * hi;

  /* (b) bisection on [lo, hi]                                  */
  for (int t = 0; t < 100; t++) {
    beta = (lo + hi) / 2;
    double H = row_entropy(w, d, wmax, beta, p);
    if (fabs(H - target) < 1e-5) break;
    if (H > target) lo = beta;     /* row too flat  -> larger beta  */
    else            hi = beta;     /* row too sharp -> smaller beta */
  }

  row_entropy(w, d, wmax, beta, p);   /* p must match the final beta */
  return beta;
}

/* For entry k = (i, j) find the index of entry (j, i).
 * MATLAB stores row indices sorted within each column -> binary search. */
void build_mirror(const graph_t *W, int64_t *mirror) {
  for (int64_t j = 0; j < W->n; j++)
    for (int64_t k = W->colptr[j]; k < W->colptr[j + 1]; k++) {
      int64_t i = W->row[k];                         /* entry (i, j)        */
      int64_t lo = W->colptr[i], hi = W->colptr[i + 1] - 1;
      while (lo < hi) {                              /* find row j in col i */
        int64_t mid = (lo + hi) / 2;
        if (W->row[mid] < j) lo = mid + 1; else hi = mid;
      }
      mirror[k] = lo;
    }
}