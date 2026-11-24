#define _POSIX_C_SOURCE 199309L   /* needed for clock_gettime; must come before includes */
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "tsne.h"

double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

double rel_err(const double *x, const double *ref, int64_t m) {
  double e = 0, r = 0;
  for (int64_t k = 0; k < m; k++) {
    e = fmax(e, fabs(x[k] - ref[k]));
    r = fmax(r, fabs(ref[k]));
  }
  return e / r;
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

/* Attraction on point i: one row of (D - A) Y^T, with
 *   A_ij = P_ij / (1 + |y_i - y_j|^2) computed on the fly.
 *   Fattr_i = sum_j A_ij (y_i - y_j)                          */
void attr_row(const graph_t *W, const double *P, const double *Y,
              int64_t i, double *Fattr) {
  double xi = Y[2 * i], yi = Y[2 * i + 1];
  double fx = 0, fy = 0;
  for (int64_t k = W->colptr[i]; k < W->colptr[i + 1]; k++) {
    int64_t j = W->row[k];
    double dx = xi - Y[2 * j], dy = yi - Y[2 * j + 1];
    double A = P[k] / (1 + dx * dx + dy * dy);
    fx += A * dx;
    fy += A * dy;
  }
  Fattr[2 * i] = fx;
  Fattr[2 * i + 1] = fy;
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

/* ---------------------------------------------------------------- */
/* Quadtree                                                          */

int tree_depth(int64_t n) {
  int L = (int)lround(log(n / 8.0) / log(4.0));
  if (L < 1) L = 1;
  if (L > MAXL) L = MAXL;
  return L;
}

void tree_build(quadtree_t *T, const double *Y, int64_t n, int L) {
  /* bounding square of all points */
  double xmin = Y[0], xmax = Y[0], ymin = Y[1], ymax = Y[1];
  for (int64_t i = 1; i < n; i++) {
    xmin = fmin(xmin, Y[2 * i]);     xmax = fmax(xmax, Y[2 * i]);
    ymin = fmin(ymin, Y[2 * i + 1]); ymax = fmax(ymax, Y[2 * i + 1]);
  }
  T->L = L; T->x0 = xmin; T->y0 = ymin;
  T->side = fmax(xmax - xmin, ymax - ymin);

  /* one array per level */
  for (int l = 0; l <= L; l++) {
    int64_t cells = (int64_t)1 << (2 * l);              /* 4^l */
    T->cnt[l]  = malloc(cells * sizeof(int64_t));
    T->sumx[l] = malloc(cells * sizeof(double));
    T->sumy[l] = malloc(cells * sizeof(double));
  }

  /* counting sort of the points by leaf */
  int64_t side = (int64_t)1 << L, leaves = side * side;
  double  h = T->side / side;                           /* leaf side */
  int64_t *leaf = malloc(n * sizeof *leaf);
  T->start = calloc(leaves + 1, sizeof(int64_t));
  T->perm  = malloc(n * sizeof(int64_t));

  for (int64_t i = 0; i < n; i++) {                     /* 1. leaf of each point */
    int64_t cx = (int64_t)((Y[2 * i]     - T->x0) / h);
    int64_t cy = (int64_t)((Y[2 * i + 1] - T->y0) / h);
    if (cx > side - 1) cx = side - 1;                   /* points on the max edge */
    if (cy > side - 1) cy = side - 1;
    leaf[i] = cy * side + cx;
    T->start[leaf[i] + 1]++;                            /* 2. count */
  }
  for (int64_t c = 0; c < leaves; c++)                  /* 3. prefix sum */
    T->start[c + 1] += T->start[c];
  int64_t *next = malloc(leaves * sizeof *next);        /* 4. scatter */
  for (int64_t c = 0; c < leaves; c++) next[c] = T->start[c];
  for (int64_t i = 0; i < n; i++) T->perm[next[leaf[i]]++] = i;
  free(next); free(leaf);
}

void cell_mass(quadtree_t *T, const double *Y, int l, int64_t c) {
  int64_t N = 0; double sx = 0, sy = 0;
  if (l == T->L) {                                      /* leaf: its points */
    for (int64_t k = T->start[c]; k < T->start[c + 1]; k++) {
      int64_t j = T->perm[k];
      N++; sx += Y[2 * j]; sy += Y[2 * j + 1];
    }
  } else {                                              /* 4 children */
    int64_t side = (int64_t)1 << l, cx = c % side, cy = c / side;
    for (int b = 0; b < 2; b++)
      for (int a = 0; a < 2; a++) {
        int64_t ch = (2 * cy + b) * (2 * side) + (2 * cx + a);
        N  += T->cnt[l + 1][ch];
        sx += T->sumx[l + 1][ch];
        sy += T->sumy[l + 1][ch];
      }
  }
  T->cnt[l][c] = N; T->sumx[l][c] = sx; T->sumy[l][c] = sy;
}

/* Visit cell (cx, cy) at level l for point i; add to fx, fy, z. */
static void visit(const quadtree_t *T, const double *Y, int64_t i, double theta,
                  int l, int64_t cx, int64_t cy,
                  double *fx, double *fy, double *z) {
  int64_t side = (int64_t)1 << l, c = cy * side + cx;
  int64_t N = T->cnt[l][c];
  if (N == 0) return;                                   /* empty cell */

  double xi = Y[2 * i], yi = Y[2 * i + 1];
  double dx = xi - T->sumx[l][c] / N;                   /* to center of mass */
  double dy = yi - T->sumy[l][c] / N;
  double d2 = dx * dx + dy * dy;
  double r  = sqrt(2.0) * T->side / side;               /* cell DIAGONAL */

  if (r < theta * sqrt(d2)) {                           /* far: one body of N points */
    double q = 1 / (1 + d2);
    *z  += N * q;
    *fx += N * q * q * dx;
    *fy += N * q * q * dy;
  } else if (l == T->L) {                               /* near leaf: point by point */
    for (int64_t k = T->start[c]; k < T->start[c + 1]; k++) {
      int64_t j = T->perm[k];
      if (j == i) continue;                             /* no self-interaction */
      double ex = xi - Y[2 * j], ey = yi - Y[2 * j + 1];
      double q = 1 / (1 + ex * ex + ey * ey);
      *z  += q;
      *fx += q * q * ex;
      *fy += q * q * ey;
    }
  } else {                                              /* near: open, visit children */
    for (int b = 0; b < 2; b++)
      for (int a = 0; a < 2; a++)
        visit(T, Y, i, theta, l + 1, 2 * cx + a, 2 * cy + b, fx, fy, z);
  }
}

double rep_point(const quadtree_t *T, const double *Y, int64_t i,
                 double theta, double *Frep) {
  double fx = 0, fy = 0, z = 0;
  visit(T, Y, i, theta, 0, 0, 0, &fx, &fy, &z);         /* start at the root */
  Frep[2 * i] = fx; Frep[2 * i + 1] = fy;
  return z;
}

void tree_free(quadtree_t *T) {
  for (int l = 0; l <= T->L; l++) { free(T->cnt[l]); free(T->sumx[l]); free(T->sumy[l]); }
  free(T->start); free(T->perm);
}
