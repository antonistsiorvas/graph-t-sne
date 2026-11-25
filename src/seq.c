/* seq.c -- the sequential version: every loop runs on one thread.
 * The other versions are this file with the loops split among threads. */
#include "driver.h"

int main(int argc, char **argv) {
  run_t r;
  if (run_init(&r, argc, argv)) return 1;
  const graph_t *W = &r.W;
  const double  *Y = r.S.Y0;
  int64_t n = r.n, nnz = r.nnz;
  times_t t = {0};
  double t0, t1;

  /* step 1: P */
  t0 = now();
  for (int64_t i = 0; i < n; i++) {                       /* one row at a time */
    int64_t s = W->colptr[i], d = W->colptr[i + 1] - s;
    r.beta[i] = cond_row(W->val + s, d, r.S.u, r.Pc + s);
  }
  t1 = now();
  build_mirror(W, r.mirror);
  t.mirror = now() - t1;
  for (int64_t k = 0; k < nnz; k++)                       /* symmetrize */
    r.P[k] = (r.Pc[k] + r.Pc[r.mirror[k]]) / (2.0 * n);
  t.P = now() - t0;

  /* step 2: attraction */
  t0 = now();
  for (int64_t i = 0; i < n; i++) attr_row(W, r.P, Y, i, r.Fattr);
  t.attr = now() - t0;

  /* step 3a: build the quadtree (counting sort of the points by leaf) */
  quadtree_t T;
  int64_t leaves = (int64_t)1 << (2 * r.L);
  t0 = now();
  tree_init(&T, Y, n, r.L);
  for (int64_t i = 0; i < n; i++) {                       /* leaf, count */
    T.leaf[i] = point_leaf(&T, Y, i);
    T.start[T.leaf[i] + 1]++;
  }
  for (int64_t c = 0; c < leaves; c++) T.start[c + 1] += T.start[c];   /* prefix sum */
  for (int64_t c = 0; c < leaves; c++) T.next[c] = T.start[c];
  for (int64_t i = 0; i < n; i++) T.perm[T.next[T.leaf[i]]++] = i;      /* scatter */
  t.build = now() - t0;

  /* step 3b: centers of mass, leaves first, then up */
  t0 = now();
  int64_t cells = leaves;                                 /* 4^l cells at level l */
  for (int l = r.L; l >= 0; l--, cells /= 4)
    for (int64_t c = 0; c < cells; c++) cell_mass(&T, Y, l, c);
  t.mass = now() - t0;

  /* step 3c: the walk, then Z and normalization */
  t0 = now();
  for (int64_t i = 0; i < n; i++) r.zi[i] = rep_point(&T, Y, i, r.S.theta, r.Frep);
  for (int64_t i = 0; i < n; i++) r.Z += r.zi[i];          /* same order in every version */
  for (int64_t k = 0; k < 2 * n; k++) r.Frep[k] /= r.Z;
  t.walk = now() - t0;
  tree_free(&T);

  /* step 4: the gradient */
  t0 = now();
  for (int64_t k = 0; k < 2 * n; k++) r.grad[k] = 4 * (r.Fattr[k] - r.Frep[k]);
  t.grad = now() - t0;

  /* direct O(n^2) repulsion, to compare */
  if (r.direct) {
    t0 = now();
    for (int64_t i = 0; i < n; i++) r.zi[i] = rep_direct_point(Y, n, i, r.Fdir);
    for (int64_t i = 0; i < n; i++) r.Zdir += r.zi[i];
    for (int64_t k = 0; k < 2 * n; k++) r.Fdir[k] /= r.Zdir;
    t.direct = now() - t0;
  }

  run_finish(&r, "seq", 1, &t);
  return 0;
}
