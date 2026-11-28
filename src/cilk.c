/* cilk.c -- OpenCilk version: seq.c with each loop as a cilk_for.
 * Workers: CILK_NWORKERS (default: all hardware threads).
 * cilk_for splits the range in halves recursively down to a grain of
 * iterations; idle workers steal halves from busy ones, which balances
 * loops whose iterations cost different amounts.
 * Cilk has no atomic pragma: the counting sort uses the compiler's
 * __atomic_fetch_add (one indivisible read-add-write).                   */
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
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
  cilk_for (int64_t i = 0; i < n; i++) {                  /* rows are independent */
    int64_t s = W->colptr[i], d = W->colptr[i + 1] - s;
    r.beta[i] = cond_row(W->val + s, d, r.S.u, r.Pc + s);
  }
  t1 = now();
  build_mirror(W, r.mirror);                              /* sequential */
  t.mirror = now() - t1;
  cilk_for (int64_t k = 0; k < nnz; k++)
    r.P[k] = (r.Pc[k] + r.Pc[r.mirror[k]]) / (2.0 * n);
  t.P = now() - t0;

  /* step 2: attraction */
  t0 = now();
  cilk_for (int64_t i = 0; i < n; i++) attr_row(W, r.P, Y, i, r.Fattr);
  t.attr = now() - t0;

  /* step 3a: build the quadtree */
  quadtree_t T;
  int64_t leaves = (int64_t)1 << (2 * r.L);
  t0 = now();
  tree_init(&T, Y, n, r.L);
  cilk_for (int64_t i = 0; i < n; i++) {
    T.leaf[i] = point_leaf(&T, Y, i);
    __atomic_fetch_add(&T.start[T.leaf[i] + 1], 1, __ATOMIC_RELAXED);
  }
  for (int64_t c = 0; c < leaves; c++) T.start[c + 1] += T.start[c];   /* sequential */
  cilk_for (int64_t c = 0; c < leaves; c++) T.next[c] = T.start[c];
  cilk_for (int64_t i = 0; i < n; i++) {
    int64_t slot = __atomic_fetch_add(&T.next[T.leaf[i]], 1, __ATOMIC_RELAXED);
    T.perm[slot] = i;
  }
  t.build = now() - t0;

  /* step 3b: centers of mass; the cells of one level are independent */
  t0 = now();
  int64_t cells = leaves;                     /* 4^l, without a shift by 2l: with  */
  for (int l = r.L; l >= 0; l--, cells /= 4)  /* it, cilk_for could not compute the */
    cilk_for (int64_t c = 0; c < cells; c++)  /* trip count and ran sequentially    */
      cell_mass(&T, Y, l, c);
  t.mass = now() - t0;

  /* step 3c: the walk; each point writes only its own Frep and zi */
  t0 = now();
  cilk_for (int64_t i = 0; i < n; i++) r.zi[i] = rep_point(&T, Y, i, r.S.theta, r.Frep);
  for (int64_t i = 0; i < n; i++) r.Z += r.zi[i];          /* same order as seq */
  cilk_for (int64_t k = 0; k < 2 * n; k++) r.Frep[k] /= r.Z;
  t.walk = now() - t0;
  tree_free(&T);

  /* step 4: the gradient */
  t0 = now();
  cilk_for (int64_t k = 0; k < 2 * n; k++) r.grad[k] = 4 * (r.Fattr[k] - r.Frep[k]);
  t.grad = now() - t0;

  /* direct O(n^2) repulsion */
  if (r.direct) {
    t0 = now();
    cilk_for (int64_t i = 0; i < n; i++) r.zi[i] = rep_direct_point(Y, n, i, r.Fdir);
    for (int64_t i = 0; i < n; i++) r.Zdir += r.zi[i];
    cilk_for (int64_t k = 0; k < 2 * n; k++) r.Fdir[k] /= r.Zdir;
    t.direct = now() - t0;
  }

  run_finish(&r, "cilk", __cilkrts_get_nworkers(), &t);
  return 0;
}
