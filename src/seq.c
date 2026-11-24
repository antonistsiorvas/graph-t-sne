#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "tsne.h"

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s data/<graph> [L]\n", argv[0]); return 1; }
  int L = argc > 2 ? atoi(argv[2]) : 0;           /* quadtree depth: 2^L x 2^L leaves */
  graph_t W; state_t S; ref_t R;
  if (read_graph(argv[1], &W) || read_state(argv[1], &S) || read_ref(argv[1], &R)) return 1;
  int64_t n = W.n, nnz = W.nnz;
  if (L == 0) L = tree_depth(n);

  double  *beta   = malloc(n * sizeof *beta);
  double  *Pc     = malloc(nnz * sizeof *Pc);     /* conditional p_{j|i} */
  double  *P      = malloc(nnz * sizeof *P);      /* symmetric p_ij      */
  int64_t *mirror = malloc(nnz * sizeof *mirror);

  double t0 = now();
  for (int64_t i = 0; i < n; i++) {               /* one row at a time */
    int64_t s = W.colptr[i], d = W.colptr[i + 1] - s;
    beta[i] = cond_row(W.val + s, d, S.u, Pc + s);
  }
  build_mirror(&W, mirror);
  for (int64_t k = 0; k < nnz; k++)               /* symmetrize */
    P[k] = (Pc[k] + Pc[mirror[k]]) / (2.0 * n);
  double tP = now() - t0;

  double err = 0, sum = 0;
  for (int64_t i = 0; i < n; i++) err = fmax(err, fabs(beta[i] - R.beta[i]));
  for (int64_t k = 0; k < nnz; k++) sum += P[k];
  printf("P: %.3f s | max|beta - ref| = %.2e | sum P = %.15f\n", tP, err, sum);

  double *Fattr = malloc(2 * n * sizeof *Fattr);
  t0 = now();
  for (int64_t i = 0; i < n; i++) attr_row(&W, P, S.Y0, i, Fattr);
  double tA = now() - t0;
  printf("Fattr: %.3f s | max|Fattr - ref| / max|ref| = %.2e\n",
         tA, rel_err(Fattr, R.Fattr, 2 * n));

  /* repulsion: build the tree, centers of mass, walk */
  quadtree_t T;
  t0 = now();
  tree_build(&T, S.Y0, n, L);
  double tB = now() - t0;

  t0 = now();
  for (int l = L; l >= 0; l--)                    /* leaves first, then up */
    for (int64_t c = 0; c < ((int64_t)1 << (2 * l)); c++)
      cell_mass(&T, S.Y0, l, c);
  double tM = now() - t0;

  double *Frep = malloc(2 * n * sizeof *Frep);
  t0 = now();
  double Z = 0;
  for (int64_t i = 0; i < n; i++) Z += rep_point(&T, S.Y0, i, S.theta, Frep);
  for (int64_t k = 0; k < 2 * n; k++) Frep[k] /= Z;   /* normalize */
  double tW = now() - t0;

  printf("tree (L = %d): build %.3f s | mass %.3f s | walk %.3f s\n", L, tB, tM, tW);
  printf("Frep: max|Frep - ref| / max|ref| = %.2e | |Z - Zref| / Zref = %.2e\n",
         rel_err(Frep, R.Frep, 2 * n), fabs(Z - R.Z) / R.Z);
  tree_free(&T);
  return 0;
}