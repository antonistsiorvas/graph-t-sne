#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "tsne.h"

int main(int argc, char **argv) {
  if (argc != 2) { fprintf(stderr, "usage: %s data/<graph>\n", argv[0]); return 1; }
  graph_t W; state_t S; ref_t R;
  if (read_graph(argv[1], &W) || read_state(argv[1], &S) || read_ref(argv[1], &R)) return 1;
  int64_t n = W.n, nnz = W.nnz;

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
  return 0;
}