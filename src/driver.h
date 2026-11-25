/* driver.h -- what the four drivers (seq, omp, cilk, pth) share:
 * reading the input, the arrays, checking, printing and saving.
 * The drivers themselves only contain the phases and how their loops are split.
 *
 *   usage: ./<version> data/<graph> [L] [direct]
 *     L       quadtree depth; 0 or missing: tree_depth(n)
 *     direct  also compute the repulsion directly, O(n^2)            */
#ifndef DRIVER_H
#define DRIVER_H
#include "tsne.h"

typedef struct {
  graph_t W; state_t S; ref_t R;
  int64_t n, nnz;
  int L, direct;
  const char *name;            /* "data/k49_norm_10NN" -> "k49_norm_10NN" */
  double  *beta;               /* n                                      */
  double  *Pc, *P;             /* nnz: conditional p_{j|i}, symmetric p_ij */
  int64_t *mirror;             /* nnz                                    */
  double  *Fattr, *Frep, *grad;/* 2n                                     */
  double  *zi;                 /* n: each point's share of Z             */
  double   Z;
  double  *Fdir, Zdir;         /* direct repulsion (if direct)           */
} run_t;

typedef struct {               /* seconds per phase */
  double P, mirror, attr, build, mass, walk, grad, direct;
} times_t;

int  run_init(run_t *r, int argc, char **argv);
void run_finish(run_t *r, const char *version, int p, const times_t *t);

#endif
