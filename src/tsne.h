#ifndef TSNE_H
#define TSNE_H
#include <stdint.h>
#include "data.h"

/* Row i of W: d neighbors with weights w[0..d-1].
 * Writes p[0..d-1] = p_{j|i} and returns beta_i (0 if uniform). */
double cond_row(const double *w, int64_t d, double u, double *p);

/* mirror[k] = index of entry (j,i) for entry k = (i,j).
 * One counter per column, O(nnz); sequential (see tsne.c). */
void build_mirror(const graph_t *W, int64_t *mirror);

/* Fattr[2i], Fattr[2i+1] = attraction on point i (row i of (D - A) Y^T). */
void attr_row(const graph_t *W, const double *P, const double *Y,
              int64_t i, double *Fattr);

/* Fixed-depth quadtree. The bounding square [x0, x0+side] x [y0, y0+side]
 * is cut into a 2^L x 2^L grid of leaves; every level l = 0..L is kept.
 * At level l, cell (cx, cy) has index c = cy * 2^l + cx.
 * Its children are (2cx + a, 2cy + b), a, b in {0, 1}, at level l+1.  */
#define MAXL 15
typedef struct {
  int L;
  double x0, y0, side;
  int64_t *start;            /* 4^L + 1: points of leaf c are          */
  int64_t *perm;             /* perm[start[c] .. start[c+1]-1]         */
  int64_t *leaf;             /* n: leaf of each point                  */
  int64_t *next;             /* 4^L: scratch for the scatter           */
  int64_t *cnt[MAXL + 1];    /* cnt[l][c]  = number of points in cell  */
  double  *sumx[MAXL + 1];   /* sumx[l][c] = sum of their x            */
  double  *sumy[MAXL + 1];   /* center of mass = (sumx, sumy) / cnt    */
} quadtree_t;

/* Default depth: about 8 points per leaf, L = round(log4(n / 8)).
 * Measured best on k49 (L = 6) and com-Amazon (L = 8). */
int tree_depth(int64_t n);

/* Building the tree is a counting sort of the points by leaf, written out
 * in each driver so each version can split its loops:
 *   tree_init                          bounding square, arrays, start = 0
 *   leaf[i] = point_leaf(i)            for every point
 *   start[leaf[i] + 1]++               count
 *   start[c+1] += start[c]             prefix sum (sequential)
 *   next[c] = start[c]
 *   perm[next[leaf[i]]++] = i          scatter                           */
void    tree_init(quadtree_t *T, const double *Y, int64_t n, int L);
int64_t point_leaf(const quadtree_t *T, const double *Y, int64_t i);

/* cnt, sumx, sumy of cell c at level l: from its points if l == L,
 * else from its 4 children (so level l+1 must be done first). */
void cell_mass(quadtree_t *T, const double *Y, int l, int64_t c);

/* Barnes-Hut walk for point i: writes the UNnormalized repulsion
 * sum_j (1+d^2)^-2 (y_i - y_j) to Frep[2i], Frep[2i+1] and returns
 * this point's share of Z, sum_j (1+d^2)^-1. Divide Frep by Z later. */
double rep_point(const quadtree_t *T, const double *Y, int64_t i,
                 double theta, double *Frep);

void tree_free(quadtree_t *T);

/* Direct O(n^2) repulsion for point i: same outputs as rep_point,
 * summed over all j != i. */
double rep_direct_point(const double *Y, int64_t n, int64_t i, double *Frep);

/* max_i |x_i - ref_i| / max_i |ref_i| over m values. */
double rel_err(const double *x, const double *ref, int64_t m);

/* Write beta (n x 1), Fattr, Frep, grad (2 x n) and Z to a MAT v7.3 file,
 * same variable names as <graph>-ref.mat (save.c). */
int save_results(const char *path, int64_t n, double *beta, double *Fattr,
                 double *Frep, double Z, double *grad);

/* Read back a file written by save_results (arrays of the sizes above). */
int read_results(const char *path, int64_t n, double *beta, double *Fattr,
                 double *Frep, double *Z, double *grad);

double now(void);   /* wall-clock time in seconds */
#endif