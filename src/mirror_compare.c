/* mirror_compare.c -- the mirror table two ways: the first version (a binary
 * search per entry, kept here only for this comparison) against build_mirror
 * in tsne.c (one counter per column). Times both, REPS runs each, and checks
 * that the two tables are identical.
 *
 *   ./mirror_compare data/<graph>                                          */
#include <stdio.h>
#include <stdlib.h>
#include "tsne.h"

enum { REPS = 5 };

/* first version: for entry (i, j) in column j, binary search for row j in
 * column i (row indices are sorted within each column). Independent per entry,
 * so it could run in parallel; O(nnz log d_max).                          */
static void mirror_binary_search(const graph_t *W, int64_t *mirror) {
  for (int64_t j = 0; j < W->n; j++)
    for (int64_t k = W->colptr[j]; k < W->colptr[j + 1]; k++) {
      int64_t i = W->row[k];
      int64_t lo = W->colptr[i], hi = W->colptr[i + 1] - 1;
      while (lo < hi) {
        int64_t mid = (lo + hi) / 2;
        if (W->row[mid] < j) lo = mid + 1; else hi = mid;
      }
      mirror[k] = lo;
    }
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s data/<graph>\n", argv[0]); return 1; }
  graph_t W;
  if (read_graph(argv[1], &W)) return 1;
  int64_t *a = malloc(W.nnz * sizeof *a), *b = malloc(W.nnz * sizeof *b);
  double tb = 1e30, tc = 1e30;                    /* best of REPS runs each */
  for (int r = 0; r < REPS; r++) {
    double t0 = now();
    mirror_binary_search(&W, a);
    double t = now() - t0;
    if (t < tb) tb = t;
    t0 = now();
    build_mirror(&W, b);
    t = now() - t0;
    if (t < tc) tc = t;
  }
  int64_t diff = 0;
  for (int64_t k = 0; k < W.nnz; k++) diff += a[k] != b[k];
  printf("%s nnz=%ld: binary search %.4f s | counter %.4f s | %.1fx faster | "
         "entries that differ: %ld (best of %d runs)\n",
         argv[1], (long)W.nnz, tb, tc, tb / tc, (long)diff, REPS);
  return 0;
}
