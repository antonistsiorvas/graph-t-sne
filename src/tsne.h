#ifndef TSNE_H
#define TSNE_H
#include <stdint.h>
#include "data.h"

/* Row i of W: d neighbors with weights w[0..d-1].
 * Writes p[0..d-1] = p_{j|i} and returns beta_i (0 if uniform). */
double cond_row(const double *w, int64_t d, double u, double *p);

/* mirror[k] = index of entry (j,i) for entry k = (i,j). */
void build_mirror(const graph_t *W, int64_t *mirror);

double now(void);   /* wall-clock time in seconds */
#endif