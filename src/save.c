/* save.c -- write our results in the same format as <graph>-ref.mat. */
#include <matio.h>
#include <stdio.h>
#include <string.h>
#include "tsne.h"

/* write one real matrix (rows x cols, column by column) as variable name */
static void write_var(mat_t *f, const char *name, double *x, size_t rows, size_t cols) {
  size_t dims[2] = {rows, cols};
  matvar_t *v = Mat_VarCreate(name, MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, x,
                              MAT_F_DONT_COPY_DATA);   /* matio reads x, does not free it */
  Mat_VarWrite(f, v, MAT_COMPRESSION_ZLIB);
  Mat_VarFree(v);
}

int save_results(const char *path, int64_t n, double *beta, double *Fattr,
                 double *Frep, double Z, double *grad) {
  mat_t *f = Mat_CreateVer(path, NULL, MAT_FT_MAT73);
  if (!f) { fprintf(stderr, "cannot create %s\n", path); return -1; }
  write_var(f, "beta",  beta,  n, 1);
  write_var(f, "Fattr", Fattr, 2, n);
  write_var(f, "Frep",  Frep,  2, n);
  write_var(f, "Z",     &Z,    1, 1);
  write_var(f, "grad",  grad,  2, n);
  Mat_Close(f);
  return 0;
}
