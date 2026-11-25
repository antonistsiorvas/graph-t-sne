/* driver.c -- see driver.h. */
#include "driver.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int run_init(run_t *r, int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s data/<graph> [L] [direct]\n", argv[0]);
    return -1;
  }
  if (read_graph(argv[1], &r->W) || read_state(argv[1], &r->S) || read_ref(argv[1], &r->R))
    return -1;
  r->n = r->W.n; r->nnz = r->W.nnz;
  r->L = argc > 2 ? atoi(argv[2]) : 0;
  if (r->L == 0) r->L = tree_depth(r->n);
  r->direct = argc > 3 && strcmp(argv[3], "direct") == 0;
  const char *slash = strrchr(argv[1], '/');
  r->name = slash ? slash + 1 : argv[1];

  int64_t n = r->n, nnz = r->nnz;
  r->beta   = malloc(n * sizeof(double));
  r->Pc     = malloc(nnz * sizeof(double));
  r->P      = malloc(nnz * sizeof(double));
  r->mirror = malloc(nnz * sizeof(int64_t));
  r->Fattr  = malloc(2 * n * sizeof(double));
  r->Frep   = malloc(2 * n * sizeof(double));
  r->grad   = malloc(2 * n * sizeof(double));
  r->zi     = malloc(n * sizeof(double));
  r->Fdir   = r->direct ? malloc(2 * n * sizeof(double)) : NULL;
  r->Z = r->Zdir = 0;
  return 0;
}

void run_finish(run_t *r, const char *version, int p, const times_t *t) {
  int64_t n = r->n, nnz = r->nnz;

  /* against the course reference */
  double eb = 0, sum = 0;
  for (int64_t i = 0; i < n; i++) eb = fmax(eb, fabs(r->beta[i] - r->R.beta[i]));
  for (int64_t k = 0; k < nnz; k++) sum += r->P[k];
  printf("%s, %s, p = %d, L = %d\n", version, r->name, p, r->L);
  printf("  P      %.4f s (mirror %.4f s) | max|beta - ref| = %.2e | sum P = %.15f\n",
         t->P, t->mirror, eb, sum);
  printf("  attr   %.4f s | Fattr err = %.2e\n", t->attr, rel_err(r->Fattr, r->R.Fattr, 2 * n));
  printf("  build  %.4f s | mass %.4f s | walk %.4f s | Frep err = %.2e | Z err = %.2e\n",
         t->build, t->mass, t->walk, rel_err(r->Frep, r->R.Frep, 2 * n),
         fabs(r->Z - r->R.Z) / r->R.Z);
  printf("  grad   %.4f s\n", t->grad);
  if (r->direct)
    printf("  direct %.4f s | Frep err = %.2e | Z err = %.2e | Barnes-Hut is %.1fx faster\n",
           t->direct, rel_err(r->Fdir, r->R.Frep, 2 * n), fabs(r->Zdir - r->R.Z) / r->R.Z,
           t->direct / (t->build + t->mass + t->walk));

  /* one line for the timing scripts */
  printf("TIME version=%s graph=%s n=%ld nnz=%ld p=%d L=%d P=%.6f mirror=%.6f attr=%.6f"
         " build=%.6f mass=%.6f walk=%.6f grad=%.6f direct=%.6f\n",
         version, r->name, (long)n, (long)nnz, p, r->L, t->P, t->mirror, t->attr,
         t->build, t->mass, t->walk, t->grad, r->direct ? t->direct : 0.0);

  /* save, and compare with the sequential version if it has been run */
  char path[4096];
  snprintf(path, sizeof path, "results/%s-%s.mat", r->name, version);
  if (save_results(path, n, r->beta, r->Fattr, r->Frep, r->Z, r->grad) == 0)
    printf("  saved %s\n", path);
  if (strcmp(version, "seq") != 0) {
    snprintf(path, sizeof path, "results/%s-seq.mat", r->name);
    double *b = malloc(n * sizeof *b), *fa = malloc(2 * n * sizeof *fa),
           *fr = malloc(2 * n * sizeof *fr), *g = malloc(2 * n * sizeof *g), Z;
    if (read_results(path, n, b, fa, fr, &Z, g) == 0)
      printf("  vs seq: beta %.1e | Fattr %.1e | Frep %.1e | Z %.1e | grad %.1e\n",
             rel_err(r->beta, b, n), rel_err(r->Fattr, fa, 2 * n),
             rel_err(r->Frep, fr, 2 * n), fabs(r->Z - Z) / Z, rel_err(r->grad, g, 2 * n));
    else
      printf("  vs seq: %s not found, run ./seq first\n", path);
    free(b); free(fa); free(fr); free(g);
  }
}
