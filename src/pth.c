/* pth.c -- Pthreads version: seq.c with each loop run by parallel_for.
 * Threads: PTH_NUM_THREADS (default: all hardware threads).
 *
 * parallel_for(n, chunk, body) starts p threads; each repeatedly takes the
 * next chunk of iterations [lo, lo + chunk) from a shared atomic counter and
 * runs body(lo, hi), until the counter passes n; then all threads are joined.
 * Taking chunks on demand balances loops whose iterations cost different
 * amounts (like OpenMP's schedule(dynamic)).
 * The loop bodies are small functions over a range [lo, hi); they read the
 * data from the globals below.                                             */
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <unistd.h>
#include "driver.h"

/* ---- parallel_for ---------------------------------------------------- */

typedef void (*body_t)(int64_t lo, int64_t hi);

static int     p;                        /* number of threads            */
static body_t  job_body;                 /* the loop being run           */
static int64_t job_n, job_chunk;
static atomic_int_fast64_t job_next;     /* first iteration not taken    */

static void *worker(void *unused) {
  (void)unused;
  for (;;) {
    int64_t lo = atomic_fetch_add(&job_next, job_chunk);   /* take a chunk */
    if (lo >= job_n) return NULL;                           /* none left    */
    int64_t hi = lo + job_chunk < job_n ? lo + job_chunk : job_n;
    job_body(lo, hi);
  }
}

static void parallel_for(int64_t n, int64_t chunk, body_t body) {
  job_body = body; job_n = n; job_chunk = chunk;
  atomic_store(&job_next, 0);
  pthread_t th[p];
  for (int k = 0; k < p; k++) pthread_create(&th[k], NULL, worker, NULL);
  for (int k = 0; k < p; k++) pthread_join(th[k], NULL);
}

/* ---- data the loop bodies use ---------------------------------------- */

static run_t        r;
static const double *Y;
static quadtree_t   T;
static int          level;               /* level for mass_cells         */
static double       Zsum;                /* Z or Zdir, for normalize     */
static double      *Fnorm;               /* Frep or Fdir, for normalize  */

/* ---- loop bodies: the loops of seq.c over [lo, hi) -------------------- */

static void P_rows(int64_t lo, int64_t hi) {
  for (int64_t i = lo; i < hi; i++) {
    int64_t s = r.W.colptr[i], d = r.W.colptr[i + 1] - s;
    r.beta[i] = cond_row(r.W.val + s, d, r.S.u, r.Pc + s);
  }
}
static void P_symmetrize(int64_t lo, int64_t hi) {
  for (int64_t k = lo; k < hi; k++)
    r.P[k] = (r.Pc[k] + r.Pc[r.mirror[k]]) / (2.0 * r.n);
}
static void attr_rows(int64_t lo, int64_t hi) {
  for (int64_t i = lo; i < hi; i++) attr_row(&r.W, r.P, Y, i, r.Fattr);
}
static void leaf_count(int64_t lo, int64_t hi) {
  for (int64_t i = lo; i < hi; i++) {
    T.leaf[i] = point_leaf(&T, Y, i);
    __atomic_fetch_add(&T.start[T.leaf[i] + 1], 1, __ATOMIC_RELAXED);
  }
}
static void next_copy(int64_t lo, int64_t hi) {
  for (int64_t c = lo; c < hi; c++) T.next[c] = T.start[c];
}
static void scatter(int64_t lo, int64_t hi) {
  for (int64_t i = lo; i < hi; i++) {
    int64_t slot = __atomic_fetch_add(&T.next[T.leaf[i]], 1, __ATOMIC_RELAXED);
    T.perm[slot] = i;
  }
}
static void mass_cells(int64_t lo, int64_t hi) {
  for (int64_t c = lo; c < hi; c++) cell_mass(&T, Y, level, c);
}
static void walk_points(int64_t lo, int64_t hi) {
  for (int64_t i = lo; i < hi; i++) r.zi[i] = rep_point(&T, Y, i, r.S.theta, r.Frep);
}
static void normalize(int64_t lo, int64_t hi) {
  for (int64_t k = lo; k < hi; k++) Fnorm[k] /= Zsum;
}
static void gradient(int64_t lo, int64_t hi) {
  for (int64_t k = lo; k < hi; k++) r.grad[k] = 4 * (r.Fattr[k] - r.Frep[k]);
}
static void direct_points(int64_t lo, int64_t hi) {
  for (int64_t i = lo; i < hi; i++) r.zi[i] = rep_direct_point(Y, r.n, i, r.Fdir);
}

/* ---- the phases, as in seq.c ------------------------------------------ */

enum { ROWS = 256, ELEMS = 4096 };       /* chunk sizes: costly / cheap iterations */

int main(int argc, char **argv) {
  if (run_init(&r, argc, argv)) return 1;
  const char *env = getenv("PTH_NUM_THREADS");
  p = env ? atoi(env) : (int)sysconf(_SC_NPROCESSORS_ONLN);
  if (p < 1) p = 1;
  Y = r.S.Y0;
  int64_t n = r.n, nnz = r.nnz;
  times_t t = {0};
  double t0, t1;

  /* step 1: P */
  t0 = now();
  parallel_for(n, ROWS, P_rows);
  t1 = now();
  build_mirror(&r.W, r.mirror);                           /* sequential */
  t.mirror = now() - t1;
  parallel_for(nnz, ELEMS, P_symmetrize);
  t.P = now() - t0;

  /* step 2: attraction */
  t0 = now();
  parallel_for(n, ROWS, attr_rows);
  t.attr = now() - t0;

  /* step 3a: build the quadtree */
  int64_t leaves = (int64_t)1 << (2 * r.L);
  t0 = now();
  tree_init(&T, Y, n, r.L);
  parallel_for(n, ELEMS, leaf_count);
  for (int64_t c = 0; c < leaves; c++) T.start[c + 1] += T.start[c];   /* sequential */
  parallel_for(leaves, ELEMS, next_copy);
  parallel_for(n, ELEMS, scatter);
  t.build = now() - t0;

  /* step 3b: centers of mass, one level at a time */
  t0 = now();
  int64_t cells = leaves;                                 /* 4^l cells at level l */
  for (level = r.L; level >= 0; level--, cells /= 4)
    parallel_for(cells, ROWS, mass_cells);
  t.mass = now() - t0;

  /* step 3c: the walk, then Z (same order as seq) and normalization */
  t0 = now();
  parallel_for(n, ROWS, walk_points);
  for (int64_t i = 0; i < n; i++) r.Z += r.zi[i];
  Zsum = r.Z; Fnorm = r.Frep;
  parallel_for(2 * n, ELEMS, normalize);
  t.walk = now() - t0;
  tree_free(&T);

  /* step 4: the gradient */
  t0 = now();
  parallel_for(2 * n, ELEMS, gradient);
  t.grad = now() - t0;

  /* direct O(n^2) repulsion */
  if (r.direct) {
    t0 = now();
    parallel_for(n, ROWS, direct_points);
    for (int64_t i = 0; i < n; i++) r.Zdir += r.zi[i];
    Zsum = r.Zdir; Fnorm = r.Fdir;
    parallel_for(2 * n, ELEMS, normalize);
    t.direct = now() - t0;
  }

  run_finish(&r, "pth", p, &t);
  return 0;
}
