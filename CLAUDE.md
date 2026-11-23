# PDS Homework 1: graph t-SNE gradient with Barnes–Hut

The assignment is README.md. I must explain every line in an oral exam.

## Rules
- Keep the code as simple as possible. Prefer clear loops over clever tricks.
  No optimizations unless a measurement shows they matter.
- Explain every change you make, and why it is the simple choice.
- Do not run data/coPapersDBLP: I must predict its timings first.
- Use the variable names from README.md: beta, P, Fattr, Frep, Z, theta, u.

## Design
- src/tsne.c: all the math, one function per row or point (cond_row, attr_row,
  rep_point, quadtree build). Shared by every version.
- src/seq.c, omp.c, cilk.c, pth.c: drivers that only differ in how loops are split.
- Quadtree: fixed depth L, levels stored as plain arrays, points counting-sorted
  into leaves, centers of mass summed from 4 children. Walk = short recursion.
  Cell criterion uses the cell DIAGONAL; skip the point itself at leaves.
- Pthreads: one parallel_for helper with a shared atomic counter; per-thread
  partial sums for Z.
- data.c/data.h are the course starter reader, unchanged.

## Checks
- beta and Fattr match the -ref.mat values (~1e-4 for k49, round-off otherwise).
- Frep max-relative error and Z relative error ~1e-2 or less at theta = 0.5.
- All four versions agree.
