# Sequential implementation

All the computation lives in `src/tsne.c` as one function per row, point, or cell
(`cond_row`, `attr_row`, `cell_mass`, `rep_point`), shared by every version; the drivers
(`seq.c`, and later `omp.c`, `cilk.c`, `pth.c`) differ only in how they split the loops
over these functions. $W$ is kept in the CSC form given by the course reader: since $W$
is symmetric, column $i$ is row $i$, so the neighbors of node $i$ are
`row[colptr[i] .. colptr[i+1]-1]` with weights at the same indices in `val`. $P$ is stored
as a second value array over the same pattern, so $p_{ij}$ sits at the same index as
$w_{ij}$. The embedding $Y$ and the forces are arrays of $2n$ doubles, point $i$ at
`[2i], [2i+1]`.

**Step 1: $P$.** For each row, `cond_row` returns the uniform row ($\beta_i = 0$) if
$u \ge d_i$ or all weights are equal. Otherwise it brackets $\beta_i$ by doubling an upper
bound from 1 until $H_i(\beta) \le \log u$ (at most 60 doublings), then bisects until
$|H_i - \log u| < 10^{-5}$ (at most 100 steps); bisection is valid because $H_i$
decreases monotonically in $\beta$, from $\log d_i$ at $\beta = 0$. Weights are used as
$(w_{ij}/\max_k w_{ik})^{\beta}$ to avoid overflow. To symmetrize, `build_mirror` finds
for each entry $(i,j)$ the index `mirror[k]` of its partner $(j,i)$, which lies in
column $i$. Row indices are sorted within each column, so if the columns $j$ are
visited in increasing order, the entries pointing into column $i$ arrive in the same
order as column $i$'s own entries; with one counter per column, `next[i]` (the first
unmatched slot of column $i$), the partner of the current entry is simply
`next[i]++`. This is the standard CSC transpose, $O(\mathrm{nnz})$, with no search.
Then $p_{ij} = (p_{j|i} + p_{i|j})/2n$ is one pass over the nonzeros. Cost:
$O(\mathrm{nnz} \cdot s)$ for $s$ entropy evaluations per row, plus $O(\mathrm{nnz})$.
I first found the partner by binary search in column $i$, $O(\mathrm{nnz}\log d_{\max})$;
the counter gives the identical table and is 5–7× faster (k49: 16 → 2.5 ms;
com-Amazon: 71 → 12 ms; `results/mirror_compare.txt`). The price is that it is inherently sequential: `next[]` carries
state from column to column, so the loop cannot be split among threads, while every
binary search was independent. It remains the serial part of the $P$ phase in the
parallel versions (Section X).

**Step 2: $F^{\mathrm{attr}}$.** `attr_row` computes row $i$ of $(D - A)\,Y^{\mathsf T}$
without forming $A$ or $D$: for each neighbor $j$ it computes
$A_{ij} = p_{ij}/(1 + \Vert y_i - y_j \Vert^2)$ on the fly and accumulates
$A_{ij}(y_i - y_j)$, which is the $D$ and the $A$ terms together. Each row writes only its
own output. Cost: $O(\mathrm{nnz})$, a few flops per nonzero and an indirect read of $y_j$.

**Step 3: $F^{\mathrm{rep}}$ and $Z$.** I use a fixed-depth quadtree. The bounding square
of $Y$ (side = the larger of the two ranges) is cut into a $2^L \times 2^L$ grid of leaves,
and all levels $l = 0..L$ are kept as plain arrays of $4^l$ cells (`cnt`, `sumx`, `sumy`).
Cell $(c_x, c_y)$ at level $l$ is at index $c_y 2^l + c_x$, and its children are
$(2c_x + a, 2c_y + b)$, $a, b \in \{0,1\}$; the tree needs no pointers.
*Construction* is a counting sort: compute each point's leaf, count points per leaf,
prefix-sum the counts into `start`, and scatter the point indices into `perm`, so leaf
$c$ holds `perm[start[c] .. start[c+1]-1]`. Cost $O(n + 4^L)$.
*Centers of mass*: `cell_mass` sums the coordinates of a leaf's points, or adds the
counts and coordinate sums of the four children; levels are processed from $L$ up to 0.
Sums rather than averages are stored because they add; the center is `sum / cnt`.
Cost $O(n + 4^L)$.
*Walk*: for each point, `rep_point` recurses from the root. Empty cells are skipped. If
$r_{\mathrm{cell}} < \theta \Vert y_i - y_{\mathrm{cell}} \Vert$, with $r_{\mathrm{cell}}$ the
cell **diagonal**, the cell contributes $N_{\mathrm{cell}}q^2(y_i - y_{\mathrm{cell}})$ to the
unnormalized force and $N_{\mathrm{cell}} q$ to $Z$, with $q = (1 + d^2)^{-1}$. Otherwise a
leaf adds its points one by one, skipping $j = i$, and an internal cell visits its four
children. A cell that contains $y_i$ is never accepted as a single body: $y_i$ is then
within one diagonal of the center of mass, while acceptance at $\theta = 0.5$ requires a
distance of more than two diagonals. Each walk returns its share of $Z$; after all walks,
the shares are summed and every force is divided by $Z$. The recursion depth is at most
$L + 1$.

**Step 4.** $\nabla C = 4(F^{\mathrm{attr}} - F^{\mathrm{rep}})$, one pass over $2n$ values.
$\beta$, $F^{\mathrm{attr}}$, $F^{\mathrm{rep}}$, $Z$ and the gradient are written to
`results/<graph>-seq.mat` (MAT v7.3, the variable names of `<graph>-ref.mat`).

**Choice of $L$.** Table 1 shows the walk time against $L$. Shallow trees have large
leaves and the walk spends its time in direct sums (com-Amazon: 29 s at $L = 5$); deep
trees make the walk visit many small cells, and the $4^L$ arrays of the deepest level,
mostly empty, make construction and centers of mass grow (about 1 s at $L = 12$). The
best depth gives about 9 points per leaf on average for k49 and about 5 for com-Amazon,
so I set by default $L = \mathrm{round}(\log_4(n/8))$, about 8 points per leaf. It picks
the measured best depth on both graphs ($L = 6$ and $L = 8$) and depends only on $n$, so it
is not tuned to these graphs. The accuracy does not depend on $L$
($F^{\mathrm{rep}}$ error 7–9·10⁻³ for every $L$ from 5 to 12).

*Table 1. Walk time (s) against depth $L$, one run each (`results/depth_sweep_seq.txt`).*

| $L$                 | 5    | 6        | 7    | 8        | 9    | 10   | 11   | 12   |
|---------------------|------|----------|------|----------|------|------|------|------|
| k49_norm_10NN       | 0.35 | **0.26** | 0.29 | 0.37     | 0.43 | 0.51 | 0.54 | 0.94 |
| com-Amazon          | 29.2 | 7.7      | 3.9  | **3.6**  | 5.0  | 5.9  | 7.4  | 7.5  |

**Work per point.** To see what the tree saves, `src/count_terms.c` repeats the walk
but, instead of forces, counts for each point the cells accepted as one body and the
points added one by one at near leaves, and checks that these terms cover every other
point exactly once (Table 1b). Each point computes about 570 terms instead of $n - 1$:
68× fewer on k49 and 570× fewer on com-Amazon. The count barely grows with $n$
(566 → 584 for 8.7× the points), as expected from the roughly logarithmic cost of
Barnes–Hut. The walk time therefore grows almost linearly with $n$; the remaining
difference is in the time per term, 6.5 ns on k49 against 14.4 ns on com-Amazon
(walk time of Table 3 divided by $n \times$ terms per point), which is
consistent with cache effects (the coordinates of com-Amazon, 5.4 MB, do not fit in the
1.25 MB L2 per core; those of k49, 0.6 MB, do), not yet confirmed by measurement.

*Table 1b. Terms per point in the walk at the default $L$, $\theta = 0.5$
(`results/terms_per_point.txt`).*

| graph         | $L$ | far cells | near points | terms (max)  | $n - 1$ | covered exactly once |
|---------------|-----|-----------|-------------|--------------|---------|----------------------|
| k49_norm_10NN | 6   | 219       | 347         | 566 (848)    | 38,546  | all points           |
| com-Amazon    | 8   | 375       | 210         | 584 (1,579)  | 334,862 | all points           |

**Correctness.** Table 2 compares with the course's reference. $\beta$ and
$F^{\mathrm{attr}}$ agree to the bisection tolerance on the weighted k49 and to round-off
on the unweighted com-Amazon, where every row is uniform. The Barnes–Hut errors are below
10⁻² at $\theta = 0.5$. I also checked the walk on a 4-node example against the exact
$O(n^2)$ sum, tracing every accepted and opened cell.

*Table 2. Errors against `<graph>-ref.mat`.*

| graph         | $\max\lvert\beta - \beta_{\mathrm{ref}}\rvert$ | $F^{\mathrm{attr}}$ | $F^{\mathrm{rep}}$ | $Z$ |
|---------------|--------|---------|--------|--------|
| k49_norm_10NN | 1.2e-4 | 1.2e-5  | 8.9e-3 | 5.1e-3 |
| com-Amazon    | 0      | 7.9e-16 | 8.1e-3 | 4.9e-3 |

($F$: $\max\lvert x - x_{\mathrm{ref}}\rvert / \max\lvert x_{\mathrm{ref}}\rvert$;
$Z$: $\lvert Z - Z_{\mathrm{ref}}\rvert / Z_{\mathrm{ref}}$.)

**Sequential times.** Table 3 gives the median of 5 runs for each phase.

*Table 3. Sequential time per phase (s), median of 5 runs (min–max)
(`results/seq_timings.txt`).*

| graph ($n$, nnz)                 | $P$                | attraction            | build      | mass       | walk               |
|----------------------------------|--------------------|-----------------------|------------|------------|--------------------|
| k49_norm_10NN (38,547; 618,158)  | 0.203 (0.189–0.212)| 0.002 (0.001–0.002)   | < 0.001    | < 0.001    | 0.141 (0.130–0.147)|
| com-Amazon (334,863; 1,851,744)  | 0.048 (0.037–0.057)| 0.010 (0.010–0.012)   | 0.006      | 0.002      | 2.816 (2.495–3.531)|

The walk dominates on com-Amazon (98% of the time). $P$ is *faster* on com-Amazon than on
k49 although it has 3× the nonzeros: com-Amazon is unweighted, so every row is uniform
and needs no bisection, while 34,098 of the 38,547 k49 rows do, at 18.3 entropy
evaluations per row on average (doubling, bisection and the final recomputation; at most
24), each evaluating `pow` and `log` for every neighbor. On com-Amazon, with no
bisection, the mirror table is a large part of $P$: with the binary search it took 71 of
118 ms, and switching to the counter cut $P$ to 48 ms. Tree construction and centers of
mass are below 1% of the walk.
