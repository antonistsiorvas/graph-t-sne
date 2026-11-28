# coPapersDBLP: prediction against measurement (ms, medians of 5 runs)

Prediction: `results/prediction_coPapersDBLP.md` (committed before the run). Measured: `results/timings_median.csv`. Error = measured / predicted - 1.

## seq, p = 1

| phase | predicted | measured | error |
|---|---|---|---|
| P (incl. mirror) | 913 | 574 | -37% |
|   of which mirror | 253 | 151 | -40% |
| attraction | 171 | 81 | -53% |
| tree build | 12 | 10 | -16% |
| centers of mass | 3.3 | 2.6 | -21% |
| walk | 4,192 | 3,275 | -22% |
| gradient | 6.5 | 9.6 | +47% |
| total | 5,298 | 3,952 | -25% |

## omp, p = 12

| phase | predicted | measured | error |
|---|---|---|---|
| P (incl. mirror) | 499 | 261 | -48% |
|   of which mirror | 195 | 114 | -42% |
| attraction | 88 | 26 | -71% |
| tree build | 20 | 9.5 | -52% |
| centers of mass | 11 | 6.6 | -38% |
| walk | 706 | 662 | -6% |
| gradient | 2.6 | 2.7 | +4% |
| total | 1,327 | 968 | -27% |

## cilk, p = 12

| phase | predicted | measured | error |
|---|---|---|---|
| P (incl. mirror) | 525 | 242 | -54% |
|   of which mirror | 237 | 115 | -51% |
| attraction | 83 | 25 | -70% |
| tree build | 15 | 9.6 | -35% |
| centers of mass | 1.4 | 2.0 | +44% |
| walk | 748 | 657 | -12% |
| gradient | 1.3 | 2.7 | +113% |
| total | 1,374 | 938 | -32% |

## pth, p = 12

| phase | predicted | measured | error |
|---|---|---|---|
| P (incl. mirror) | 525 | 278 | -47% |
|   of which mirror | 195 | 105 | -46% |
| attraction | 120 | 24 | -80% |
| tree build | 23 | 9.4 | -58% |
| centers of mass | 12 | 6.3 | -49% |
| walk | 730 | 637 | -13% |
| gradient | 2.7 | 3.5 | +30% |
| total | 1,414 | 957 | -32% |
