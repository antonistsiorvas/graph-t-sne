#!/usr/bin/env bash
# Run every version for several thread counts, R times each, and append the
# TIME lines to results/timings.txt (one line per run, all phases).
#
#   bash scripts/bench.sh data/k49_norm_10NN data/com-Amazon
#
# For k49_norm_10NN the direct O(n^2) repulsion is also timed.
# The repetition loop is the outermost, so a slow period of the machine
# (another program, thermal throttling) spreads over all cases instead of
# hitting one.
set -eu
cd "$(dirname "$0")/.."
make -s all

R=${R:-5}                    # repetitions
PS=${PS:-"1 2 4 8 12"}       # thread counts
VERSIONS=${VERSIONS:-"seq omp cilk pth"}   # e.g. VERSIONS=omp to rerun one

# OpenMP threads sleep between parallel loops instead of spinning (gcc's
# default is to spin for a while). Spinning threads competed with the
# sequential parts (the mirror table: 11 ms passive, 19-66 ms spinning, on
# com-Amazon with 12 threads; results/omp_wait_policy.txt).
export OMP_WAIT_POLICY=passive
OUT=results/timings.txt
[ $# -gt 0 ] || { echo "usage: bash scripts/bench.sh data/<graph> ..."; exit 1; }

echo "# $(date -Iseconds) R=$R PS=\"$PS\" VERSIONS=\"$VERSIONS\" OMP_WAIT_POLICY=passive graphs: $*" >> "$OUT"
for r in $(seq "$R"); do
  for g in "$@"; do
    extra=""
    [ "$(basename "$g")" = k49_norm_10NN ] && extra="0 direct"
    echo "run $r/$R: $g" >&2
    for v in $VERSIONS; do
      if [ "$v" = seq ]; then ./seq "$g" $extra | grep '^TIME' >> "$OUT"; continue; fi
      for p in $PS; do
        OMP_NUM_THREADS=$p CILK_NWORKERS=$p PTH_NUM_THREADS=$p \
          ./"$v" "$g" $extra | grep '^TIME' >> "$OUT"
      done
    done
  done
done
echo "done: $(grep -c '^TIME' "$OUT") TIME lines in $OUT" >&2
