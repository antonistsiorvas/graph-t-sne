#!/usr/bin/env bash
# Walk time against the quadtree depth L (sequential, one run per L), appended to
# results/depth_sweep_seq.txt in the format plot.py reads.
#
#   bash scripts/depth_sweep.sh coPapersDBLP          (L = 5 .. 12)
set -eu
cd "$(dirname "$0")/.."
make -s seq
g=$1
for L in ${LS:-5 6 7 8 9 10 11 12}; do
  out=$(./seq "data/$g" "$L")
  get() { echo "$out" | grep '^TIME' | grep -o " $1=[0-9.e+-]*" | cut -d= -f2; }
  fe=$(echo "$out" | sed -n 's/.*Frep err = \([0-9.e+-]*\).*/\1/p')
  ze=$(echo "$out" | sed -n 's/.*Z err = \([0-9.e+-]*\).*/\1/p' | head -1)
  printf "%s tree (L = %s): build %s s | mass %s s | walk %s s Frep: max|Frep - ref| / max|ref| = %s | |Z - Zref| / Zref = %s\n" \
    "$g" "$L" "$(get build)" "$(get mass)" "$(get walk)" "$fe" "$ze" | tee -a results/depth_sweep_seq.txt
done
# each run above saved results/<graph>-seq.mat at its own L: run the default depth
# again, so that file keeps the results the report uses
./seq "data/$g" > /dev/null
