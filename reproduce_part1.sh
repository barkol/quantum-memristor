#!/usr/bin/env bash
# Part I: figures and checks. Needs cvxpy and MOSEK for make_fig1_thesis and the sdp_ scripts.
# Usage: ./reproduce_part1.sh [--full]. --full adds the point-process runs (about 20 min on 6 cores).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; OUT="${TMPDIR:-/tmp}/part1_out"; mkdir -p "$OUT"
cd "$HERE/exploratory"
for s in make_fig1_thesis generate_setup_diagram make_fingerprint_grid make_evolution_grid basis_hysteresis sim_kernel_strong make_fq_arch sim_certify_coherent sim_steering_budget; do
  python "$s.py" > "$OUT/$s.out"
done
cd "$HERE/part1"
python validate_nonmarkov.py > "$OUT/validate_nonmarkov.out"
(cd "$OUT" && for s in sdp_budget sdp_addon sdp_physical_assemblage squash c13 validate_locality; do python "$HERE/part1/$s.py" > "$s.out"; done)
python sim_certify_feedback.py > "$OUT/sim_certify_feedback.out"
if [ "$1" = "--full" ]; then
  (cd "$OUT" && python "$HERE/part1/pp_event_driven.py" pp_event_driven.json > pp_event_driven.out
   python "$HERE/part1/validate_extra.py" validate_extra.json > validate_extra.out
   python "$HERE/part1/validate_tail.py" > validate_tail.out; python "$HERE/part1/validate_unipolar.py" > validate_unipolar.out
   python "$HERE/part1/obal_pp.py" > obal_pp.out; python "$HERE/part1/validate_rates.py" > validate_rates.out)
fi
echo "done, outputs in $OUT; compare with results/part1/"
