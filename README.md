# Single-photon quantum memristor — simulation code

Code and data accompanying two companion papers on a polarization-encoded, single-photon
quantum memristor. Every figure and the numbers quoted in the two papers are produced by the scripts in this repository; `reproduce_part1.sh` runs Part I.

Authors and contributions: see `AUTHORS` and `CITATION.cff`. License: MIT (see `LICENSE`).
The dependency `qsdl` (<https://github.com/Askartan/qsdl>, used only in the comparison section) is an external
package that is not included in this repository.

## Papers

**Part I — _Memory-kernel dynamics of a photonic memristor and steering in its coherent extension_**
Karol Bartkiewicz, Patrycja Tulewicz, Adam Miranowicz. Companion paper, submitted together with Part II.

Models a classical register driven by fresh photons: the exponential kernel of the mean register, the Nakajima-Zwanzig equation for the record-averaged state with its range checked against an event-driven simulation (`part1/nz/`), the mean-field closure and the validity range checked against the point process, the Debye law of the loop area, and, for a coherent delay-loop extension, a steering test with closed-form loss budgets. Scripts: `part1/obal_pp.py`, `part1/obal_pp_area.py`, `part1/validate_rates.py`, `part1/validate_locality.py` (point process and locality check; `validate_rates.py` writes `results/part1/obal2.out`), `part1/sdp_budget.py`, `part1/sdp_addon.py` (budgets; need cvxpy and MOSEK), `part1/squash.py`, `part1/c13.py` (multi-photon pulses and fake analyzers; numpy and scipy), `part1/pp_event_driven.py` (loop areas quoted in the paper, no time step), `part1/validate_extra.py`, `part1/validate_tail.py` (remaining register checks), `part1/sdp_physical_assemblage.py` (robustness of the assemblage with random outcomes).

**Part II — _Memristive associative memories and superposed inputs: A measure-and-prepare limit and a decoder beyond the network_**
Karol Bartkiewicz, Patrycja Tulewicz, Adam Miranowicz. Companion paper, submitted together with Part I.

Asks what networks of such devices compute: hardware Hebbian learning, retrieval by quantum versus classical annealing (no detectable gain on
classical patterns at these sizes), a palimpsest, an optical-state benchmark, and exact-recovery capacity with three-body
couplings. For a memory of superposed inputs it gives the measure-and-prepare limit 1/2, and a standard coherent decoder that restores the superposition
for bit-flip errors but does not use the network.

## Part I: figures and numbers

| Item | Script |
|---|---|
| Fig. 1 | `exploratory/make_fig1_thesis.py` |
| closed-loop response | `exploratory/sim_kernel_strong.py` |
| setup | `exploratory/generate_setup_diagram.py` |
| fingerprint, evolution | `exploratory/make_fingerprint_grid.py`, `exploratory/make_evolution_grid.py` |
| basis dependence, index 0.97 | `exploratory/basis_hysteresis.py` |
| architecture | `exploratory/make_fq_arch.py` |
| device certification, loss budget | `exploratory/sim_certify_coherent.py`, `exploratory/sim_steering_budget.py`, `part1/sim_certify_feedback.py` |
| register validation | `part1/validate_nonmarkov.py` (writes `figures/memristor_nonmarkov_validation.pdf`, included in Part I as `register_validation.pdf`) |
| Nakajima-Zwanzig equation, Fig. 2 and its numbers (hierarchy, two-time kernel, range) | `part1/nz/make_fig_nz.py` (figure from `results/part1/nz/`), `part1/nz/grid_mc.py` (records, C engine `mc.c`), `part1/nz/grid_cmp.py` (models against records), `part1/nz/ana_grid.py`, `part1/nz/exps.py`, `part1/nz/ana_exps.py`, `part1/nz/final_extra.py`, `part1/nz/volterra.py`, `part1/nz/volt_run.py`, `part1/nz/volt2.py`, `part1/nz/stab.py`, `part1/nz/scan_c.py`, `part1/nz/variants.py`, `part1/nz/sym_check.py` |
| loop-area deviations at the Debye peak | `part1/pp_event_driven.py` |
| RMS, well escape, spread, X and Y bases, kernel at 0.5 rad, Debye range | `part1/validate_extra.py`, `part1/validate_tail.py` |
| unipolar response rate | `part1/validate_unipolar.py`, `part1/validate_rates.py` |
| budgets, Table of budgets | `part1/sdp_budget.py`, `part1/sdp_addon.py`, `part1/sdp_physical_assemblage.py` |
| multi-photon pulses, fake analyzers | `part1/squash.py`, `part1/c13.py` |

The NARMA-10 reservoir (`part2/qrc_benchmark.py`, `part2/sim_qrc_narma.py`, `paper_qrc_gamma.pdf`) was removed from both papers; it stays here for provenance, and its feedback does not help.

## Layout

| Directory | Contents |
|---|---|
| `part1/` | memory kernel, register validation, feedback certification and operando training |
| `part2/` | everything behind Part II: figure scripts, the supporting analyses quoted in the text, and the cross-checks |
| `exploratory/` | earlier and auxiliary scripts: phase-shift models, hysteresis, animations, presentation figures. The figure scripts of Part I live here; the rest is kept for provenance. |
| `papers/figures/` | the figures that go into the two manuscripts |
| `figures/` | presentation figures, and the cached simulation results in `figures/data/*.npz` |
| `results/` | recorded JSON outputs of the supporting analyses |

Nothing imports across `part1/`, `part2/` and `exploratory/`: each directory runs
on its own, which is why `figstyle.py` is present in all three.

## Part II — figure scripts

The four figures of Part II, and what builds each. Figures land in `papers/figures/`; results are cached first in `figures/data/`, so restyling a
figure never re-runs a simulation. All four figures are drawn by `part2/fig_redesign.py`.

| Figure | File | Data from |
|---|---|---|
| 1 | `fig_concept.pdf` | schematic: `part2/fig_concept_data.py` and `part2/fig_concept_render.py` (Blender) write the cached images |
| 2 | `memristor_network.pdf` | drawing only |
| 3 | `paper_qhnm.pdf` | `part2/fixes_compute.py`, with panel (b) from `part2/panel_b_big.py` and panel (c) from `part2/make_paper_figures.py` |
| 4 | `paper_dense_capacity.pdf` | panel (a) `part2/fixes_compute.py`, panel (b) `part2/capacity_code.py` and `part2/capacity_summary.py` |

`part2/sim_qhnm.py` is the underlying Hopfield-network simulation. The qsdl comparison needs commit a110562 of qsdl: later commits change the noise
model and do not reproduce the numbers of the paper.

## Part II — supporting analyses

These produce numbers quoted in the text and in the Appendix rather than figures of
their own. They are listed here because a figure script alone does not reproduce
the paper.

| Script | Question it settles |
|---|---|
| `part2/capacity_code.py` | Hebbian capacity for random patterns and Hadamard codewords, two- and three-body couplings; bisection in `M` (tolerance 1%, limit 16384, 12 seeds; raw runs in `results/capacity_runs/`, summary by `part2/capacity_summary.py`) |
| `part2/capacity_summary.py` | capacities over seeds, power-law exponents with a bootstrap interval over seeds (the numbers quoted in the Codes subsection) |
| `part2/check_mindist.py` | distribution of the minimum Hamming distance of 32 random words at `n = 256` against the Hadamard words (d = 128) |
| `part2/check_nonlinear_code.py` | the syndrome-ancilla decoder on a nonlinear four-word set, on random word sets and on a d = 2 code: fidelity 1 for the same error of weight <= floor((d-1)/2) on both branches, so linearity buys efficiency, not coherence |
| `part2/redraw_panel_b.py` | six independent 60-network redraws of the cell `n = 8, L = 3, f = 0.1875` (biased greedy, simulated annealing, quantum annealing, paired differences with standard errors); the pooled gaps quoted in the text; about 14 s per draw |
| `part2/panel_b_big.py` | regenerates `figures/data/fix_panel_b_big.npz`, the 60-network draw plotted in Fig. 3(b): `python part2/panel_b_big.py --out x.npz` (seed 20260911, about 4 minutes, checkpoint and resume; reproduces the shipped cache bit for bit). The keys `ident` and `exact_g` are the fraction of trials on which simulated annealing and biased greedy return the same configuration, and the fraction of trials on which unbiased greedy recovers the target exactly |
| `part2/check1_capacity.py` | nearest-word criterion at `M = n-1` and at large three-body `M` (`n = 256`); the probe alone passes this criterion, so it does not measure a memory, and with exact recovery the rate at `M = n-1` is 0 (`check6_supporting.py`) |
| `part2/check2_quantum_input.py` | unitary evolution of a corrupted superposition under the network couplings: best fidelity 0.053 in a 20-point scan, no gain (0.051 with the couplings removed, in `check6_supporting.py`) |
| `part2/check4_qhop.py` | the syndrome-ancilla decoder: that the syndrome of a linear code depends only on the error, the resulting capacity and error tolerance, and the CNOT cost of syndrome extraction |
| `part2/scale_fig2b.py` | whether the Fig. 3(b) ordering survives at `n = 12` and `n = 14`, with the unbiased and the biased greedy control side by side |
| `part2/sweep_bias.py` | whether the collapse of the quantum lead at larger `n` is a size effect or an artefact of the input bias `h = 0.6` tuned at `n = 8` |
| `part2/sweep_anneal_tuning.py` | whether the annealer's loss survives re-tuning its own schedule: a 4x4 scan over total time and transverse field against the same classical controls |
| `part2/sweep_noise_bias.py` | joint `(n, f, h)` sweep: does the optimal bias depend on the corruption level; bootstrap versus Gaussian confidence intervals; Trotter-resolution diagnostic |
| `part2/gates_bias.py` | gate-count and bias bookkeeping for the annealed schedule |
| `part2/sweep_anneal_tuning_f0125.py` | the same 4x4 annealer scan at `f = 0.125` on two draws (seeds 20260912, 4242): retuned anneal within 0.004 to 0.010 of biased descent, against 0.046 to 0.051 at the fixed schedule (12, 1); writes `results/part2/sweep_anneal_tuning_f0125.json` |
| `part2/check_L2_control.py` | `n = 8, L = 2, f = 0.125`: unbiased greedy control (0.94) and biased descent at `h = 0.2` (0.98 to 0.99), with the streamed couplings of the paper and with the exact Hebbian matrix |
| `part2/fit_palimpsest.py` | the forgetting rate of Fig. 3(d): fit of ln R over ages 0 to 4 (0.75 per pattern) and its percentile bootstrap interval over the 60 streams ([0.60, 1.07]) |
| `part2/check_trotter.py` | Trotterized propagator at a bias far above the coupling scale (works up to h dt = 3.3, fails at 3.6) |
| `part2/qsdl_overlaps.py` | prototype overlaps of the qsdl codebook (mean 0.254, closest pair 0.996, random patterns 0.025); needs qsdl |
| `part2/fixes_compute.py` | shared compute layer for the above (biased greedy descent, corruption, `figures/data/*.npz` caches) |
| `part2/fixes_figures.py` | plotting layer; reads the caches only, never recomputes |

### Comparison with the qsdl reconstructor (Sec. "Denoising optical quantum states")

| Script | Role |
|---|---|
| `part2/framework.py` | common benchmark: the qsdl state sampler, noise channel, Wigner transform and fidelity convention, so memristive attractor and convolutional reconstructor are scored on identical data |
| `part2/experiment.py` | the memristive Hopfield denoiser itself (fixed random projection, sign binarisation, attractor descent) |
| `part2/superpos.py` | what both methods do to a coherent *superposition* of states whose Wigner functions were the training patterns |

These three need the external package **qsdl** (by F. Grzywa, acknowledged in Part II),
<https://github.com/Askartan/qsdl>, and `qutip`. Clone qsdl next to this
repository, or point `QSDL_PATH` at your checkout:

```bash
git clone https://github.com/Askartan/qsdl.git ../qsdl && git -C ../qsdl checkout a110562     # or:
export QSDL_PATH=/path/to/qsdl
```

The other scripts have no such dependency.

## Reproducibility

The simulations are deterministic: random seeds are fixed inside the scripts, so a
given script reproduces its numbers exactly. No quantum hardware and no cloud
account is needed — everything runs on a laptop with the packages in
`requirements.txt`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python part2/fig_redesign.py       # rebuilds all figures from the shipped caches
./reproduce.sh                    # about a minute: all figures from the caches and the quoted numbers (capacity, NARMA-10, coherent decoder, minimum distance, nonlinear code)
./reproduce.sh --full             # adds the regeneration of the Fig. 3(b) cache (about 4 minutes, compared bit for bit) and the six redraws (about 1.5 minutes)
```

Scripts may be run from any working directory, except `trained_schedule.py` and `trained_schedule_test.py`, which run from `part2/`. The cached results of the expensive runs are shipped in
`figures/data/` and the recorded outputs in `results/`, so every figure can be
rebuilt in seconds; delete a cache file to force its recomputation. The long sweeps (`sweep_bias.py`, `sweep_noise_bias.py`,
`scale_fig2b.py`, `capacity_code.py`) checkpoint after every cell into
`figures/data/`, and resume by skipping completed cells if restarted. The
`check*.py` scripts write their JSON result next to the current working directory.

The Nakajima-Zwanzig checks of Part I (`part1/nz/`) need the C engine compiled once, `gcc -O3 -shared -fPIC -o part1/nz/mc.so part1/nz/mc.c -lm`. The stored records and comparisons are in `results/part1/nz/` (set `NZ_DIR` to write elsewhere), and `python part1/nz/make_fig_nz.py` rebuilds Fig. 2 from them in seconds. Regenerating the 240 grid points with `grid_mc.py` and `grid_cmp.py` took about 25 minutes and under 2 hours on 10 cores; both write one file per point and skip finished points when restarted.

Expected cost on one laptop core: the figure scripts minutes each; the `n = 14`
sweeps hours; `capacity_code.py` at `n = 256` hours.

## Use of AI tools

The AI assistant Claude Code (Anthropic) was used to write and run code, to check the numerical results
against the text and to draft parts of the text, as stated in the papers. The authors reviewed the code
and verified every number and statement.

## Funding

European High Performance Computing Joint Undertaking (EuroHPC JU), Grant Agreement
No. 101194322 (QEC4QEA), co-funded by the Polish National Centre for Research and
Development (NCBiR) under decision No. DWM/EuroHPC/2023/429/2025.

The optical quantum-state dataset, noise model and convolutional reconstructor used
in the comparison section were developed within the project *Deep learning for the
classification and reconstruction of optical quantum states*, carried out under the
Excellence Initiative — Research University (IDUB) programme at Adam Mickiewicz
University.

## Contact

Karol Bartkiewicz — bark@amu.edu.pl
Institute of Spintronics and Quantum Information, Faculty of Physics and Astronomy,
Adam Mickiewicz University, Poznań, Poland

## Note on `results/check6_supporting.jsonl`

These outputs were recorded before a bug fix: `check6_supporting.py` called a non-existent function for the unbiased greedy control, so that control is `null` in the recorded file. The script now calls `retrieve_classical_batch`; the control value quoted in Part II (0.94 at `n = 8, L = 2, f = 0.125`) comes from `part2/check_L2_control.py`.
