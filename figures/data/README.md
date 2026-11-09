# Cached results

`*.npz` files are the cached outputs of the expensive simulations. The compute
layer (`part2/fixes_compute.py` and the sweep scripts) writes them; the plotting
layer (`part2/fixes_figures.py`) only reads them, so a figure can be restyled
without re-running anything. Delete a file to force its recomputation.

| Cache | Written by | Feeds |
|---|---|---|
| `fix_panel_a_shotnoise.npz` | `fixes_compute.py` | Fig. 3(a), Poisson-sampled coincidence budget |
| `fix_panel_b_big.npz`, `fix_panel_b_disorder.npz` | `part2/panel_b_big.py` (`fix_panel_b_big.npz`), `fixes_compute.py` (`fix_panel_b_disorder.npz`) | Fig. 3(b), recall overlap versus input noise, disorder-averaged bands |
| `fix_panel_c_capacity.npz` | `make_paper_figures.py` | Fig. 3(c), recall success versus number of stored patterns |
| `fix_panel_d_palimpsest.npz` | `fixes_compute.py` | Fig. 3(d), retrievability versus pattern age |
| `fix_dense_bootstrap.npz` | `fixes_compute.py` | Fig. 4, two- versus three-body capacity, bootstrap slopes |
| `fix_scale_fig2b.npz` | `scale_fig2b.py` | the `n = 12, 14` repeat of Fig. 3(b) quoted in the Appendix |
| `fix_sweep_bias.npz` | `sweep_bias.py` | the input-bias sweep behind "provided the input bias is set to the value best for that size" |
| `fix_noise_bias.npz` | `sweep_noise_bias.py` | the joint (n, f, h) sweep and the bootstrap-versus-Gaussian interval check |
| `fix_gs_matched.npz`, `fix_gs_degeneracy_scale.npz` | `fixes_compute.py` | ground-state and degeneracy controls |
| `fix_quantum_ceiling.npz` | `fixes_compute.py` | the 1/2 readout ceiling on a superposed input |
| `fix_sweep_longT.npz` | `fixes_compute.py` | longer-anneal control |
| `fix_anneal_tuning.npz` | `sweep_anneal_tuning.py` | the (T, Omega0) schedule scan quoted in the Appendix |
