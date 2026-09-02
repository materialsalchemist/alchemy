# Original-QM9-geometry PaiNN training pipeline

This directory retrains a PaiNN model on **original QM9 DFT geometries** (as
opposed to the OpenBabel-reoptimized geometries the released `model.script`
was trained on). It exists because no training script for either geometry
variant survived from the original project — only the released inference
code and one TorchScript checkpoint did — so the JCIM Table 1 cell "trained
on original QM9" could not be regenerated or checked by anyone. This
pipeline replaces that unreproducible number with a real, rerunnable one.

Everything below that isn't already self-evident from the code is a
**newly-chosen assumption**, disclosed here because no record of the
original run's choices survives.

## Pipeline

```
build_split.py          -> split.npz               (train/val/test indices)
train_painn.py           run_seed{1,2}/*.ckpt       (2 independently seeded runs)
export_torchscript.py    model_original_geometry_seed{1,2}.script
```

`run_detached.sh <seed> <out_dir>` launches `launch_instrumented.sh` fully
detached (see "Silent deaths" below) — this is how both real runs were
actually started, not a direct `python3 train_painn.py` invocation.

## Data

QM9 is downloaded and parsed by `schnetpack.datasets.QM9` into
`qm9_data/qm9.db` on first run (same figshare tarball the paper's other
reconstructed baselines use). **Raw QM9 `.xyz` files are not kept anywhere
on this machine** — only the parsed ASE db. Anyone reusing this pipeline
should know:

- `qm9.db` stores `free_energy` in **raw Hartree**, not eV, regardless of
  the `property_units={"free_energy": "eV"}` argument passed when building
  the datamodule. That argument only controls the unit conversion applied
  on-the-fly to batches inside the training/eval DataLoader pipeline — it
  does not rewrite what schnetpack physically wrote to the on-disk db while
  parsing. Confirmed directly: `qm9.db` row 1 (methane) has
  `free_energy = -40.4986`, the well-known Hartree-scale DFT total energy for
  CH4, not an eV-scale value. Any script reading `row.data["free_energy"]`
  straight from the db (as `../../a2i2_alchemy/benchmarks/baselines/eval_original_geometry_cells.py`
  does) must convert from Hartree, not eV — using eV-conversion constants on
  a Hartree value silently produces errors off by the Hartree/eV ratio
  (~27.2×) that still look like "a model", not obviously wrong, until you
  check the absolute magnitude against literature values.

## Split (`build_split.py`)

Replays `benchmarks/baselines/qm9_baselines.py`'s split procedure
(`rng seed=0`, permutation, `test_frac=0.0764`, its exact rounding rule)
against schnetpack's QM9 row order. schnetpack sorts molecule files the same
way `qm9_baselines.py`/`geometry_cell.py` already did, so **train/test
membership here is index-identical with the existing published 1.07/9.87
baselines**, not just a similarly-distributed split.

- `test_frac=0.0764`, `seed=0` — matches `qm9_baselines.py`'s defaults; gives
  exactly 10,229 test molecules.
- `val_frac=0.05` — **new, disclosed choice**; no original validation-set
  size survives from the original run.
- Resulting split (`split.npz`): 116,962 train / 6,694 val / 10,229 test.

## Architecture and training choices

Architecture matches what `model.script` (the released checkpoint) reveals
under `torch.jit.load` introspection: PaiNN, `n_atom_basis=128`,
`n_interactions=3`, `BesselRBF(n_rbf=20, cutoff=5.0)`, `CosineCutoff(5.0)`,
sum-aggregated `Atomwise` readout, per-element atomref + global mean energy
offset, `free_energy` output key in eV.

Everything below this line is **newly chosen** — no record of the original
training recipe survives:

- Optimizer: AdamW, `lr=5e-4`.
- LR schedule: `ReduceLROnPlateau(mode="min", factor=0.8, patience=15)`,
  monitoring `val_loss`.
- Early stopping: patience 30 epochs on `val_loss`.
- Batch size 64, up to `max_epochs=500`, wall-clock cap `max_time_hours=8.0`
  per launch (not per run — see "Resuming" below).
- 2 independently seeded runs (`--seed 1`, `--seed 2`), same data split, to
  report a small cross-seed spread rather than a single point estimate.

## Resuming (`--resume`)

Each `max_time_hours=8.0` launch is one slice of a longer run. `--resume`
finds the newest `*.ckpt` in `--out-dir` (by mtime) and passes it to
`trainer.fit(..., ckpt_path=ckpt_path)`, so the same command can be
relaunched repeatedly until the run actually finishes (converges or
early-stops). `run_detached.sh`/`launch_instrumented.sh` always pass
`--resume`.

**Gotcha — the EarlyStopping-triggered stop message never showed up in
either real run's captured log.** Both seed runs demonstrably finished
cleanly (`run_seed{1,2}.exitstatus` both show `python3 exit code: 0`), and
seed 1's log shows four separate `--resume` relaunches all resuming from the
same `epoch=142-step=261404.ckpt` with the checkpoint epoch never advancing
— i.e. the run had already converged and each relaunch just re-confirmed
that and exited quickly. But grepping either real run's log (or its
`.nohup.log`) for anything like `early stop|Monitored metric|stopping
threshold|did not improve|best model|converge|plateau` returns **zero
matches**. The only "stopped" message found anywhere is
`smoke_test.log:28: `Trainer.fit` stopped: `max_epochs=2` reached.` — from
the unrelated 2-epoch smoke test, which stopped via `max_epochs`, not
`EarlyStopping`. So: **the only reliable signal that a `--resume`'d run has
already finished is a checkpoint epoch that fails to advance across
repeated relaunches — do not rely on an explicit stopping message showing up
in the log.**

Seed 2's log shows this pipeline being resumed several more times than seed
1 before settling (best checkpoint at `epoch=221-step=405816.ckpt`, later
epochs up to ~251 logged without a new best), including at least one
relaunch that picked up right after a `SIGTERM` was received mid-run — i.e.
exactly the kind of interruption the `setsid`/`nohup` fix below exists to
survive without losing progress.

## Silent deaths (`run_detached.sh`)

Two earlier long training runs died silently with no error captured. Root
cause (confirmed via `last -F` utmp records matched to the second against
each run's last log timestamp): the **tmux pane** hosting each run was
destroyed at the exact death time, while the tmux **server** itself stayed
up throughout. Closing a pane kills its entire foreground process group,
including a process merely started via `tmux send-keys` — the tmux session
surviving is not enough.

Fix, in `run_detached.sh`:

```bash
setsid nohup bash launch_instrumented.sh "$SEED" "$OUT_DIR" \
    > "${OUT_DIR}.nohup.log" 2>&1 < /dev/null &
```

`setsid` gives the process its own session with no controlling terminal at
all, so there is no tty left for a pane/session close to hang up on;
`nohup` is a redundant second layer that ignores `SIGHUP` outright. Use
`run_detached.sh`, not a bare `tmux send-keys` invocation, for any run
expected to outlive the terminal it was started from.

`launch_instrumented.sh` additionally captures the *real* python exit code
via `PIPESTATUS[0]` (a plain `python3 | tee` pipeline hides the true exit
code behind tee's own), and runs a `poll_resources` loop every 10s writing
memory/process/GPU stats to `<out_dir>.resources.log`, so that if a run does
die unexpectedly there is forensic data instead of nothing.

## Export (`export_torchscript.py`)

Exports each trained checkpoint to the `(Z, R, cell, pbc) -> {"free_energy":
...}` TorchScript contract `reaction_space/energy_predictor.py` expects.
`torch.jit.script` on schnetpack's stock modules required several
workarounds (the original export wrapper class did not survive either, so
this is a from-scratch reimplementation of an equivalent, not a restoration
of the original):

- Extract and script individual submodules rather than scripting the whole
  `NeuralNetworkPotential` — scripting it directly fails on TorchScript's
  inability to infer the `required_derivatives` attribute.
- Replace `CastTo32`/`CastTo64` with manual per-key `.float()`/`.double()`
  casts — the stock transforms do a global-dict `as_dtype` lookup that
  TorchScript can't compile.
- Replace `SubtractCenterOfMass` with a manual buffer-based mean-subtraction
  — the stock transform closes over a global numpy array, which TorchScript
  can't script.
- Replace `TorchNeighborList` with a hand-written brute-force (non-periodic)
  neighbor list — the stock implementation calls a deprecated
  `torch.Tensor([0], ...)` constructor form that TorchScript rejects.
- Use `torch.norm(x, p=2, dim=-1)` instead of the `.norm(dim=-1)` method
  form — TorchScript's overload resolution only accepts the functional form
  here.

Also loads checkpoints via a `weights_only=False` monkeypatch on
`torch.load` (both here and in `train_painn.py`'s `--resume` path) — recent
`torch` defaults `weights_only=True`, and these checkpoints embed schnetpack
classes that aren't on torch's safe-globals allowlist.

## What is NOT committed here

Checkpoints (`run_seed{1,2}/*.ckpt`), TorchScript exports (`*.script`), the
downloaded QM9 db (`qm9_data/qm9.db`), and run-artifact logs
(`*.log`, `*.nohup.log`, `*.resources.log`, `*.exitstatus`) are all large
binaries/run artifacts and are **not** committed to this repository. They
currently live on-disk at `/home/abshe/MyCodes/alchemy/training/` on the
machine the runs were performed on. Only the pipeline scripts themselves
(`build_split.py`, `train_painn.py`, `export_torchscript.py`,
`launch_instrumented.sh`, `run_detached.sh`, this README) and `split.npz`
(small, and needed to reproduce exact train/val/test membership) are
tracked.
