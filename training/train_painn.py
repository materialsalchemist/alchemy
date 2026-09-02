"""Train PaiNN on original QM9 DFT geometries (no OpenBabel regeneration).

Reconstructs the architecture recovered from model.script via torch.jit.load
introspection: PaiNN, cutoff 5.0 A, BesselRBF n_rbf=20, n_atom_basis=128,
3 interaction blocks, Atomwise(128->64->1) readout, sum aggregation, per-element
atomref + global mean offset, free_energy output key, eV units.

Newly chosen (undocumented in the original run -- see training/README.md):
optimizer, LR schedule, batch size, epoch/stopping criterion, seed handling.
"""
import argparse
import pathlib

import pytorch_lightning as pl
import schnetpack as spk
import schnetpack.transform as trn
import torch
import torchmetrics

# Our own checkpoints embed schnetpack classes (e.g. NeuralNetworkPotential) that
# aren't on torch>=2.6's default weights_only safe-globals allowlist. These files are
# entirely self-produced by this script, so force the pre-2.6 trusted-load behavior.
_orig_torch_load = torch.load


def _trusting_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _trusting_torch_load


def build_datamodule(data_dir: pathlib.Path, split_file: pathlib.Path, batch_size: int, num_workers: int):
    cutoff = 5.0
    return spk.datasets.QM9(
        datapath=str(data_dir / "qm9.db"),
        batch_size=batch_size,
        num_train=None,  # driven entirely by split_file
        num_val=None,
        split_file=str(split_file),
        remove_uncharacterized=False,  # confirmed: no exclusion in the original 133,885-molecule pool
        num_workers=num_workers,
        property_units={"free_energy": "eV"},
        distance_unit="Ang",
        transforms=[
            trn.SubtractCenterOfMass(),
            trn.RemoveOffsets("free_energy", remove_atomrefs=True, remove_mean=True),
            trn.TorchNeighborList(cutoff=cutoff),  # matches model.script's recovered preprocessing
            trn.CastTo32(),
        ],
    )


def build_task(lr: float):
    cutoff = 5.0
    radial_basis = spk.nn.BesselRBF(n_rbf=20, cutoff=cutoff)
    representation = spk.representation.PaiNN(
        n_atom_basis=128,
        n_interactions=3,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff),
    )
    output = spk.atomistic.Atomwise(
        n_in=128,
        output_key="free_energy",
        aggregation_mode="sum",
    )
    nnp = spk.model.NeuralNetworkPotential(
        representation=representation,
        input_modules=[spk.atomistic.PairwiseDistances()],
        output_modules=[output],
        postprocessors=[
            trn.CastTo64(),
            trn.AddOffsets("free_energy", add_mean=True, add_atomrefs=True),
        ],
    )
    task = spk.task.AtomisticTask(
        model=nnp,
        outputs=[
            spk.task.ModelOutput(
                name="free_energy",
                loss_fn=torch.nn.MSELoss(),
                loss_weight=1.0,
                metrics={"MAE": torchmetrics.MeanAbsoluteError()},
            )
        ],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": lr},
        scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_args={"mode": "min", "factor": 0.8, "patience": 15},
        scheduler_monitor="val_loss",
    )
    return task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path("qm9_data"))
    ap.add_argument("--split-file", type=pathlib.Path, default=pathlib.Path("split.npz"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--max-time-hours", type=float, default=8.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--smoke-test", action="store_true", help="tiny run: few epochs, quick sanity check")
    ap.add_argument("--resume", action="store_true", help="resume from the newest *.ckpt in --out-dir, if any")
    args = ap.parse_args()

    pl.seed_everything(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = None
    if args.resume:
        ckpts = sorted(args.out_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
        if ckpts:
            ckpt_path = str(ckpts[-1])
            print(f"resuming from {ckpt_path}")
        else:
            print(f"--resume given but no checkpoint found in {args.out_dir}, starting fresh")

    dm = build_datamodule(args.data_dir, args.split_file, args.batch_size, args.num_workers)
    task = build_task(args.lr)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=2 if args.smoke_test else args.max_epochs,
        max_time={"hours": args.max_time_hours} if not args.smoke_test else None,
        default_root_dir=str(args.out_dir),
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, mode="min"),
            pl.callbacks.ModelCheckpoint(dirpath=str(args.out_dir), monitor="val_loss", save_top_k=1),
        ],
        log_every_n_steps=20,
    )
    trainer.fit(task, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
