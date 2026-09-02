"""Export a trained PaiNN checkpoint to the (Z, R, cell, pbc) -> dict TorchScript
contract energy_predictor.py expects. Matches model.script's I/O behaviorally
(exact internal module nesting of the original 'PortableModel' wrapper is not
recoverable and does not need to be -- only the call signature and output key do).
"""
import argparse
import pathlib
from typing import Dict

import schnetpack.transform as trn
import torch
from ase.data import atomic_masses as _ase_atomic_masses
from torch import nn

# Our checkpoints embed schnetpack classes not on torch>=2.6's default
# weights_only safe-globals allowlist. Entirely self-produced files -- force
# the pre-2.6 trusted-load behavior, matching train_painn.py.
_orig_torch_load = torch.load


def _trusting_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _trusting_torch_load


class PortablePaiNN(nn.Module):
    def __init__(self, task_checkpoint_path: str):
        super().__init__()
        import schnetpack.task as spk_task

        task = spk_task.AtomisticTask.load_from_checkpoint(task_checkpoint_path, map_location="cpu")
        model = task.model
        model.eval()

        # Pull the pieces we need out of NeuralNetworkPotential/AtomisticModel
        # directly rather than keeping `model` itself as a submodule.
        # AtomisticModel.collect_derivatives() sets `required_derivatives` to a
        # bare (empty, for us) list in a method outside __init__, so it carries
        # no annotation TorchScript's attribute-inference can see -- scripting
        # `model` as a whole fails with "List trace inputs must have elements"
        # no matter how required_derivatives is reassigned afterwards. We don't
        # need derivatives for a plain energy predictor, so we skip
        # AtomisticModel.forward/initialize_derivatives/extract_outputs
        # entirely and replicate NeuralNetworkPotential.forward's remaining
        # steps ourselves against the submodules directly.
        self.input_modules = model.input_modules
        self.representation = model.representation
        self.output_modules = model.output_modules

        # trn.CastTo64 (one of the model's postprocessors) calls a schnetpack
        # helper (`as_dtype`) that indexes a module-global dict -- TorchScript
        # can't compile a global dict lookup, so scripting the model whole
        # fails with "python value of type 'dict' cannot be used as a value".
        # Drop it here and do the equivalent float64 cast ourselves below in
        # plain, scriptable tensor ops; keep every other postprocessor
        # (notably AddOffsets, which restores the real energy scale) as-is.
        kept = [m for m in model.postprocessors if not isinstance(m, trn.CastTo64)]
        self.postprocessors = nn.ModuleList(kept)

        # trn.SubtractCenterOfMass has the same problem as CastTo64: its
        # forward() indexes `ase.data.atomic_masses`, a module-global numpy
        # array, which TorchScript can't close over ("python value of type
        # 'ndarray' cannot be used as a value"). Register it as a tensor
        # buffer instead and do the center-of-mass subtraction ourselves in
        # forward with plain, scriptable tensor ops.
        self.register_buffer(
            "atomic_masses", torch.tensor(_ase_atomic_masses, dtype=torch.float64)
        )
        self.cutoff: float = 5.0

    def forward(self, Z: torch.Tensor, R: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor) -> Dict[str, torch.Tensor]:
        n = Z.shape[0]
        Zl = Z.long()
        masses = self.atomic_masses.index_select(0, Zl).to(R.dtype)
        R = R - (masses.unsqueeze(-1) * R).sum(0) / masses.sum()

        # trn.TorchNeighborList itself calls `torch.Tensor([0], ...)` internally
        # (schnetpack/transform/neighborlist.py's _get_shifts), an old-style
        # constructor call that isn't in this torch version's TorchScript op
        # table ("Unknown builtin op: aten::Tensor") -- and that code path gets
        # compiled unconditionally regardless of pbc at runtime. QM9 geometries
        # are always isolated (non-periodic) molecules, so we replicate the
        # non-periodic special case directly: all pairs within cutoff, zero
        # offsets (no periodic images needed).
        idx = torch.arange(n, dtype=torch.long)
        idx_i = idx.repeat_interleave(n)
        idx_j = idx.repeat(n)
        keep = idx_i != idx_j
        idx_i = idx_i[keep]
        idx_j = idx_j[keep]
        dist = torch.norm(R[idx_j] - R[idx_i], p=2, dim=-1)
        within = dist <= self.cutoff
        idx_i = idx_i[within]
        idx_j = idx_j[within]
        offsets = torch.zeros((idx_i.shape[0], 3), dtype=R.dtype)

        inputs: Dict[str, torch.Tensor] = {
            "_atomic_numbers": Zl,
            "_positions": R,
            "_cell": cell.unsqueeze(0),
            "_pbc": pbc,
            "_idx_m": torch.zeros(n, dtype=torch.long),
            "_n_atoms": torch.tensor([n], dtype=torch.long),
            "_idx_i": idx_i,
            "_idx_j": idx_j,
            "_offsets": offsets,
        }
        # trn.CastTo32 has the same as_dtype-global-dict problem as CastTo64
        # (both subclass CastMap, whose forward() calls it) -- cast manually.
        cast_inputs: Dict[str, torch.Tensor] = {}
        for k in inputs:
            v = inputs[k]
            cast_inputs[k] = v.to(torch.float32) if v.dtype == torch.float64 else v
        inputs = cast_inputs

        for m in self.input_modules:
            inputs = m(inputs)
        inputs = self.representation(inputs)
        for m in self.output_modules:
            inputs = m(inputs)
        for pp in self.postprocessors:
            inputs = pp(inputs)

        out: Dict[str, torch.Tensor] = {}
        out["free_energy"] = inputs["free_energy"].to(torch.float64)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    wrapper = PortablePaiNN(str(args.checkpoint))
    wrapper.eval()
    scripted = torch.jit.script(wrapper)
    scripted.save(str(args.out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
