import logging
import os
import shutil
from pathlib import Path

import torch
from openbabel import openbabel

# --- OpenBabel data files ---------------------------------------------------
# Two defects in openbabel-wheel >= 3.1.1.23 make geometry generation fail
# silently rather than loudly, so both are repaired here, once, at import time.
#
# 1. The wheel's own __init__ sets BABEL_DATADIR to share/openbabel/<version>,
#    which contains only splash.png. The real data is in bin/data. Without it
#    OBBuilder runs with no fragment or typing tables and returns much poorer
#    geometries: a trimethylamine reaction energy moves from -80 to -36 kJ/mol
#    with nothing but an error line on stderr to show for it. The wheel sets the
#    variable when openbabel is imported, so the repair has to happen after that
#    import, which is why this runs here rather than in a caller.
#
# 2. bin/data ships a rigid-fragments-index.txt whose byte offsets do not match
#    rigid-fragments.txt, and no ring-fragments-index.txt at all. OBBuilder then
#    reads garbage coordinates for some fragments and dies with an access
#    violation, killing the interpreter rather than raising, on roughly 4% of
#    ring-containing molecules. Dropping the stale index makes OpenBabel fall
#    back to a linear scan, which is correct: published reaction energies
#    reproduce either way, and the crashes stop.
#
# The repair is made in a cached copy so the installed package is never touched.

_DATA_MARKER = "mmff94.ff"
_STALE_INDEX = "rigid-fragments-index.txt"


def _openbabel_data_candidates():
	import openbabel as openbabel_package

	base = Path(openbabel_package.__file__).resolve().parent
	candidates = [base / "bin" / "data", base / "data"]
	share = base / "share" / "openbabel"
	if share.is_dir():
		candidates.extend(sorted(p for p in share.iterdir() if p.is_dir()))
	return candidates


def configure_openbabel_data(cache_dir: str = None) -> str:
	"""Point BABEL_DATADIR at a complete, self-consistent OpenBabel data directory.

	Returns the directory in use. Raises if no candidate contains the force-field
	parameters, because every geometry produced without them is unusable.
	"""
	target = Path(cache_dir or Path.home() / ".cache" / "alchemy" / "openbabel_data")

	if (target / _DATA_MARKER).is_file() and not (target / _STALE_INDEX).is_file():
		os.environ["BABEL_DATADIR"] = str(target)
		return str(target)

	source = next((c for c in _openbabel_data_candidates() if (c / _DATA_MARKER).is_file()), None)
	if source is None:
		raise RuntimeError(
			"No OpenBabel data directory containing "
			f"{_DATA_MARKER} was found. Geometry generation cannot work. "
			"Check the openbabel installation."
		)

	if not (source / _STALE_INDEX).is_file():
		os.environ["BABEL_DATADIR"] = str(source)
		return str(source)

	target.parent.mkdir(parents=True, exist_ok=True)
	if target.exists():
		shutil.rmtree(target)
	shutil.copytree(source, target)
	(target / _STALE_INDEX).rename(target / f"{_STALE_INDEX}.disabled")
	logging.info(f"Repaired OpenBabel data directory cached at {target}")
	os.environ["BABEL_DATADIR"] = str(target)
	return str(target)


configure_openbabel_data()


def _find_force_field():
	ff = openbabel.OBForceField.FindForceField("mmff94")
	if ff is None:
		ff = openbabel.OBForceField.FindForceField("uff")
	if ff is None:
		raise RuntimeError("Could not find MMFF94 or UFF force field in OpenBabel.")
	return ff


_FORCE_FIELD_CHECKED = False


def assert_force_field_works():
	"""Confirm on a known molecule that the force field actually moves atoms.

	OBForceField.Setup returns a bool that is easy to discard, and when it is
	False the optimisation calls that follow are silent no-ops: the geometry that
	reaches the model is then an unrelaxed OBBuilder structure and the prediction
	looks perfectly normal. Failing once at startup is far better than degrading
	every prediction thereafter.
	"""
	global _FORCE_FIELD_CHECKED
	if _FORCE_FIELD_CHECKED:
		return

	conversion = openbabel.OBConversion()
	conversion.SetInFormat("smi")
	mol = openbabel.OBMol()
	conversion.ReadString(mol, "CCO")
	mol.AddHydrogens()
	openbabel.OBBuilder().Build(mol)

	before = [(mol.GetAtom(i).GetX(), mol.GetAtom(i).GetY(), mol.GetAtom(i).GetZ())
	          for i in range(1, mol.NumAtoms() + 1)]

	ff = _find_force_field()
	if not ff.Setup(mol):
		raise RuntimeError(
			"OpenBabel force field failed to initialise on ethanol. Geometries would "
			"be unoptimised OBBuilder structures and every predicted energy would be "
			f"wrong without warning. BABEL_DATADIR={os.environ.get('BABEL_DATADIR')!r}"
		)
	ff.SteepestDescent(200, 1.0e-4)
	ff.GetCoordinates(mol)

	after = [(mol.GetAtom(i).GetX(), mol.GetAtom(i).GetY(), mol.GetAtom(i).GetZ())
	         for i in range(1, mol.NumAtoms() + 1)]
	moved = max(abs(a - b) for pa, pb in zip(before, after) for a, b in zip(pa, pb))
	if moved < 1e-6:
		raise RuntimeError(
			"OpenBabel force field reported success but moved no atoms, so it is inert. "
			f"BABEL_DATADIR={os.environ.get('BABEL_DATADIR')!r}"
		)
	_FORCE_FIELD_CHECKED = True


class EnergyPredictor:
	# Geometry sanity limits, matching the criteria used to prepare the training set.
	MIN_INTERATOMIC_DISTANCE = 0.15

	def __init__(self, model_path="model.script", check_force_field=True):
		self.model = None
		if check_force_field:
			assert_force_field_works()
		if os.path.exists(model_path):
			try:
				self.model = torch.jit.load(model_path)
				self.model.eval()
			except Exception as e:
				print(f"Failed to load model: {e}")

	def get_molecule_from_smiles(self, smiles):
		"""
		Converts a SMILES string to the tensors required by the model using OpenBabel.
		Identical logic to updated run_model.py
		"""
		# Setup OpenBabel conversion
		ob_conversion = openbabel.OBConversion()
		ob_conversion.SetInFormat("can")  # Canonical SMILES
		ob_mol = openbabel.OBMol()

		if not ob_conversion.ReadString(ob_mol, smiles):
			raise ValueError(f"Invalid SMILES string: {smiles}")

		# Add Hydrogens and generate 3D coordinates
		ob_mol.AddHydrogens()
		builder = openbabel.OBBuilder()
		builder.Build(ob_mol)

		ff = _find_force_field()

		# Setup returns False when the force field cannot type the molecule or its
		# parameter files are missing. Continuing past that silently optimises
		# nothing and yields an unrelaxed geometry that still produces a number.
		if not ff.Setup(ob_mol):
			raise RuntimeError(
				f"Force field setup failed for {smiles!r}; refusing to return an "
				f"unoptimised geometry. BABEL_DATADIR={os.environ.get('BABEL_DATADIR')!r}"
			)

		ff.SteepestDescent(1000, 1.0e-4)
		ff.WeightedRotorSearch(50, 20)
		ff.ConjugateGradients(3000, 1.0e-6)
		ff.GetCoordinates(ob_mol)

		# Extract atomic numbers and positions
		num_atoms = ob_mol.NumAtoms()
		atomic_numbers = []
		positions = []

		for i in range(1, num_atoms + 1):
			atom = ob_mol.GetAtom(i)
			atomic_numbers.append(atom.GetAtomicNum())
			positions.append([atom.GetX(), atom.GetY(), atom.GetZ()])

		Z = torch.tensor(atomic_numbers, dtype=torch.long)
		R = torch.tensor(positions, dtype=torch.float32)

		self._assert_geometry_is_physical(smiles, R)

		# Centering positions
		R = R - R.mean(dim=0)

		# Cell and PBC
		cell = torch.eye(3, dtype=torch.float32)
		pbc = torch.zeros(3, dtype=torch.bool)

		return Z, R, cell, pbc

	def _assert_geometry_is_physical(self, smiles, R):
		"""Reject undefined or collapsed geometries instead of scoring them.

		Some graph-valid records cannot be realised in three dimensions at all;
		strained polycyclic radicals are the usual case. Without this check the
		builder returns whatever it managed and the model ranks the result
		alongside well-formed structures.
		"""
		if not torch.isfinite(R).all():
			raise ValueError(f"Non-finite coordinates generated for {smiles!r}")
		if R.shape[0] > 1:
			distances = torch.cdist(R, R)
			distances.fill_diagonal_(float("inf"))
			closest = float(distances.min())
			if closest < self.MIN_INTERATOMIC_DISTANCE:
				raise ValueError(
					f"Interatomic distance {closest:.3f} A below "
					f"{self.MIN_INTERATOMIC_DISTANCE} A for {smiles!r}; geometry is unphysical"
				)

	def predict_free_energy(self, smiles):
		if self.model is None:
			return None

		try:
			Z, R, cell, pbc = self.get_molecule_from_smiles(smiles)

			with torch.no_grad():
				outputs = self.model(Z, R, cell=cell, pbc=pbc)

			if "free_energy" in outputs:
				return float(outputs["free_energy"].item())

			for val in outputs.values():
				return float(val.item())

			return 0.0
		except Exception:
			return None
