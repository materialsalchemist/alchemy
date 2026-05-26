import torch
import os
from openbabel import openbabel


class EnergyPredictor:
	def __init__(self, model_path="model.script"):
		self.model = None
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

		# Optimize with MMFF94
		ff = openbabel.OBForceField.FindForceField("mmff94")
		if ff is None:
			ff = openbabel.OBForceField.FindForceField("uff")

		if ff is None:
			raise RuntimeError("Could not find MMFF94 or UFF force field in OpenBabel.")

		ff.Setup(ob_mol)
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

		# Centering positions
		R = R - R.mean(dim=0)

		# Cell and PBC
		cell = torch.eye(3, dtype=torch.float32)
		pbc = torch.zeros(3, dtype=torch.bool)

		return Z, R, cell, pbc

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
