import os
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_mopac_input(smiles, filename):
    # Convert SMILES to molecule and optimize geometry
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)  # Add hydrogens
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())  # Generate 3D coordinates
    AllChem.UFFOptimizeMolecule(mol)  # Optimize geometry with UFF

    # Write MOPAC input file
    with open(filename, 'w') as f:
        f.write("PM7\n")  # Specify the PM7 method
        f.write("HOMO LUMO\n")  # Request HOMO and LUMO properties
        f.write("\n")  # Blank line before geometry
        for atom in mol.GetAtoms():
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            f.write(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")

def run_mopac(input_file):
    # Run MOPAC
    output_file = input_file.replace('.mop', '.out')
    subprocess.run(['/home/mila/f/florence.cloutier/.conda/envs/env_chem/bin/mopac', input_file], check=True)
    return output_file

def extract_homo_lumo_gap(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            if "HOMO LUMO ENERGIES (EV)" in line:
                # Extract HOMO and LUMO values
                parts = line.split("=")[1].strip().split()
                homo = float(parts[0])
                lumo = float(parts[1])
                # Calculate HOMO-LUMO gap
                return abs(lumo - homo)
    raise ValueError("HOMO-LUMO energies not found in the file.")
