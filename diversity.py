from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
from itertools import combinations

def compute_internal_diversity(smiles_list, radius=2, n_bits=2048):
    """
    Computes the internal diversity of a set of molecules using average pairwise Tanimoto similarity.
    
    :param smiles_list: List of SMILES strings
    :param radius: Radius of the Morgan Fingerprint (default=2)
    :param n_bits: Size of the fingerprint vector (default=2048)
    :return: Internal diversity score (1 - average pairwise similarity)
    """
    # Convert SMILES to RDKit molecules
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    
    # Remove invalid molecules
    mols = [mol for mol in mols if mol is not None]
    
    if len(mols) < 2:
        print("Not enough valid molecules to compute internal diversity.")
        return None
    
    # Generate Morgan Fingerprints
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits) for mol in mols]

    # Compute pairwise Tanimoto similarity
    similarities = []
    for fp1, fp2 in combinations(fingerprints, 2):
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        similarities.append(similarity)

    # Compute the average similarity
    avg_similarity = np.mean(similarities)

    # Compute internal diversity
    internal_diversity = 1 - avg_similarity

    return internal_diversity
