"""Outer cross-validation partitions shared by step 01 (classical models) and steps 09/12 (graph network), so that
both families are always evaluated on identical folds."""
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import RepeatedKFold


def murcko_groups(smiles):
    """Bemis-Murcko scaffold (without stereochemistry) of each molecule; acyclic molecules share the empty scaffold."""
    groups = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        groups.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol is not None else "")
    return np.array(groups)


def scaffold_repeated_kfold(smiles, n_splits, n_repeats, seed):
    """Repeated scaffold-grouped K-fold: all molecules sharing a scaffold fall in the same fold. In each repeat the
    scaffolds are visited in a random order and each goes to the currently smallest fold."""
    groups = murcko_groups(smiles)
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.RandomState(seed)
    for _ in range(n_repeats):
        folds = [[] for _ in range(n_splits)]
        for g in rng.permutation(unique):
            min(folds, key=len).extend(members[g])
        for k in range(n_splits):
            test = np.sort(np.array(folds[k]))
            train = np.sort(np.concatenate([np.array(folds[j]) for j in range(n_splits) if j != k]))
            yield train, test


def outer_splits(smiles, n_splits, n_repeats, seed, kind="random"):
    """Yield (train_idx, test_idx) in fold order. kind='random' is sklearn's RepeatedKFold; 'scaffold' groups by scaffold."""
    if kind == "scaffold":
        yield from scaffold_repeated_kfold(smiles, n_splits, n_repeats, seed)
    elif kind == "random":
        yield from RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed).split(np.zeros(len(smiles)))
    else:
        raise ValueError(f"Unknown split kind '{kind}' (expected 'random' or 'scaffold')")
