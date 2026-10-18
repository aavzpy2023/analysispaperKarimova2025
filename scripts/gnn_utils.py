"""D-MPNN training helpers for the tuned graph-network benchmark (step 12). The default configuration reproduces the
model of step 09 (Chemprop defaults: hidden size 300, depth 3, no dropout, 100 epochs)."""
import itertools
import json
import os
import time
import numpy as np

HIDDEN_SIZES = (300, 600)
DEPTHS = (3, 5)
DROPOUTS = (0.0, 0.2)
GRID = [dict(d_h=h, depth=d, dropout=p) for h, d, p in itertools.product(HIDDEN_SIZES, DEPTHS, DROPOUTS)]
DEFAULT_CONFIG = dict(d_h=300, depth=3, dropout=0.0)
EPOCHS = int(os.environ.get("MLGNN_GNN_EPOCHS", "100"))
INNER_VALIDATION_FRACTION = 0.2


def config_name(cfg):
    return f"h{cfg['d_h']}_d{cfg['depth']}_p{cfg['dropout']}"


def train_predict(smiles_train, y_train, smiles_test, seed, cfg, threads):
    """Train a D-MPNN with configuration `cfg` on the training molecules and return predictions (original pIC50 scale)
    for `smiles_test`. The targets are standardized with the training mean and standard deviation."""
    import chemprop
    import torch
    from lightning import pytorch as pl
    from chemprop import data as cpdata, models, nn as cpnn
    torch.set_num_threads(threads)
    pl.seed_everything(seed, workers=True)
    featurizer = chemprop.featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dset = cpdata.MoleculeDataset([cpdata.MoleculeDatapoint.from_smi(s, [t]) for s, t in zip(smiles_train, y_train)],
                                        featurizer)
    test_dset = cpdata.MoleculeDataset([cpdata.MoleculeDatapoint.from_smi(s, [0.0]) for s in smiles_test], featurizer)
    scaler = train_dset.normalize_targets()
    test_dset.normalize_targets(scaler)
    train_loader = cpdata.build_dataloader(train_dset, shuffle=True, num_workers=0)
    test_loader = cpdata.build_dataloader(test_dset, shuffle=False, num_workers=0)
    mp = cpnn.BondMessagePassing(d_h=cfg['d_h'], depth=cfg['depth'], dropout=cfg['dropout'])
    ffn = cpnn.RegressionFFN(input_dim=cfg['d_h'], hidden_dim=cfg['d_h'], dropout=cfg['dropout'])
    mpnn = models.MPNN(mp, cpnn.MeanAggregation(), ffn, batch_norm=True, metrics=[cpnn.metrics.RMSE()])
    trainer = pl.Trainer(max_epochs=EPOCHS, enable_progress_bar=False, enable_model_summary=False,
                         enable_checkpointing=False, logger=False, accelerator='cpu')
    trainer.fit(mpnn, train_loader)
    preds = trainer.predict(mpnn, test_loader)
    return scaler.inverse_transform(torch.cat(preds).numpy().reshape(-1, 1)).flatten()


def run_outer_fold(task):
    """One outer fold of the nested search: pick the configuration of GRID with the lowest RMSE on a held-out part of
    the training set, retrain it on the whole training set and predict the outer test fold. Runs in a worker process."""
    from sklearn.metrics import mean_absolute_error, r2_score
    fold, smiles_tr, y_tr, smiles_te, y_te, seed, threads, cache_dir = task
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    t0 = time.time()
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(smiles_tr))
    n_val = int(round(INNER_VALIDATION_FRACTION * len(smiles_tr)))
    val_idx, fit_idx = perm[:n_val], perm[n_val:]
    smiles_tr, y_tr = np.asarray(smiles_tr), np.asarray(y_tr)
    inner = []
    for cfg in GRID:
        cache = os.path.join(cache_dir, f"fold{fold}_{config_name(cfg)}.json") if cache_dir else None
        if cache and os.path.exists(cache):
            with open(cache) as fh:
                inner.append(json.load(fh)["rmse"])
            continue
        pred = train_predict(smiles_tr[fit_idx].tolist(), y_tr[fit_idx].tolist(), smiles_tr[val_idx].tolist(), seed, cfg, threads)
        rmse = float(np.sqrt(np.mean((pred - y_tr[val_idx]) ** 2)))
        inner.append(rmse)
        if cache:
            with open(cache, "w") as fh:
                json.dump({"rmse": rmse}, fh)
    best = int(np.argmin(inner))
    pred = train_predict(smiles_tr.tolist(), y_tr.tolist(), list(smiles_te), seed, GRID[best], threads)
    return {"Fold": fold, "Best_Config": config_name(GRID[best]), "Inner_RMSE_Best": inner[best],
            "Inner_RMSE_Default": inner[0], "R2": float(r2_score(y_te, pred)),
            "MAE": float(mean_absolute_error(y_te, pred)), "N_train": len(smiles_tr), "N_test": len(smiles_te),
            "Time_s": time.time() - t0, "pred": pred.tolist()}
