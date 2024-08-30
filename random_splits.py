#!/usr/bin/env python

"""
    head_to_head.py
    
    [TODO]
    - [ ] Train on randomized positive points
    - [ ] Train on geographically stratified positive points
    - [x] `vizier` HPO for RandomForest
"""

try:
    import cupy as cp
    from cuml.ensemble import RandomForestClassifier as cuRandomForestClassifier
except:
    print('!! no cuda support')

import os
import argparse
import numpy as np
import pandas as pd
from time import time
from tqdm import trange
from tifffile import imread as tiffread
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from rcode import *
import matplotlib.pyplot as plt

# --
# Helpers

def do_rank(y_valid, p, n=None):
    out = np.cumsum(y_valid[np.argsort(-p)])
    if n:
        out = out[:n]
    
    return out

# --
# CLI

TARGETS = [
    ("natl_maniac_mae",       'National MaNiAC'),
    ("natl_mvt_mae",          'National MVT'),
    ("natl_cu_mae",           'National Porphyry Copper'),
    ("natl_w_mae",            'National Tungsten-skarn'),
    ("umidwest_mamanico_mae", 'Regional Mafic Magmatic Nickel-Cobalt'),
    ("smidcont_mvt_mae",      'Regional MVT'),
    ("ytu_w_mae",             'Regional Tungsten-skarn'),
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_idx", type=int,   default=0)
    parser.add_argument("--seed",       type=int,   default=123)
    parser.add_argument("--n_neg",      type=int,   default=20_000)
    parser.add_argument("--p_pos",      type=float, default=0.5)
    parser.add_argument("--n_viz",      type=int,   default=1_000_000)
    parser.add_argument("--split",      type=str,   default="random")
    parser.add_argument("--cuda",       action="store_true")
    args = parser.parse_args()
    
    args.target_str, args.target_name = TARGETS[args.target_idx]
    args.model_type = "MAE"
    print(f"target={args.target_str, args.target_name}")
    
    if args.split == 'random':
        assert args.p_pos is not None, "args.split == random -> must set args.p_pos"
    
    return args

args = parse_args()
np.random.seed(args.seed)
os.makedirs('plots', exist_ok=True)

t0 = time()

# --
# Load features (computed by Jataware ; output of prep.py)

df = pd.read_feather(f'maps_prepped/{args.target_str}.feather')

# --
# Features

X = df[[c for c in df.columns if 'infeat' in c]].values
E = df[[c for c in df.columns if 'srifeat' in c]].values
y = df.labs.values.astype(int)

X = X.astype(np.float32)
E = E.astype(np.float32) # [NOTE] rapids prefers this ... I assume it doesn't matter

assert ((y == 0) | (y == 1)).all()

# <<
# Stupid, but probably helps
# C = df[['lat', 'lon']].values
# X = np.column_stack([X, C])
# >>

# --
# Train/test split

_X  = cp.array(X)
_E  = cp.array(E)

idx     = np.arange(X.shape[0])
pos_idx = np.where(y)[0]

x_res, e_res = [], []
for it in trange(32):
    
    # --
    # Train/test split

    args.n_pos = int(args.p_pos * len(pos_idx))
    pos_sel    = np.random.choice(pos_idx, args.n_pos, replace=False)

    neg_sel   = np.random.choice(np.setdiff1d(idx, pos_sel), args.n_neg, replace=False)

    train_sel = np.sort(np.hstack([pos_sel, neg_sel]))
    valid_sel = np.setdiff1d(idx, pos_sel)

    X_train  = X[train_sel]
    E_train  = E[train_sel]
    y_train  = y[train_sel]
    y_valid  = y[valid_sel]

    # --
    # Fit model

    _X_train  = cp.array(X_train)
    _E_train  = cp.array(E_train)
    _y_train  = cp.array(y_train)

    rf_x  = cuRandomForestClassifier(n_estimators=2048, verbose=1).fit(_X_train, _y_train)
    rf_e  = cuRandomForestClassifier(n_estimators=2048, verbose=1).fit(_E_train, _y_train)

    del _X_train
    del _E_train
    del _y_train

    p_x  = rf_x.predict_proba(_X)[valid_sel,1]
    p_e  = rf_e.predict_proba(_E)[valid_sel,1]

    p_x  = p_x.get()
    p_e  = p_e.get()

    x_res.append(do_rank(y_valid, p_x,  n=args.n_viz))
    e_res.append(do_rank(y_valid, p_e,  n=args.n_viz))

    # --
    # Traj plots

    for i, (r_e, r_x) in enumerate(zip(e_res, x_res)):
        _ = plt.plot(r_e, c='red', alpha=0.25)
        _ = plt.plot(r_x, c='blue', alpha=0.25)

    _ = plt.plot(np.row_stack(e_res).mean(axis=0), c='red', alpha=1, linewidth=2, label='EMB')
    _ = plt.plot(np.row_stack(x_res).mean(axis=0), c='blue', alpha=1, linewidth=2, label='RAW')

    _ = plt.xscale('log')
    _ = plt.legend()
    _ = plt.grid('both', alpha=0.25)
    _ = plt.xlabel('Rank')
    _ = plt.ylabel('# positive')
    _ = plt.title(f'{args.target_str} - {args.n_pos=} - {args.n_neg=} - {args.split=}')
    show_plot(f'plots/{args.target_str}-traj.png')

    # --
    # Paired plots
    
    for i, (r_e, r_x) in enumerate(zip(e_res, x_res)):
        _ = plt.plot(r_e - r_x, c='black', alpha=0.25)
    
    mu = (np.row_stack(e_res) - np.row_stack(x_res)).mean(axis=0)
    _  = plt.plot(mu, alpha=1, color='black', linewidth=3)
    
    _ = plt.xscale('log')
    # _ = plt.legend()
    _ = plt.grid('both', alpha=0.25)
    _ = plt.xlabel('Rank')
    _ = plt.ylabel('SRI Advantage (# positive)')
    _ = plt.title(f'{args.target_str} - {args.n_pos=} - {args.n_neg=} - {args.split=}')
    show_plot(f'plots/{args.target_str}-paired.png')
    
    np.save(f'plots/{args.target_str}-random_splits-e_res.npy', np.row_stack(e_res))
    np.save(f'plots/{args.target_str}-random_splits-x_res.npy', np.row_stack(x_res))

