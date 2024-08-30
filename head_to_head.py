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
from tifffile import imread as tiffread
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from rcode import *
import matplotlib.pyplot as plt

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
    parser.add_argument("--p_pos",      type=float, default=None)
    parser.add_argument("--n_viz",      type=int,   default=160_000)
    parser.add_argument("--split",      type=str,   default="orig")
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

# What is the grographic distribution of training points?
# pos = df[df.labs == 1]
# _   = plt.scatter(pos.lon, pos.lat, c=pos.o_train, s=4, alpha=0.5)
# show_plot()

# --
# Features

X = df[[c for c in df.columns if 'infeat' in c]].values
E = df[[c for c in df.columns if 'srifeat' in c]].values
y = df.labs.values.astype(int)

X = X.astype(np.float32)
E = E.astype(np.float32) # [NOTE] rapids prefers this ... I assume it doesn't matter

X1 = X.reshape(X.shape[0], -1, 5, 5)[...,2:-2,2:-2].mean(axis=(2, 3))
X3 = X.reshape(X.shape[0], -1, 5, 5)[...,1:-1,1:-1].mean(axis=(2, 3))
X5 = X.reshape(X.shape[0], -1, 5, 5).mean(axis=(2, 3))

assert ((y == 0) | (y == 1)).all()

# --
# Train/test split

idx = np.arange(X.shape[0])

if args.split == 'orig':
    # sri's positive examples
    pos_sel    = np.where((df.o_train & (df.labs == 1)).values)[0]
    args.n_pos = pos_sel.shape[0]
elif args.split == 'random':
    # randomized positive examples
    pos_idx    = np.where(y)[0]
    args.n_pos = int(args.p_pos * len(pos_idx))
    pos_sel    = np.random.choice(pos_idx, args.n_pos, replace=False)

# [NOTE] We never use SRI's split beyond the positives, because their
#        "pu_sampler" chooses biased valid & test sets.
#        [TODO] Train a model using their train negatives ... 

# [NOTE] "validation" split is _everything_ except the training positives
#        because those are the only points for which we have labels.

neg_sel   = np.random.choice(np.setdiff1d(idx, pos_sel), args.n_neg, replace=False)

train_sel = np.sort(np.hstack([pos_sel, neg_sel]))
valid_sel = np.setdiff1d(idx, pos_sel)

X_train  = X[train_sel]
X1_train = X1[train_sel]
X3_train = X3[train_sel]
X5_train = X5[train_sel]
E_train  = E[train_sel]
y_train  = y[train_sel]
y_valid  = y[valid_sel]

# --
# With hyperparameter tuning

# from rf_hpo import HPO_RandomForestClassifier

# def r_auc(target, scores, n=None, p=None):
#     n_pos = target.sum()
    
#     curve = np.cumsum(target[np.argsort(-scores)])
    
#     if p:
#         n = np.where(curve > (n_pos * p))[0][0]
    
#     if n:
#         curve = curve[:n]
    
#     metric = np.trapz(
#         y = curve / target.sum(),
#         x = np.linspace(0, 1, curve.shape[0])
#     )
    
#     return metric

# rf_hpo  = HPO_RandomForestClassifier(cuda=True, n_estimators=1000)
# rf_hpo  = rf_hpo.tune(X_train, y_train, n_iters=32, score_fn=lambda *args: r_auc(*args, n=5 * y_train.sum()))
# oparams = rf_hpo.best_params()
# rf_hpo  = rf_hpo.fit(X_train, y_train, oparams)
# p_hpo   = rf_hpo.predict_proba(cp.array(X))[valid_sel].get()

# --
# Fit model

if args.split == 'orig':
    p_o = df.sri_score.values[valid_sel]

t1 = time()

if args.cuda:
    t2 = time()

    _X_train  = cp.array(X_train)
    _X1_train = cp.array(X1_train)
    _X3_train = cp.array(X3_train)
    _X5_train = cp.array(X5_train)
    _E_train  = cp.array(E_train)
    _y_train  = cp.array(y_train)

    rf_x  = cuRandomForestClassifier(n_estimators=2048, verbose=1).fit(_X_train, _y_train)
    rf_x1 = cuRandomForestClassifier(n_estimators=2048, verbose=1).fit(_X1_train, _y_train)
    rf_x3 = cuRandomForestClassifier(n_estimators=2048, verbose=1).fit(_X3_train, _y_train)
    rf_x5 = cuRandomForestClassifier(n_estimators=2048, verbose=1).fit(_X5_train, _y_train)
    rf_e  = cuRandomForestClassifier(n_estimators=2048, verbose=1).fit(_E_train, _y_train)

    del _X_train
    del _X1_train
    del _X3_train
    del _X5_train
    del _E_train
    del _y_train

    print('fit_time', time() - t2)
    t2 = time()

    _X  = cp.array(X)
    _X1 = cp.array(X1)
    _X3 = cp.array(X3)
    _X5 = cp.array(X5)
    _E  = cp.array(E)

    p_x  = rf_x.predict_proba(_X)[valid_sel,1]
    p_x1 = rf_x1.predict_proba(_X1)[valid_sel,1]
    p_x3 = rf_x3.predict_proba(_X3)[valid_sel,1]
    p_x5 = rf_x5.predict_proba(_X5)[valid_sel,1]
    p_e  = rf_e.predict_proba(_E)[valid_sel,1]

    p_x  = p_x.get()
    p_x1 = p_x1.get()
    p_x3 = p_x3.get()
    p_x5 = p_x5.get()
    p_e  = p_e.get()

    del _X
    del _X1
    del _X3
    del _X5
    del _E

    print('predict_time', time() - t2)
    
else:
    t2 = time()

    rf_x  = RandomForestClassifier(n_estimators=1024, n_jobs=-1, verbose=1).fit(X_train, y_train)
    rf_xc = RandomForestClassifier(n_estimators=1024, n_jobs=-1, verbose=1).fit(X_train[:,12::25], y_train)
    rf_e  = RandomForestClassifier(n_estimators=1024, n_jobs=-1, verbose=1).fit(E_train, y_train)

    print('fit_time', time() - t2)
    t2 = time()

    p_x  = rf_x.predict_proba(X)[valid_sel,1]
    p_xc = rf_xc.predict_proba(X[:,12::25])[valid_sel,1]
    p_e  = rf_e.predict_proba(E)[valid_sel,1]

    print('predict_time', time() - t2)



# --
# Plot

def do_rank(y_valid, p, n=None):
    out = np.cumsum(y_valid[np.argsort(-p)])
    if n:
        out = out[:n]
    
    return out

_ = plt.plot(do_rank(y_valid, p_x,  n=args.n_viz), label='rf  - raw features (flat)')
_ = plt.plot(do_rank(y_valid, p_x1, n=args.n_viz), label='rf  - raw features (1avg)')
_ = plt.plot(do_rank(y_valid, p_x3, n=args.n_viz), label='rf  - raw features (3avg)')
_ = plt.plot(do_rank(y_valid, p_x5, n=args.n_viz), label='rf  - raw features (5avg)')
_ = plt.plot(do_rank(y_valid, p_e,  n=args.n_viz), label='rf  - sri features')

if args.split == 'orig':
    _ = plt.plot(do_rank(y_valid, p_o,  n=args.n_viz), label='sri - orig')
    
# _ = plt.plot(do_rank(y_valid, p_hpo,  n=args.n_viz), label='rf - raw features (HPO)')
# _ = plt.xscale('log')
_ = plt.legend()
_ = plt.grid('both', alpha=0.25)
_ = plt.xlabel('rank')
_ = plt.ylabel('# positive')
_ = plt.title(f'{args.target_str} - {args.n_pos=} - {args.n_neg=} - {args.split=}')
show_plot(f'plots/{args.target_str}-{args.split}.png')

total_elapsed = time() - t0
model_elapsed = time() - t1
print(f'{args.cuda=} | {total_elapsed=} | {model_elapsed=}')