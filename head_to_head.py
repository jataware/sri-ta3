#!/usr/bin/env python

"""
    head_to_head.py
    
    [TODO] Train on randomized positive points
    [TODO] Train on geographically stratified positive points
    [TODO] `vizier` HPO for RandomForest
"""

import os
import argparse
import numpy as np
import pandas as pd
from tifffile import imread as tiffread
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from rcode import *
import matplotlib.pyplot as plt

# --
# Helpers

def prep_df(df):
    n_row = df.shape[0]
    df    = df.drop_duplicates().reset_index(drop=True)
    
    # if df.shape[0] != n_row:
    #     print(f'!! dropped {n_row - df.shape[0]} rows')
    
    _lon = df.lon.apply(int)
    _lat = df.lat.apply(int)
    assert (_lon == df.lon).all()
    assert (_lat == df.lat).all()
    df.lon = _lon
    df.lat = _lat

    df['hash'] = df[['lon', 'lat']].apply(lambda x: hash(tuple(x)), axis=1)
    
    return df

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
    parser.add_argument('--target_idx', type=int, default=0)
    parser.add_argument('--seed',       type=int, default=123)
    parser.add_argument('--n_neg',      type=int, default=10_000)
    parser.add_argument('--n_pos',      type=int, default=None)
    parser.add_argument('--n_viz',      type=int, default=160_000)
    parser.add_argument('--splits',     type=str, default='orig')
    args = parser.parse_args()
    
    args.target = TARGETS[args.target_idx]
    args.model_type = 'MAE'
    print(f'target={args.target}')
    return args

args = parse_args()
np.random.seed(args.seed)
os.makedirs('plots', exist_ok=True)

target_str, target_name  = TARGETS[args.target_idx]

# --
# Load features (computed by Jataware)

df = pd.read_feather(f'maps_prepped/{target_str}.feather')

# --
# Features

X = df[[c for c in df.columns if 'infeat' in c]].values
E = df[[c for c in df.columns if 'srifeat' in c]].values
y = df.labs.values.astype(int)

assert ((y == 0) | (y == 1)).all()

# --
# Train/test split

idx = np.arange(X.shape[0])

if args.splits == 'orig':
    # sri's positive examples
    pos_sel    = np.where((df.o_train & (df.labs == 1)).values)[0]
    args.n_pos = pos_sel.shape[0]
elif args.split == 'random':
    # randomized positive examples
    pos_idx = np.where(y)[0]
    pos_sel = np.random.choice(pos_idx, args.n_pos, replace=False)

# [NOTE] We never use SRI's splits beyond the positives, because their
#        "pu_sampler" chooses biased valid & test sets.
#        [TODO] Train a model using their train negatives ... 

# [NOTE] "validation" split is _everything_ except the training positives
#        because those are the only points for which we have labels.

neg_sel   = np.random.choice(np.setdiff1d(idx, pos_sel), args.n_neg, replace=False)

train_sel = np.sort(np.hstack([pos_sel, neg_sel]))
valid_sel = np.setdiff1d(idx, pos_sel)

X_train = X[train_sel]
E_train = E[train_sel]
y_train = y[train_sel]
y_valid = y[valid_sel]

breakpoint()

# --
# Fit model

from cuml.ensemble import RandomForestClassifier as cuRandomForestClassifier

rf_x  = cuRandomForestClassifier(n_estimators=1024, n_jobs=-1, verbose=1).fit(X_train, y_train)
rf_xc = cuRandomForestClassifier(n_estimators=1024, n_jobs=-1, verbose=1).fit(X_train[:,12::25], y_train)
rf_e  = cuRandomForestClassifier(n_estimators=1024, n_jobs=-1, verbose=1).fit(E_train, y_train)

p_x  = rf_x.predict_proba(X)[valid_sel,1]
p_xc = rf_xc.predict_proba(X[:,12::25])[valid_sel,1]
p_e  = rf_e.predict_proba(E)[valid_sel,1]
p_o  = df.sri_score.values[valid_sel]

# --
# Plot

def do_rank(y_valid, p, n=None):
    out = np.cumsum(y_valid[np.argsort(-p)])
    if n:
        out = out[:n]
    
    return out

_ = plt.plot(do_rank(y_valid, p_x,  n=args.n_viz), label='rf  - raw features')
_ = plt.plot(do_rank(y_valid, p_xc, n=args.n_viz), label='rf  - raw features (center)')
_ = plt.plot(do_rank(y_valid, p_e,  n=args.n_viz), label='rf  - sri features')
_ = plt.plot(do_rank(y_valid, p_o,  n=args.n_viz), label='sri - orig')
# _ = plt.xscale('log')
_ = plt.legend()
_ = plt.grid('both', alpha=0.25)
_ = plt.xlabel('rank')
_ = plt.ylabel('# positive')
_ = plt.title(f'{target_str} - {args.n_pos=} - {args.n_neg=}')
show_plot(f'plots/{target_str}.png')