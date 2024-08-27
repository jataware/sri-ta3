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
    
    assert (df.lon.apply(int) == df.lon).all()
    assert (df.lat.apply(int) == df.lat).all()
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

df = pd.read_feather(f'maps/{target_str}.feather')
df = prep_df(df)

# --
# Original train/test splits

root = Path("/home/paperspace/data/sri/maps/") / target_name

o_train = prep_df(pd.read_csv(root / 'train.csv'))
o_valid = prep_df(pd.read_csv(root / 'valid.csv'))
o_test  = prep_df(pd.read_csv(root / 'test.csv'))

df['o_train'] = df.hash.isin(o_train.hash)
df['o_valid'] = df.hash.isin(o_valid.hash)
df['o_test']  = df.hash.isin(o_test.hash)
df['o_sel']   = df.o_train | df.o_valid | df.o_test

# --
# Add SRI's likelihoods

L = tiffread(root / args.model_type / 'Likelihoods.tif')

# <<
# [HACK] should really do this with GDAL
from scipy.stats import linregress
lr_x = linregress(o_train.lon, o_train.x)
assert (lr_x.intercept + lr_x.slope * o_train.lon - o_train.x).abs().max() < 1e-10
df['c'] = (lr_x.intercept + lr_x.slope * df.lon).round().astype(int)

lr_y = linregress(o_train.lat, o_train.y)
assert (lr_y.intercept + lr_y.slope * o_train.lat - o_train.y).abs().max() < 1e-10
df['r'] = (lr_y.intercept + lr_y.slope * df.lat).round().astype(int)
# >>

df['sri_score'] = L[(df.r.values, df.c.values)]
df = df[df.sri_score.notnull()].reset_index(drop=True)

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

val_sel = np.setdiff1d(idx, pos_sel)
neg_sel = np.random.choice(np.setdiff1d(idx, pos_sel), args.n_neg, replace=False)

X_train = np.row_stack([
    X[pos_sel],
    X[neg_sel],
])

E_train = np.row_stack([
    E[pos_sel],
    E[neg_sel],
])

y_train = np.concatenate([
    np.ones(args.n_pos),
    np.zeros(args.n_neg),
])

y_valid = y[val_sel]

# --
# Fit model

rf_x  = RandomForestClassifier(n_estimators=1024, n_jobs=-1, verbose=1).fit(X_train, y_train)
rf_xc = RandomForestClassifier(n_estimators=1024, n_jobs=-1, verbose=1).fit(X_train[:,12::25], y_train)
rf_e  = RandomForestClassifier(n_estimators=1024, n_jobs=-1, verbose=1).fit(E_train, y_train)

p_x  = rf_x.predict_proba(X)[val_sel,1]
p_xc = rf_xc.predict_proba(X[:,12::25])[val_sel,1]
p_e  = rf_e.predict_proba(E)[val_sel,1]
p_o  = df.sri_score.values[val_sel]

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