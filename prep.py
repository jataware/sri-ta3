#!/usr/bin/env python

"""
    prep.py
"""

import os
import argparse
import numpy as np
import pandas as pd
from tifffile import imread as tiffread
from pathlib import Path

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
    
    args.target_str, args.target_name = TARGETS[args.target_idx]
    args.model_type = 'MAE'
    print(f'target={args.target_str, args.target_name}')
    return args

args = parse_args()
np.random.seed(args.seed)
os.makedirs('maps_prepped', exist_ok=True)

inpath  = f'maps/{args.target_str}.feather'
outpath = f'maps_prepped/{args.target_str}.feather'

# --
# Load features (computed by Jataware)

df = pd.read_feather(inpath)
df = prep_df(df)

# --
# Original train/test splits

root = Path("/home/paperspace/data/sri/maps/") / args.target_name

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
# [HACK] should really do this with GDAL ... but I hate GDAL
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

df.to_feather(outpath, compression='zstd')