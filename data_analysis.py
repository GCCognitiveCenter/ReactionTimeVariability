# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 14:25:55 2026

Script to run the analyses for RTV. Receives as input the processed datasets, 
output of the script preprocessing.py

@author: EmanueleCiardo
"""
# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import exponnorm
from scipy.stats import linregress, t

#### Functions ####

#Prepocessing
def preproc_RTV(df): # Preprocessing routine for Reaction Time Variability

    # remove trial 1 to 7, since it's practice + 2 to remove
    # df = df[~df['trial'].isin(range(1, 4))]
    
    # Keep only pattern trials. This removes any SL
    # df = df[df['trialtype'] == 1]
    
    df = df[df['hit']==1] # df for Hits (correct responses)
    
    return df

def preproc_SL(df): # Preprocessing routine for Statistical Learning
    # remove trills and repetitions
    df = df[~df['triplet_type'].isin(['R','T'])]
    
    # remove trial 1 to 7, since it's practice + 2 to remove
    df = df[~df['trial'].isin(range(1, 8))]
    
    # remove extremely fast responses
    df = df[~(df['RT']<50)]
    
    # remove extremely slow responses
    df = df[~(df['RT']>1000)]
    
    # loop participants, calculate mean RT and SD, and remove trials +/- 3SD
    # compute per-ID mean and std for RT
    # grouped = df.groupby('ID')['RT']
    # means = grouped.transform('mean')
    # stds  = grouped.transform('std')
    
    # keep only values within ±3 SD per ID
    # mask = (df['RT'] >= means - 3 * stds) & (df['RT'] <= means + 3 * stds)
    
    # Alternative, MAD approach
    grouped = df.groupby('ID')['RT']
    medians = grouped.transform('median')
    mads    = grouped.transform(lambda x: (x - x.median()).abs().median())
    
    sigma_robust = 1.4826 * mads
    mask = (df['RT'] >= medians - 3 * sigma_robust) & (df['RT'] <= medians + 3 * sigma_robust)
    df = df[mask]
    
    df = df[df['hit']==1] # df for Hits (correct responses)
    
    return df


# Reaction Tme Variability 
def block_CV(df):  # Calculate RTV as Coefficient of Variation
    # One groupby + transform (no reset_index, no merge)
    g = df.groupby(['ID', 'block'])['RT']
    df = df.copy()
    df['cv_block'] = g.transform('std') / g.transform('mean')
    return df

def fit_exgauss(df,
               rt_col='RT',
               group_cols=('ID', 'block'),
               min_n=20):

    group_cols = list(group_cols)

    # work on just what we need (less data moved around)
    sub = df[group_cols + [rt_col]]

    rows = []
    for keys, g in sub.groupby(group_cols, sort=False):
        x = g[rt_col].to_numpy(dtype=float)
        x = x[~np.isnan(x)]

        if x.size < min_n:
            mu = sigma = tau = np.nan
        else:
            K, loc, scale = exponnorm.fit(x)
            mu = loc
            sigma = scale
            tau = K * scale

        # keys can be scalar or tuple depending on group_cols length
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append((*keys, mu, sigma, tau))

    df_exg = pd.DataFrame(rows, columns=group_cols + ['mu', 'sigma', 'tau'])

    # join back (one operation)
    return df.merge(df_exg, on=group_cols, how='left')

# Statistical Learning
def SL(df):
    df_SL_avg = df.groupby(['ID','block','triplet_type'])['RT'].mean().reset_index()
    
    df_SL_avg            = df_SL_avg.pivot(index=['ID','block'], columns='triplet_type', values='RT').reset_index()
    df_SL_avg['RT_diff'] = df_SL_avg['L'] - df_SL_avg['H']
    df_SL_avg = df_SL_avg.drop(['H', 'L'], axis=1)
    return df_SL_avg