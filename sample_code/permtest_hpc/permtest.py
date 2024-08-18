import copy
import random
import math
import numpy as np
import pandas as pd
from mbpls.mbpls import MBPLS
from sklearn.utils import check_array
from numpy import linalg as la
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from datetime import datetime

now0 = datetime.now()
print('Starts at:', now0)

# %% Load Data
Y = pd.read_csv('../../sample_data/alz_metadata.csv')
Y.insert(0, 'output_var', Y["Gender"].apply(lambda x: 1 if x == 'Female' else 0))

HPOS = pd.read_csv('../../sample_data/alz_hpos.csv')
LPOS = pd.read_csv('../../sample_data/alz_lpos.csv')
LNEG = pd.read_csv('../../sample_data/alz_lneg.csv')


# %% Load Functions
def vip_multiblock(x_weights, x_super_scores, x_loadings, y_loadings):
    """
    Variable importance in projection (VIP) for multiblock PLS model
    :param x_weights: predictors' weights
    :param x_super_scores: predictors' super scores (i.e. for all blocks)
    :param x_loadings: predictors' loadings
    :param y_loadings: outcome variables' loadings
    :return: VIP score for each variable
    """

    # stack the weights from all blocks
    weights = np.vstack(x_weights)
    # calculate product of sum of squares of super scores and y loadings
    sum_squares = np.sum(x_super_scores ** 2, axis=0) * np.sum(y_loadings ** 2, axis=0)
    # p = number of variables - stack the loadings from all blocks
    p = np.vstack(x_loadings).shape[0]
    # VIP is a weighted sum of squares of PLS weights
    vip_scores = np.sqrt(p * np.sum(sum_squares * (weights ** 2), axis=1) / np.sum(sum_squares))
    return vip_scores


def vip_permutations(_, x_blocks, y, n_components=2, **kwargs):

    pls = MBPLS(n_components, **kwargs)
    x = copy.deepcopy(x_blocks)
    y = y.copy()

    y_perm = np.random.permutation(y)

    pls.fit(x, y_perm)

    vip = vip_multiblock(x_weights=pls.W_, x_super_scores=pls.Ts_, x_loadings=pls.P_, y_loadings=pls.V_)

    return vip


# %% Set parameters
# Number of CPU cores utilised (must be identical to PBS file)
ncpus = 200
# Number of latent variables:
lvs = 6
# Number of permutations:
perms = 1000000
# Predictors:
x = [HPOS, LPOS, LNEG]
# Response:
y = Y.output_var

now = datetime.now()

# %% Fit observed model and permutations

# Fit observed model
mbpls = MBPLS(n_components=lvs)
mbpls.fit(x, y)
vip_obs = vip_multiblock(x_weights=mbpls.W_, x_super_scores=mbpls.Ts_, x_loadings=mbpls.P_, y_loadings=mbpls.V_)
vip_observed = vip_obs[:, np.newaxis]

# Load Permutations
vip_perm = Parallel(n_jobs=ncpus)(delayed(vip_permutations)(i, x, y, n_components=lvs) for i in range(perms))
vip_null = np.stack(vip_perm, axis=1)

# Calculate empirical p-values
vip_greater = np.sum(vip_null >= vip_observed, axis=1)

p_vals = vip_greater/perms

# Calculate total run time
later = datetime.now()
difference = later - now
print('Permutations run time:', difference)




# %% Save outcomes

# Save p-values
np.savetxt("../../sample_data/p-values_classification.csv", p_vals, delimiter=",")

later = datetime.now()
print('Finished at:', later)

difference = later - now0
print('Total run time:', difference)

