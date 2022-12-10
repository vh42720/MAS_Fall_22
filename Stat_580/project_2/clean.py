# standard imports
import os

# extra imports
import numpy as np
import pandas as pd

from utils.helper_functions import load_config


# load config file
config = load_config()


# read in raw data
# add neighborhood indicator and KitchenAbvGr for CollegeCr dataset
df_college = pd.read_csv(os.path.join(config['path']['raw'], 'CollegeCr.csv'))
df_college['neighborhood'] = 'CollegeCr'
df_college['KitchenAbvGr'] = np.nan
df_college['uniqueID'] = 'train'

df_college_test = pd.read_csv(os.path.join(config['path']['raw'], 'CollegeCr.test.csv'))
df_college_test['neighborhood'] = 'CollegeCr'
df_college_test['KitchenAbvGr'] = np.nan

# add neighborhood indicator and BsmtUnfSF for Edwards dataset
df_edwards = pd.read_csv(os.path.join(config['path']['raw'], 'Edwards.csv'))
df_edwards['neighborhood'] = 'Edwards'
df_edwards['BsmtUnfSF'] = np.nan
df_edwards['uniqueID'] = 'train'

df_edwards_test = pd.read_csv(os.path.join(config['path']['raw'], 'Edwards.test.csv'))
df_edwards_test['neighborhood'] = 'Edwards'
df_edwards_test['BsmtUnfSF'] = np.nan

# add neighborhood indicator for OldTown dataset
df_oldtown = pd.read_csv(os.path.join(config['path']['raw'], 'OldTown.csv'))
df_oldtown['neighborhood'] = 'OldTown'
df_oldtown['uniqueID'] = 'train'

df_oldtown_test = pd.read_csv(os.path.join(config['path']['raw'], 'OldTown.test.csv'))
df_oldtown_test['neighborhood'] = 'OldTown'


# full data set
df_lst = [df_college, df_edwards, df_oldtown, df_college_test, df_edwards_test, df_oldtown_test]
df_full = pd.concat(df_lst)
df_full.head()