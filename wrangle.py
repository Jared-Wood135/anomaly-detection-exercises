# =======================================================================================================
# Table of Contents START
# =======================================================================================================

'''
1. Orientation
2. Imports
3. acquire
4. prepare
5. wrangle
6. split
7. scale
8. sample_dataframe
9. remove_outliers_tukey
10. find_outliers_tukey
11. find_outliers_sigma
12. drop_nullpct
13. check_nulls
'''

# =======================================================================================================
# Table of Contents END
# Table of Contents TO Orientation
# Orientation START
# =======================================================================================================

'''
The purpose of this file is to create functions for both the acquire & preparation phase of the data
science pipeline or also known as 'wrangling' the data...
'''

# =======================================================================================================
# Orientation END
# Orientation TO Imports
# Imports START
# =======================================================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# =======================================================================================================
# Imports END
# Imports TO acquire
# acquire START
# =======================================================================================================

def acquire():
    '''
    Obtains the vanilla version of the mass_shooters dataframe

    INPUT:
    NONE

    OUTPUT:
    mass_shooters = pandas dataframe
    '''
    print('Acquire dat shit!')

# =======================================================================================================
# acquire END
# acquire TO prepare
# prepare START
# =======================================================================================================

def prepare():
    '''
    Takes in the vanilla mass_shooters dataframe and returns a cleaned version that is ready 
    for exploration and further analysis

    INPUT:
    NONE

    OUTPUT:
    .csv = ONLY IF FILE NONEXISTANT
    prepped_mass_shooters = pandas dataframe of the prepared mass_shooters dataframe
    '''
    if os.path.exists('mass_shooters.csv'):
        print('Prep dat shit!')
    else:
        print('Prep dat shit!')

# =======================================================================================================
# prepare END
# prepare TO wrangle
# wrangle START
# =======================================================================================================

def wrangle():
    '''
    Function that acquires, prepares, and splits the mass_shooters dataframe for use as well as 
    creating a csv.

    INPUT:
    NONE

    OUTPUT:
    .csv = ONLY IF FILE NONEXISTANT
    train = pandas dataframe of training set for mass_shooter data
    validate = pandas dataframe of validation set for mass_shooter data
    test = pandas dataframe of testing set for mass_shooter data
    '''
    if os.path.exists('mass_shooters.csv'):
        mass_shooters = pd.read_csv('mass_shooters.csv', index_col=0)
        train, validate, test = split(mass_shooters, stratify='shooter_volatility')
        return train, validate, test
    else:
        mass_shooters = prepare()
        mass_shooters.to_csv('mass_shooters.csv')
        train, validate, test = split(mass_shooters, stratify='shooter_volatility')
        return train, validate, test
    
# =======================================================================================================
# wrangle END
# wrangle TO split
# split START
# =======================================================================================================

def split(df, stratify=None):
    '''
    Takes a dataframe and splits the data into a train, validate and test datasets

    INPUT:
    df = pandas dataframe to be split into
    stratify = Splits data with specific columns in consideration

    OUTPUT:
    train = pandas dataframe with 70% of original dataframe
    validate = pandas dataframe with 20% of original dataframe
    test = pandas dataframe with 10% of original dataframe
    '''
    train_val, test = train_test_split(df, train_size=0.9, random_state=1349, stratify=df[stratify])
    train, validate = train_test_split(train_val, train_size=0.778, random_state=1349, stratify=train_val[stratify])
    return train, validate, test


# =======================================================================================================
# split END
# split TO scale
# scale START
# =======================================================================================================

def scale(train, validate, test, cols, scaler):
    '''
    Takes in a train, validate, test dataframe and returns the dataframes scaled with the scaler
    of your choice

    INPUT:
    train = pandas dataframe that is meant for training your machine learning model
    validate = pandas dataframe that is meant for validating your machine learning model
    test = pandas dataframe that is meant for testing your machine learning model
    cols = List of column names that you want to be scaled
    scaler = Scaler that you want to scale columns with like 'MinMaxScaler()', 'StandardScaler()', etc.

    OUTPUT:
    new_train = pandas dataframe of scaled version of inputted train dataframe
    new_validate = pandas dataframe of scaled version of inputted validate dataframe
    new_test = pandas dataframe of scaled version of inputted test dataframe
    '''
    original_train = train.copy()
    original_validate = validate.copy()
    original_test = test.copy()
    scale_cols = cols
    scaler = scaler
    scaler.fit(original_train[scale_cols])
    original_train[scale_cols] = scaler.transform(original_train[scale_cols])
    scaler.fit(original_validate[scale_cols])
    original_validate[scale_cols] = scaler.transform(original_validate[scale_cols])
    scaler.fit(original_test[scale_cols])
    original_test[scale_cols] = scaler.transform(original_test[scale_cols])
    new_train = original_train
    new_validate = original_validate
    new_test = original_test
    return new_train, new_validate, new_test

# =======================================================================================================
# scale END
# scale TO sample_dataframe
# sample_dataframe START
# =======================================================================================================

def sample_dataframe(train, validate, test):
    '''
    Takes train, validate, test dataframes and reduces the shape to no more than 1000 rows by taking
    the percentage of 1000/len(train) then applying that to train, validate, test dataframes.

    INPUT:
    train = Split dataframe for training
    validate = Split dataframe for validation
    test = Split dataframe for testing

    OUTPUT:
    train_sample = Reduced size of original split dataframe of no more than 1000 rows
    validate_sample = Reduced size of original split dataframe of no more than 1000 rows
    test_sample = Reduced size of original split dataframe of no more than 1000 rows
    '''
    ratio = 1000/len(train)
    train_samples = int(ratio * len(train))
    validate_samples = int(ratio * len(validate))
    test_samples = int(ratio * len(test))
    train_sample = train.sample(train_samples)
    validate_sample = validate.sample(validate_samples)
    test_sample = test.sample(test_samples)
    return train_sample, validate_sample, test_sample

# =======================================================================================================
# sample_dataframe END
# sample_dataframe TO remove_outliers_tukey
# remove_outliers_tukey START
# =======================================================================================================

def remove_outliers_tukey(df, col_list, k=1.5):
    '''
    Remove outliers from a dataframe based on a list of columns using the tukey method and then
    returns a single dataframe with the outliers removed

    INPUT:
    df = pandas dataframe
    col_list = List of columns that you want outliers removed
    k = Defines range for fences, default/normal is 1.5, 3 is more extreme outliers

    OUTPUT:
    df = pandas dataframe with outliers removed
    '''
    col_qs = {}
    for col in col_list:
        col_qs[col] = q1, q3 = df[col].quantile([0.25, 0.75])
    for col in col_list:
        iqr = col_qs[col][0.75] - col_qs[col][0.25]
        lower_fence = col_qs[col][0.25] - (k*iqr)
        upper_fence = col_qs[col][0.75] + (k*iqr)
        df = df[(df[col] > lower_fence) & (df[col] < upper_fence)]
    return df

# =======================================================================================================
# remove_outliers_tukey END
# remove_outliers_tukey TO find_outliers_tukey
# find_outliers_tukey START
# =======================================================================================================

def find_outliers_tukey(df, col_list, k=1.5):
    '''
    Find outliers from a dataframe based on a list of columns using the tukey method and then
    returns all of the values identifed as outliers

    INPUT:
    df = pandas dataframe
    col_list = List of columns that you want outliers removed
    k = Defines range for fences, default/normal is 1.5, 3 is more extreme outliers

    OUTPUT:
    NONE
    '''
    for col in col_list:
        lower_vals = []
        upper_vals = []
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - iqr * k
        upper_fence = q3 + iqr * k
        for val in df[col]:
            if val < lower_fence:
                lower_vals.append(val)
            elif val > upper_fence:
                upper_vals.append(val)
        print(f'\033[35m =========={col} // k={k}==========\033[0m')
        print(f'\033[32mValues < {lower_fence:.2f}: {len(lower_vals)} Values\033[0m')
        print(lower_vals)
        print(f'\n\033[32mValues > {upper_fence:.2f}: {len(upper_vals)} Values\033[0m')
        print(upper_vals)
        print(f'\n')

# =======================================================================================================
# find_outliers_tukey END
# find_outliers_tukey TO find_outliers_sigma
# find_outliers_sigma START
# =======================================================================================================

def find_outliers_sigma(df, col_list, sigma=2):
    '''
    Find outliers from a dataframe based on a list of columns using the three sigma rule and then
    returns all of the values identifed as outliers

    INPUT:
    df = pandas dataframe
    col_list = List of columns that you want outliers removed
    sigma = How many z-scores a value must at least be to identify as an outlier

    OUTPUT:
    NONE
    '''
    for col in col_list:
        mean = df[col].mean()
        std = df[col].std()
        z_scores = ((df[col] - mean) / std)
        outliers = df[col][z_scores.abs() >= sigma]
        print(f'\033[35m =========={col} // sigma={sigma}==========\033[0m')
        print(f'\033[32mMEAN:\033[0m {mean:.2f}')
        print(f'\033[32mSTD:\033[0m {std:.2f}')
        print(f'\033[32mOutliers:\033[0m {len(outliers)}')
        print(outliers)
        print(f'\n')

# =======================================================================================================
# find_outliers_sigma END
# find_outliers_sigma TO drop_nullpct
# drop_nullpct START
# =======================================================================================================

def drop_nullpct(df, percent_cutoff):
    '''
    Takes in a dataframe and a percent_cutoff of nulls to drop a column on
    and returns the new dataframe and a dictionary of dropped columns and their pct...
    
    INPUT:
    df = pandas dataframe
    percent_cutoff = Null percent cutoff amount
    
    OUTPUT:
    new_df = pandas dataframe with dropped columns
    drop_null_pct_dict = dict of column names dropped and pcts
    '''
    drop_null_pct_dict = {
        'column_name' : [],
        'percent_null' : []
    }
    for col in df:
        pct = df[col].isna().sum() / df.shape[0]
        if pct > 0.20:
            df = df.drop(columns=col)
            drop_null_pct_dict['column_name'].append(col)
            drop_null_pct_dict['percent_null'].append(pct)
    new_df = df
    return new_df, drop_null_pct_dict

# =======================================================================================================
# drop_nullpct END
# drop_nullpct TO check_nulls
# check_nulls START
# =======================================================================================================

def check_nulls(df):
    '''
    Takes a dataframe and returns a list of columns that has at least one null value
    
    INPUT:
    df = pandas dataframe
    
    OUTPUT:
    has_nulls = List of column names with at least one null
    '''
    has_nulls = []
    for col in df:
        nulls = df[col].isna().sum()
        if nulls > 0:
            has_nulls.append(col)
    return has_nulls

# =======================================================================================================
# check_nulls END
# =======================================================================================================