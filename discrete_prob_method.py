# =======================================================================================================
# Table of Contents START
# =======================================================================================================

'''
1. Orientation
2. Imports
3. detect_anomaly_1
'''

# =======================================================================================================
# Table of Contents END
# Table of Contents TO Orientation
# Orientation START
# =======================================================================================================

'''
The purpose of this file is to create functions for detecting anomalies specifically with
questions from 'Discrete Probabilistic Methods'...
'''

# =======================================================================================================
# Orientation END
# Orientation TO Imports
# Imports START
# =======================================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =======================================================================================================
# Imports END
# Imports TO detect_anomaly_1
# detect_anomaly_1 START
# =======================================================================================================

def detect_anomaly_1(df, col_list):
    '''
    Gives a description and visual of a list of columns within a dataframe in order to
    detect anomalies using the 1 discrete variable method
    
    INPUT:
    df = Pandas dataframe
    col = List of columns to be described in the dataframe
    
    OUTPUT:
    NONE
    '''
    for col in col_list:
        count_df = pd.DataFrame(df[col].value_counts())
        pct_df = pd.DataFrame(df[col].value_counts(normalize=True))
        summary_df = pd.merge(count_df, pct_df, left_index=True, right_index=True)
        summary_df.rename(columns={summary_df.columns[0] : 'count', summary_df.columns[1] : 'percent'}, inplace=True)
        print(f'\033[32m=========={col} START==========\033[0m')
        print(summary_df)
        plt.hist(data=summary_df, x=summary_df.columns[0])
        plt.title(f'Distribution of {col}')
        plt.xlabel(f'{col}')
        plt.ylabel('Count')
        plt.show()
        print(f'\033[31m=========={col} END==========\033[0m')

# =======================================================================================================
# detect_anomaly_1 END
# detect_anomaly_1 TO detect_anomaly_2
# detect_anomaly_2 START
# =======================================================================================================



# =======================================================================================================
# detect_anomaly_2 END
# =======================================================================================================