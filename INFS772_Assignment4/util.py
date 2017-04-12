__author__ = 'jharrington'
import pandas as pd
import numpy as np


def variable_type(df, nominal_level = 5):
    categorical, numeric, nominal = [],[],[]
    for variable in df.columns.values:
        if np.issubdtype(np.array(df[variable]).dtype, int) or np.issubdtype(np.array(df[variable]).dtype, float):
            if len(np.unique(np.array(df[variable]))) <= nominal_level:
                nominal.append(variable)
            else:
                numeric.append(variable)
        else:
            categorical.append(variable)
    return numeric,categorical,nominal

def variable_with_missing(df):
    var_with_missing = []
    col_names = df.columns.tolist()
    for variable in col_names:
        percent = float(sum(df[variable].isnull()))/len(df.index)
        print variable+":", percent
        if percent != 0:
            var_with_missing.append(variable)
    return var_with_missing

def num_missing_mean_median(df, variable, prefix="", mean=True):
    indicator = ""
    if prefix=="":
        indicator = variable+ "_" + "missing"
    else:
        indicator = prefix + "_"+ "missing"
    df[indicator] = np.where(df[variable].isnull(),1,0)
    replaceValue = 0
    if mean== True:
        replaceValue = df[variable].mean()
    else:
        replaceValue = df[variable].median()
    df[variable].fillna(replaceValue, inplace= True)
    return df

def dummy_coding_for_vars(df, list_of_variables,  dummy_na=False, drop_first = False, prefix=None):
    if prefix==None:
        prefix = list_of_variables
    outputdata = pd.get_dummies(df, columns=list_of_variables, prefix= prefix, dummy_na=dummy_na, drop_first=drop_first)
    return outputdata
