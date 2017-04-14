__author__ = 'jharrington'
import pandas as pd
import numpy as np
import sys
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import model_selection as ms
from sklearn.model_selection import GridSearchCV

def read_data():
    df = pd.read_csv('credit.csv') # Please change this
    print df.head()
    # remove duplicates
    df = df.drop_duplicates()
    # remove rows with dependent variable missing
    df = df.dropna(subset=['TARGET'])
    return df

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

def main():
    # Step 1. Import data
    df = read_data()
    # Step 2. Explore data
    # 2.1. Get variable names
    col_names = df.columns.tolist()
    # 2.2. Classify variables into numeric, categorical (with strings), and nominal
    numeric,categorical,nominal = variable_type(df) 
    print "numeric:", numeric # ['ID', 'DerogCnt', 'CollectCnt', 'InqCnt06', 'InqTimeLast', 'InqFinanceCnt24', 'TLTimeFirst', 'TLTimeLast', 'TLCnt03', 'TLCnt12', 'TLCnt24', 'TLCnt', 'TLSum', 'TLMaxSum', 'TLSatCnt', 'TLDel60Cnt', 'TLBadCnt24', 'TL75UtilCnt', 'TL50UtilCnt', 'TLBalHCPct', 'TLSatPct', 'TLDel3060Cnt24', 'TLDel90Cnt24', 'TLDel60CntAll', 'TLOpenPct', 'TLBadDerogCnt', 'TLDel60Cnt24', 'TLOpen24Pct']
    print "categorical:", categorical # no categorical
    print "nominal:", nominal # ['TARGET', 'BanruptcyInd']
    # Your code here to drop the variable 'ID'
    # kept receiving KeyError: "['ID'] not in index" because trying to do a histogram on a column that has been removed.
    df.drop("ID", axis=1, inplace=True)
    numeric.remove("ID")
    # 2.3. Draw histogram for numeric variables
    # Your code here  - you can draw a histogram that includes multiple variables. See exploratary_analysis in toolbox
    df[numeric].hist()
    plt.show()
    # 2.4. Variables that have skewed distribution and need to be log transformed. You don't need to wrote code, just use the list below
    variables_needs_tranform = ['DerogCnt', 'CollectCnt', 'InqCnt06', 'InqTimeLast', 'InqFinanceCnt24', 'TLTimeFirst', 'TLTimeLast', 'TLCnt03', 'TLCnt12', 'TLCnt24', 'TLCnt', 'TLSum', 'TLMaxSum', 'TLDel60Cnt', 'TLBadCnt24', 'TL75UtilCnt', 'TL50UtilCnt', 'TLBalHCPct', 'TLSatPct', 'TLDel3060Cnt24', 'TLDel90Cnt24', 'TLDel60CntAll', 'TLBadDerogCnt', 'TLDel60Cnt24', 'TLOpen24Pct']
    # 2.5. Draw pie charts for nominal variables
    df.TARGET.value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.show()
    df.BanruptcyInd.value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.show()

    # Step 3. Transform variables
    '''
       your code here...
       You need to do log tranformation for the variables in the list variables_needs_tranform. You added the log of the varialbes, but don't need to remove the original variables        
    '''
    # columns have NaN values before missing value imputation
    for column in variables_needs_tranform:
        df["log_"+column] = np.log(df[column].fillna(0)+1)
    #print df.head()

    # 3.3 Missing value imputation
    # your code here: First do variable classification: classnumeric,categorical,nominal = 
    # your code here: Identify variables with missing values:
    variables_with_na = variable_with_missing(df) # your code here. Modify this
    numeric_with_na = variable_with_missing(df[numeric]) # your code here. Modify this. find numeric variables with missing values and add the variables to the list numeric_with_na
    nominal_with_na = variable_with_missing(df[nominal]) # your code here. Modify this. find nominal variable with missing values and add the variables to the list nominal_with_na. In our dataset, we have none.
    print numeric_with_na
    print nominal_with_na
    ''' 
    Your code here to do missing value imputation. For nummeric variables, replace missings with mean and add a missing value indicate. For nominal variables, if there are missing values, treat the missing values as a seperate categorical and do dummy coding.
    '''
    for column in numeric_with_na:
        num_missing_mean_median(df, column)
    dummy_coding_for_vars(df,nominal_with_na)

    # after transformation and missing value imputation, we clean our data. You don't need to change my code below
    independent_vars = ['DerogCnt', 'log_DerogCnt',  'CollectCnt', 'log_CollectCnt',  'BanruptcyInd', 'InqCnt06', 'log_InqCnt06',  'InqTimeLast', 'log_InqTimeLast',  'InqFinanceCnt24', 'log_InqFinanceCnt24',  'TLTimeFirst', 'log_TLTimeFirst',  'TLTimeLast', 'log_TLTimeLast',  'TLCnt03', 'log_TLCnt03',  'TLCnt12', 'log_TLCnt12',  'TLCnt24', 'log_TLCnt24',  'TLCnt', 'log_TLCnt',  'TLSum', 'log_TLSum',  'TLMaxSum', 'log_TLMaxSum', 'TLSatCnt', 'TLDel60Cnt', 'log_TLDel60Cnt',  'TLBadCnt24', 'log_TLBadCnt24',  'TL75UtilCnt', 'log_TL75UtilCnt', 'TL50UtilCnt', 'log_TL50UtilCnt',  'TLBalHCPct', 'log_TLBalHCPct',  'TLSatPct', 'log_TLSatPct', 'TLDel3060Cnt24', 'log_TLDel3060Cnt24',  'TLDel90Cnt24', 'log_TLDel90Cnt24',  'TLDel60CntAll', 'log_TLDel60CntAll', 'TLOpenPct', 'TLBadDerogCnt', 'log_TLBadDerogCnt',  'TLDel60Cnt24', 'log_TLDel60Cnt24', 'TLOpen24Pct', 'log_TLOpen24Pct', 'TLMaxSum_missing', 'TL50UtilCnt_missing', 'TLOpenPct_missing', 'TLBalHCPct_missing', 'TLSum_missing', 'TL75UtilCnt_missing', 'TLSatCnt_missing', 'TLCnt_missing', 'TLSatPct_missing', 'TLOpen24Pct_missing', 'InqTimeLast_missing']
    dependent_var = 'TARGET'

    # Step 4. Split data into training and test
    train_X = None
    train_y = None
    test_X = None
    test_y = None
    # I initialized these variable to None.
    # Your code here. Please split your data into training vs test (ratio: 80 : 20)
    train_X, test_X, train_y, test_y = train_test_split(df[independent_vars], df[dependent_var], test_size=0.2, random_state=123)

    # Step 5. Variable selection using SVC with linear kernal regression model (see toolbox).
    train_X_new = None 
    test_X_new = None
    # your code here.. train_X_new includes only the selected variables. 

    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear", cache_size=7000, max_iter=100)
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(train_y,5),
                  scoring='accuracy')
    selector = rfecv.fit(train_X, train_y)

    print selector.support_
    print selector.ranking_
    print("Optimal number of features : %d" % rfecv.n_features_)

    # You code here: You need to then also convert test_X into test_X_new, which only include the selected variables. Hint:you get the variable names of the selected variables from train_X_new(call this list "selected_variables"), then you can do test_X_new = test_X[[selected_variables]]
    train_X_new = selector.transform(train_X)
    test_X_new = selector.transform(test_X) # for the test dataset, you need to also keep just the selected varibles

    # Step 6. Model fitting and evaluation.
    # 6.1 fit logistic regression
    """
    Your code here.... You need to tune parameters first using 5-fold cross-validation (see INFS772_assign4.docx for details), and apply the final model to the test data set. Please remember to use the test dataset when evaluating the fitted model
    """
    # prepare cross validation folds
    num_folds = 5
    kfold = ms.StratifiedKFold(n_splits=num_folds)  
    parameters = {'C': np.arange(10,110,10)}  
    model = GridSearchCV(LogisticRegression(penalty='l2', class_weight="balanced"), parameters, cv=StratifiedKFold(train_y, 5), scoring= "accuracy")
    model.fit(train_X_new, train_y)
    # make predictions
    pred_y = model.predict(test_X_new)
    print "The logistic regression classification results:"
    """ Your code here ... you need to print metrics.classification_report"""
    for line in metrics.classification_report(test_y, pred_y).split("\n"):
        print line
    # 6.2 fit SVC.
    """
    Your code here.... You need to tune parameters first using 5-fold cross-validation (see INFS772_assign4.docx for details), and apply the final model to the test data set. Please remember to use the test dataset when evaluating the fitted model
    """
    # prepare cross validation folds
    num_folds = 5
    kfold = ms.StratifiedKFold(n_splits=num_folds)
    # in order to use RBF kernel, we need to standardize the training dataset. Actually, I would suggest to standardize data for all SVM algorithms
    from sklearn.preprocessing import StandardScaler    
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(train_X_new)
    X_test_std = scaler.fit_transform(test_X_new)
    # fit a SVM model to the data
    parameters = {'kernel':('linear','rbf','poly','sigmoid'), 'C': np.arange(.1,1.1,.1)}  
    model = GridSearchCV(SVC(class_weight='balanced', cache_size=7000, max_iter=100), parameters, cv=StratifiedKFold(train_y, 5), scoring= "accuracy")
    model.fit(X_train_std, train_y)
    # make predictions
    pred_y = model.predict(X_test_std)
    print "The SVC classification results:"
    """ Your code here ... you need to print metrics.classification_report"""
    print(metrics.classification_report(test_y, pred_y))


if __name__ == "__main__":
    main()










