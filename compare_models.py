

import numpy as np
import scipy as sc
import sklearn as sk
from sklearn import datasets, linear_model, model_selection
import pandas as pd
import datetime as dt

import basic_classify as bc

np.set_printoptions(precision=2)
plt.rcParams['figure.figsize'] = [12, 8]


# TODO run rf, run LM, run LM with counts only
# TODO fit on count data only vs all, but on same subset of counties (those with 28 days)

# TODO add mse pred for time series models (forecast only 1 day and compare?)
#preds = basic_classify.run_random_forest_regression(all_sets):
#train_data, test_data, train_targets, test_targets, train_labels, test_labels = all_sets



#cases_or_deaths = 'cases' # TODO not predicting death yet
# X is array of features, will correspond to 'data'. TODO if dtype is object, x may be list
    # y is array 'target'


# note: should be put somewhere and imported since this code also used for plotting
def load_county_datasets():
    ''' Load data into pd dataframes '''

    # Replace non-ascii chars in files (some county names have accent mark)
    counts_f = 'data/nyt-us-counties-cases-deaths.csv'
    with open(counts_f) as f:
        lines = f.readlines()
    with open(counts_f,'w') as f: # overwrite old file
        for line in lines:
            f.write(''.join([c if ord(c) < 128 else '_' for c in line]) )

    # nyt case and death counts, where each date-county pair is a row
    nyt_counts_df = pd.read_csv(counts_f)
    nyt_counts_df['date'] = pd.to_datetime(nyt_counts_df['date'])

    # load additional county-level data to use as features
    interventions_df = pd.read_csv('data/jieyingwu_dates_interventions.csv') # ordinal dates
    demographics_df = pd.read_csv('data/jieyingwu_demographics_counties.csv') # pop size, income, temp, etc


    # make pivot df where each county is a row, with dates as columns
    cases_df = pd.pivot_table(nyt_counts_df, values = 'cases', 
        index=['fips','county','state'], columns = 'date').reset_index()
    county_cols = cases_df.iloc[:,:3]


    # merge in additional features (case/death counts will be rightmost columns)
    cases_df = interventions_df.merge(cases_df, left_on='FIPS', right_on='fips', 
        suffixes=('_interventions', '_cases')) 
    cases_df = demographics_df.merge(cases_df, left_on='FIPS', right_on='FIPS', 
        suffixes=('_demog', '_cases')) 

    #  # first 3 cols will be fips, county, state. rest are numeric data
    not_fips_mask = [str(c).lower() != 'fips' for c in cases_df.columns]
    cases_df = cases_df.loc[:,not_fips_mask] # remove redundant fips
    cases_df = cases_df.select_dtypes(include='number') # numeric features only
    cases_df = county_cols.join(cases_df)

    # repeat for deaths
    deaths_df = pd.pivot_table(nyt_counts_df, values = 'deaths', 
        index=['fips','county','state'], columns = 'date').reset_index()
    deaths_df = interventions_df.merge(deaths_df, left_on='FIPS',
     right_on='fips',suffixes=('_interventions', '_deaths')) 
    deaths_df = demographics_df.merge(deaths_df, left_on='FIPS', 
        right_on='FIPS', suffixes=('_demog', '_cases')) 

    not_fips_mask = [str(c).lower() != 'fips' for c in deaths_df.columns]
    deaths_df = deaths_df.loc[:,not_fips_mask] # remove redundant fips
    deaths_df = deaths_df.select_dtypes(include='number') # numeric features only
    deaths_df = county_cols.join(deaths_df)

    return cases_df, deaths_df





def filter_and_scale_data(counts_df, min_days=0, max_frac_missing=0.1, 
    use_rel_counts=False, use_log_counts = False, use_counts_only = False):
    '''filter counties and features with missing data.
    return df of features, and array of last day's data (target).

    settings (if this code used later, could make object with members for each model/dataset
    min_days: only use counties which have data for the last min_days
    max_frac_missing: after filtering by min_days, only keep features with at most max_frac_missing
    is_pop_scale: whether to scale cases by population
    use_rel_counts: whether to pred counts/pop_size
    use_log_counts: whether to use log(counts), or log(counts/pop), as features and target
    use_counts_only # whether to use only counts as features

    Note: X_df is masked for counties with missing values.
    '''

    X_df = None        # X_df will contain only counties and features which pass filters
    if cases_or_deaths == 'cases':
        X_df = cases_df.copy()
    else:
        X_df = deaths_df.copy()

    counts_mask = [str(c)[:5] == '2020-' for c in X_df.columns] # bool which cols are counts

    if use_rel_counts:
        pop_estimates = X_df['POP_ESTIMATE_2018'].values[:,None]
        X_df.iloc[:,counts_mask] = np.divide(1.0*X_df.iloc[:,counts_mask].values, pop_estimates)
        
    if use_log_counts:
        X_df.iloc[:,counts_mask] = np.log(1.0*X_df.iloc[:,counts_mask].values)

    if use_counts_only:
        county_cols = cases_df.iloc[:,:3].copy()
        counts_mask = [str(c)[:5] == '2020-' for c in X_df.columns]
        X_df = X_df.iloc[:,counts_mask]
        X_df = county_cols.join(X_df) # stick row names back on


    # keep counties which have data from past 28 (or 0, etc) days
    end = X_df.columns[-1] # target is last day
    start = end - pd.DateOffset(min_days)
    X_df = X_df[~X_df.loc[:,start:end].isna().any(axis=1)]

    # keep columns which are >10% missing (lose a lot of prev dates if not filtering counties first)
    keep_cols = X_df.columns[X_df.isnull().mean() < max_frac_missing]
    X_df = X_df[keep_cols] 

    # make target y (last day) and exclude from X_df
    y = X_df.iloc[:,-1].values
    y = y.reshape((-1,1))
    X_df = X_df.iloc[:,:-1]

    # mask counties with any missing values
    # if any, could also mask counties w missing y here
    X = X_df.iloc[:,3:].values
    mask = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1) & ~np.isneginf(X).any(axis=1) 

    return X_df[mask,:], X, y


def printTopCoefs(values, columns, nTop=None, includeBottom=False, maxPad=70):
    '''given values and corresponding descriptions in order, print top values'''
    col_valuesL = list( zip(list(columns), values.flatten().tolist()) )
    col_valuesL.sort(key=lambda tup: abs(tup[1]), reverse=True)

    p = max( [len(str(c)) for c in columns] ) + 1 # padding for print
    p = min(p, maxPad)
    if nTop == None:
        nTop = len(columns)
    for col, effect in col_valuesL[:nTop]:
        print('%s\t%10.5f' % ((str(col).ljust(p), effect)) )
    if includeBottom and nTop*2 > len(columns): 
        # print remaining entries without duplicates
        for col, effect in col_valuesL[nTop:]:
            print('%s\t%10.5f' % ((str(col).ljust(p), effect)) )
    elif includeBottom:
        print('...')
        for col, effect in col_valuesL[-nTop:]:
            print('%s\t%10.5f' % ((str(col).ljust(p), effect)) )        
    return



def make_models(X, y):
    '''given dataset, fit some models, return list of models and descriptions'''
    # currently hardcoded
    # could add parameters in list and automatically make name and fit in for loop
    # could run it twice with different X and concat for models using subset of features (ex count data only)

    '''
    regr = linear_model.LinearRegression() 
    regr.fit(X_count, y)

    '''

    regrs = []
    names = []
    # TODO is regr copy or alias? need copy in for llop?

    name = 'OLS w demographics interventions counts'
    regr = linear_model.LinearRegression() 
    regr.fit(X, y)
    regrs.append(regr); names.append(name)

    name = 'LASSO_alpha1.0_max_iter5000'
    regr = linear_model.Lasso(alpha=1.0,max_iter=5000)
    regr.fit(X, y)
    regrs.append(regr); names.append(name)

    name = 'LASSO_alpha1.0_max_iter15k'
    regr = linear_model.Lasso(alpha=1.0,max_iter=15000)
    regr.fit(X, y)
    regrs.append(regr); names.append(name)

    
    name = 'LASSO_alpha1.0_max_iter50k'
    regr = linear_model.Lasso(alpha=1.0,max_iter=50000)
    regr.fit(X, y)
    regrs.append(regr); names.append(name)
    
    name = 'LASSO_alpha1.0_max_iter15k_positive'
    regr = linear_model.Lasso(alpha=1.0, max_iter=15000, positive=True)
    regr.fit(X, y)
    regrs.append(regr); names.append(name)

    name = 'LASSO_alpha0.0001_max_iter5000'
    regr = linear_model.Lasso(alpha=0.0001, max_iter=15000)
    regr.fit(X, y)
    regrs.append(regr); names.append(name)

    print(regrs)

return regrs, names





# edited from basic_classify.run_kfold to take list of sklearn models
def run_kfold2(regrs, names, data, targets, labels):
    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True)
    kf.get_n_splits(data)
    maeD = {} # maeD[name] = total for mean abs err / num folds
        for model_name in names:
            maeD[model_name] = 0.0

    for train_index, test_index in kf.split(data):
        train_data, test_data = data[train_index], data[test_index]
        train_targets, test_targets = targets[train_index], targets[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]
        all_sets = [train_data, test_data, train_targets, test_targets, train_labels, test_labels]
        min_bound, max_bound = bc.set_prediction_bounds(test_data, num_days=1)

        for regr, name in zip(regrs, names):
            preds = regrs.predict(test_data)
            maeD[name] += bc.mean_abs_err_eval(preds, test_targets, min_bound, max_bound)

    for regr, name in zip(regrs, names):
        maeD[name] = maeD[name]/num_folds
        print('Mean absolute error for %s ' % name + "{:.2f}".format(triv_avg))


if __name__ == '__main__':

    cases_df, deaths_df = load_county_datasets()

    # make dataset including demographics, etc (can repeat with other settings)
    print('Dataset including raw counts, county level data and interventions')
    X_df, X, y = filter_and_scale_data(cases_df, min_days=0, max_frac_missing=0.1, 
        use_rel_counts=False, use_log_counts = False, use_counts_only = False)

    # train models
    regrs, names = make_models(X, y)
    
    # run kfold cross validation
    run_kfold2(regrs, names, np_day_data, np_day_targets, np_day_labels)


