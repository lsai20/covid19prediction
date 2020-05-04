import numpy as np
import sklearn as sk
from sklearn import datasets, linear_model, model_selection, metrics
import pandas as pd
import datetime


import pmdarima as pm
from pmdarima import model_selection

import compare_models_functions as myfxns





if __name__ == '__main__':

    # use counties with at least this many days (keep in mind last 7+ for testing)
    min_days = 28

    cases_or_deaths = 'cases'
    #use_log_counts = True

    pred_seq_len_test = 7
    pred_seq_len_future = 30


    cases_df, deaths_df = myfxns.load_county_datasets()
    # dict of fips to county, state (capitalized)
    fips2countystateD = myfxns.make_fips2countystateD()
    # ARIMA for each county, treating counties as independent 
    # note: autoarima has a lot of output, scroll to plot at bottom of cell output

    num_counties = len(cases_df)
    counts_mask = [str(c)[:5] == '2020-' for c in cases_df.columns]


    #if use_log_counts: # TODO train w and wo log counts, exponentiate
    #    data = np.log(data)


    # col 0 of preds is fips, rest are predictions
    preds_test = -1.0*np.ones((num_counties, 1+pred_seq_len_test))
    preds_test[:,0] = cases_df['fips']
    preds_future = -1.0*np.ones((num_counties, 1+pred_seq_len_future))
    preds_future[:,0] = cases_df['fips']



    for i in range(num_counties):
        if i % 50 == 0:
            print('arima on county i = %d' % i)

        if cases_or_deaths == 'cases': # TODO could move np.array() outside for loop, but arima by far slowest step
            data = np.array(cases_df.iloc[i,counts_mask].values, dtype='float')
        else:
            data = np.array(deaths_df.iloc[i,counts_mask].values, dtype='float')

        # data should be a 1d array of counts over time
        # start from first date with no missing values (different for each county)
        # first_nonmiss = np.where(~np.isnan(data))[0][0] # counties can miss intermediate days
        if np.isnan(data).any():
            last_miss = np.where(np.isnan(data))[0][-1]
            data = data[(last_miss+1):] 

        if data.shape[0] < min_days: # don't predict if not at least total of min days available
            preds_test[i,1:] = np.nan 
            preds_future[i,1:] = np.nan
            continue

        # first train excluding last 7 days to test
        train, test = model_selection.train_test_split(data, test_size=pred_seq_len_test)
        # auto_arima automatically chooses parameters if not specified. can slow if d not set
        # m = 1, means non-seasonal. Setting disp, trace, suppress_warnings to silent
        # will fit arma(0,0,0) if input time series is const
        arima_fitted = pm.auto_arima(train, error_action='ignore', disp=-1,
                              suppress_warnings=True, maxiter=50, m=1)
        #print('Selected ARIMA order', arima_fitted.get_params()['order'])
        # when training on raw counts, selected d was usually 1 or 2

        # forecast
        preds_test[i,1:] = arima_fitted.predict(n_periods=pred_seq_len_test)


        ### repeat using all data to forecast
        arima_fitted = pm.auto_arima(data, error_action='ignore', trace=False,
                              suppress_warnings=True, maxiter=50, m=1, disp=-1)
        preds_future[i,1:] = arima_fitted.predict(n_periods=pred_seq_len_future)
    
    # TODO could mask preds if don't want to output counties with nan predictions
    
    # output test and future predictions
    startDate = datetime.date(2020, 4, 24) # date of first prediction
    endDate = startDate + datetime.timedelta(days = 7)
    fname = 'prediction_csv/test_arima_trained_raw_counts.csv'
    myfxns.output_csv_preds(preds_test, fips2countystateD, startDate, endDate, fname, 
                         convertLog=False) #, startCol=-30, endCol=None)

    startDate = datetime.date(2020, 5, 3) # date of first prediction
    endDate = startDate + datetime.timedelta(days = 30)
    fname = 'prediction_csv/future_arima_trained_raw_counts.csv'
    myfxns.output_csv_preds(preds_future, fips2countystateD, startDate, endDate, fname, 
                         convertLog=False) #, startCol=-30, endCol=None)


