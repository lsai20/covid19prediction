# Covid19 Hospital Surge Prediction

## Data 
us-counties.csv, county-level data and dates of interventions in data/

## Files
###### processed_counties.txt and preprocess_data.py

Counties chosen for inclusion based on having at least 28 consecutive days of data and no intermediate missing days. These counties, and their cases per day, are stored in: processed_counties.txt

The script that generated this was preprocess_data.py. The script is pretty well commented. Basically it just reads the csv data into a dict mapping county name to cases per day (it also does this per week). It then creates numpy arrays with the data (all but the last day) and target (the last day) per county. This script also maintains a list of labels, in case we want to look at the names corresponding to any county. 

###### basic_classify.py

Concretely, the task I set up was to predict the cases for the last day using as features the cases for previous days. You could extend this script by also reading in the demographic data file and adding those features to this vector. Or, you could try to split this into time series and non time series components.

Then, basic_classify.py was my initial experimentation script. It runs k-fold cross validation and applies three models: trivial (prediction for next day is sum of previous two days), linear regression, and random forest regression, outputting the mean absolute error for each method. 


###### plot_nyt_county.ipynb

Exploratory data analysis on county demographic data and interventions. (1) Plots cases and deaths over time, before and after social distancing, and color coded by population size bins. (2) Plots fold-increase in cases in last week against date of intervention. (3) Correlation between cases and other features.

###### model_nyt_county.ipynb

Fit different models, see which features have largest effects. Converts county data to array of numeric features and target. Currently has linear model, lasso, arima. [Need to add add option to predict log and count change.]



## Next steps
- Implement or run "Washington model" and "epi model" and see how those compare to the base classifiers
- (done) Add basic demographic info like population and population density; consider converting cases to "percent infected"
- Add cases / deaths in surrounding counties as features
- Predict deaths
- Predicting the next X days?
- Consider different error metrics like mean percent error or mean percent increase error
- Consider reporting both regular error and error weighted by county population
- Extend prediction script to predict with additional models, output error
- Test different combinations of models, features, and counties included (min days non-missing)
- Find appropriate parameters for time series models
- Compare targets: count, log(count), % of population, cases vs deaths
- Consider how to select which counties to predict, more timepoints vs more counties
- Consider how to handle outliers (remove or focus on predicting outlier/rapid growth?)
- Combine external data with arima (e.g. with arimax), add other time series models
- update to most recent day's count data
- Output predictions from a model as csv