# Covid19 Hospital Surge Prediction

Data: us-counties.csv

Counties chosen for inclusion based on having at least 28 consecutive days of data and no intermediate missing days. These counties, and their cases per day, are stored in: processed_counties.txt

The script that generated this was preprocess_data.py. The script is pretty well commented. Basically it just reads the csv data into a dict mapping county name to cases per day (it also does this per week). It then creates numpy arrays with the data (all but the last day) and target (the last day) per county. This script also maintains a list of labels, in case we want to look at the names corresponding to any county. 

Concretely, the task I set up was to predict the cases for the last day using as features the cases for previous days. You could extend this script by also reading in the demographic data file and adding those features to this vector. Or, you could try to split this into time series and non time series components.

Then, basic_classify.py was my initial experimentation script. It runs k-fold cross validation and applies three models: trivial (prediction for next day is sum of previous two days), linear regression, and random forest regression, outputting the mean absolute error for each method. 

Here's what I had written down as my next steps. It sounded like we were roughly on the same page.

Next steps:
- Implement or run "Washington model" and "epi model" and see how those compare to the base classifiers
- Consider different error metrics like mean percent error or mean percent increase error
- Add basic demographic info like population and population density; consider converting cases to "percent infected"
- Add cases / deaths in surrounding counties as features
- Predict deaths
- Predicting the next X days?


