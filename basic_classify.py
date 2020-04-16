import preprocess_data
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def set_prediction_bounds(test_data, num_days=1):
	min_bound = [i[-1] for i in test_data]  # set min bound to previous day total
	max_bound = [i[-1] * (2 ** num_days) for i in test_data]  # set limit to doubling every day
	return min_bound, max_bound


# Trivial baseline: prediction for next day/week is sum of previous two
def run_trivial_classifier(all_sets):
	train_data, test_data, train_targets, test_targets, train_labels, test_labels = all_sets
	preds = []
	for i in range(len(test_data)):
		preds.append(test_data[i][-1] + test_data[i][-2])
	preds = np.array(preds)
	return preds


def run_linear_regression(all_sets):
	train_data, test_data, train_targets, test_targets, train_labels, test_labels = all_sets
	regr = LinearRegression()
	regr.fit(train_data, train_targets)
	preds = regr.predict(test_data)
	return preds


def run_random_forest_regression(all_sets):
	train_data, test_data, train_targets, test_targets, train_labels, test_labels = all_sets
	regr = RandomForestRegressor(max_depth=20, random_state=0)
	regr.fit(train_data, train_targets)
	preds = regr.predict(test_data)
	return preds


def mse_eval(preds, targets, min_bound, max_bound):
	preds = np.array([min(max(preds[i], min_bound[i]), max_bound[i]) for i in range(len(preds))])  # bound the preds
	preds = preds.astype(int)  # cast float predictions to int
	mse = mean_squared_error(preds, targets)
	# print('MSE for ' + name + ': ' + "{:.2f}".format(mse))
	return mse


def mean_abs_err_eval(preds, targets, min_bound, max_bound):
	preds = np.array([min(max(preds[i], min_bound[i]), max_bound[i]) for i in range(len(preds))])  # bound the preds
	preds = preds.astype(int)  # cast float predictions to int
	mae = mean_absolute_error(preds, targets)
	# print('Mean absolute error for ' + name + ': ' + "{:.2f}".format(mae))
	return mae


def run_train_test(data, targets, labels):
	all_sets = train_test_split(data, targets, labels, test_size=0.2)
	train_data, test_data, train_targets, test_targets, train_labels, test_labels = all_sets
	min_bound, max_bound = set_prediction_bounds(test_data, num_days=1)
	triv_preds = run_trivial_classifier(all_sets)
	triv_mae = mean_abs_err_eval(triv_preds, test_targets, min_bound, max_bound)
	print('Mean absolute error for Trivial Predictor: ' + "{:.2f}".format(triv_mae))
	lr_preds = run_linear_regression(all_sets)
	lr_mae = mean_abs_err_eval(lr_preds, test_targets, min_bound, max_bound)
	print('Mean absolute error for Linear Regression: ' + "{:.2f}".format(lr_mae))
	rf_preds = run_random_forest_regression(all_sets)
	rf_mae = mean_abs_err_eval(rf_preds, test_targets, min_bound, max_bound)
	print('Mean absolute error for Random Forest Regression: ' + "{:.2f}".format(rf_mae))


def run_kfold(data, targets, labels):
	num_folds = 5
	kf = KFold(n_splits=num_folds, shuffle=True)
	kf.get_n_splits(data)
	triv_total, lr_total, rf_total = 0.0, 0.0, 0.0
	for train_index, test_index in kf.split(data):
		train_data, test_data = data[train_index], data[test_index]
		train_targets, test_targets = targets[train_index], targets[test_index]
		train_labels, test_labels = labels[train_index], labels[test_index]
		all_sets = [train_data, test_data, train_targets, test_targets, train_labels, test_labels]
		min_bound, max_bound = set_prediction_bounds(test_data, num_days=1)
		triv_preds = run_trivial_classifier(all_sets)
		triv_mae = mean_abs_err_eval(triv_preds, test_targets, min_bound, max_bound)
		triv_total += triv_mae
		lr_preds = run_linear_regression(all_sets)
		lr_mae = mean_abs_err_eval(lr_preds, test_targets, min_bound, max_bound)
		lr_total += lr_mae
		rf_preds = run_random_forest_regression(all_sets)
		rf_mae = mean_abs_err_eval(rf_preds, test_targets, min_bound, max_bound)
		rf_total += rf_mae

	triv_avg = triv_total / num_folds
	lr_avg = lr_total / num_folds
	rf_avg = rf_total / num_folds
	print('Mean absolute error for Trivial Predictor: ' + "{:.2f}".format(triv_avg))
	print('Mean absolute error for Linear Regression: ' + "{:.2f}".format(lr_avg))
	print('Mean absolute error for Random Forest Regression: ' + "{:.2f}".format(rf_avg))


if __name__ == '__main__':
	# load the data into np arrays via my preprocess_data library
	county_day_csv = 'us-counties.csv'  # csv file with raw data
	min_data_days = 28  # minimum number of days of data required to use the county
	day_data = preprocess_data.read_csv_into_dict(county_day_csv, min_data_days)
	week_data = preprocess_data.convert_day_data_to_weeks(day_data)
	# del day_data['New York City | New York | '] ; del week_data['New York City | New York | ']
	np_day_data, np_day_targets, np_day_labels = preprocess_data.convert_dict_to_numpy(day_data)
	np_week_data, np_week_targets, np_week_labels = preprocess_data.convert_dict_to_numpy(week_data)

	# split into training and test sets: [train_data, test_data, train_targets, test_targets, train_labels, test_labels]
	# run_train_test(np_day_data, np_day_targets, np_day_labels)

	# run kfold cross validation
	run_kfold(np_day_data, np_day_targets, np_day_labels)
