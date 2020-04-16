import numpy as np


def get_days_since_jan21(date, days_per_month):
	"""
	Here we take a date in the format yyyy-mm-dd and get the days since Jan. 21, where the dataset starts.
	This does so by counting back through the months, adding the number of days, until reaching January,
		and then takes the difference from 21.
	"""
	month = int(date.split('-')[1])
	day = int(date.split('-')[2])
	days_since_jan21 = 0
	while month > 1:
		days_since_jan21 += day
		month -= 1
		day = days_per_month[month]
	days_since_jan21 += day - 21
	return days_since_jan21


def read_csv_into_dict(county_day_csv, min_data_days):
	"""
	Read the csv data into a dict mapping county --> list of cases by day, where the index is days since Jan 21.
	:param county_day_csv: The csv file with the raw data in the format: date,county,state,fips,cases,deaths
	:return: a dict mapping county --> list of cases by day, where the index is days since Jan 21.
	"""
	day_data = {}
	days_per_month = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
	with(open(county_day_csv, 'r')) as infile:
		# Format: date,county,state,fips,cases,deaths
		infile.readline()  # skip header
		for line in infile:
			splits = line.strip().split(',')
			date, county, state, fips, cases, deaths = splits
			if county == 'Unknown':
				continue
			cases, deaths = int(cases), int(deaths)
			days_since_jan21 = get_days_since_jan21(date, days_per_month)
			county_name = county + ' | ' + state + ' | ' + fips  # a uniquely-identifying name for the county
			if county_name not in day_data:  # fill in zeros for days before cases appeared in this county
				day_data[county_name] = [0 for i in range(days_since_jan21)]
			day_data[county_name].append(cases)
	# clean out counties with missing days
	max_days = max([len(day_data[i]) for i in day_data])
	remove_list = [i for i in day_data if len(day_data[i]) < max_days]
	for county_name in remove_list:
		del day_data[county_name]
	# clean out counties with less than four weeks of data
	remove_list = [i for i in day_data if day_data[i][max_days - min_data_days] == 0]
	for county_name in remove_list:
		del day_data[county_name]
	return day_data


def convert_day_data_to_weeks(day_data):
	"""
	Aggregates individual day data into weeks.
	:param day_data: a dict mapping county --> list of cases by day, where the index is days since Jan 21.
	:return: a dict mapping county --> list of cases by week, where the index is weeks since Jan 21.
	"""
	week_data = {}
	for county in day_data:
		num_weeks = int(len(day_data[county]) / 7)
		week_data[county] = []
		for i in range(num_weeks):
			day_start, day_end = i * 7, (i + 1) * 7
			week_data[county].append(sum(day_data[county][day_start : day_end]))
	return week_data


def convert_dict_to_numpy(dict_data):
	"""
	Converts the dictionary-stored data to a numpy data structure for compatibility with ML libraries.
	:param dict_data: dict storing county mapped to cases per day or week
	:return:
		data, a 2d numpy array of the features, e.g. the cases list except for the last entry
		target, a 1d numpy array of target values to predict, e.g. the cases on the last day or week
		labels, an ordered list of labels corresponding to the numpy indices, for reporting later
	"""
	data, target, labels = [], [], []
	for county in dict_data:
		data.append(dict_data[county][:-1])
		target.append(dict_data[county][-1])
		labels.append(county)
	data, target, labels = np.array(data), np.array(target), np.array(labels)
	return data, target, labels


if __name__ == '__main__':
	county_day_csv = 'us-counties.csv'  # csv file with raw data
	min_data_days = 28  # minimum number of days of data required to use the county
	day_data = read_csv_into_dict(county_day_csv, min_data_days)
	week_data = convert_day_data_to_weeks(day_data)
	np_day_data, np_day_targets, np_day_labels = convert_dict_to_numpy(day_data)
	np_week_data, np_week_targets, np_week_labels = convert_dict_to_numpy(week_data)

	print(str(len(week_data)) + ' counties found with sufficient data, e.g. they have no intermediate missing days' +
							' and have at least ' + str(min_data_days) + ' total days of data.\n')
	for county in week_data:
		print(county + ': ')
		print(str(len(day_data[county])) + ' ' + str(day_data[county]))
		print(str(len(week_data[county])) + ' ' + str(week_data[county]) + '\n\n')
