import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

def cleanTest(df):
	df.drop(['survey_date', 'amount_given_mpesa', 'amount_received_mpesa', 'early_survey', 'ent_employees', 'day_of_week', 'med_vacc_newborns', 'med_child_check', 'hh_totalmembers', 'med_expenses_hh_ep', 'med_expenses_sp_ep', 'med_u5_deaths', 'asset_niceroof'], axis=1, inplace=True)

	df = df.fillna(df.median())

	return df

def main():
	train = pd.read_csv('data/train.csv')
	test = pd.read_csv('data/test.csv')

	train.drop(['surveyid', 'survey_date', 'amount_given_mpesa', 'amount_received_mpesa', 'early_survey', 'ent_employees', 'day_of_week', 'med_vacc_newborns', 'med_child_check', 'hh_totalmembers', 'asset_niceroof'], axis=1, inplace=True)

	#print(train.isnull().sum()) #-> Count empty rows per column
	train = train.loc[:, train.isnull().mean() < 0.6]
	train = train.fillna(train.median())

	test = cleanTest(test)

	train.to_csv('data/train_processed.csv', index=False)
	test.to_csv('data/test_processed.csv', index=False)





if __name__ == "__main__":
	main()