import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier



def main():
	train = pd.read_csv('data/train_processed_newest.csv')
	test = pd.read_csv('data/test_processed_newest.csv')

	X = train.drop(['depressed'], axis=1)
	Y = train['depressed']

	XTrain, XTest, yTrain, yTest = train_test_split(X, Y, test_size=0.33, random_state=42)

	# Random Forest
	rfc = RandomForestClassifier(n_estimators=150, max_depth=10, n_jobs=10, criterion='entropy', min_samples_split=5, random_state=None, bootstrap=True, max_features='auto', max_leaf_nodes=None, min_samples_leaf=1, min_weight_fraction_leaf=0.0, oob_score=False, verbose=0, warm_start=False)

	rfc.fit(XTrain, yTrain)

	preds = rfc.predict(XTest)
	accuracy = accuracy_score(yTest, preds)
	#best = rfc.feature_importances_
	confusion = confusion_matrix(yTest, preds)
	report = classification_report(yTest, preds)
	cross_val = cross_val_score(rfc, XTrain, yTrain, cv=10)
	area = roc_auc_score(yTest, preds)


	print('Accuracy: {}'.format(accuracy))
	#print('Features: {}'.format(best))
	print('Confusion Matrix \n{}'.format(confusion))
	print('Classification report \n{}'.format(report))
	print('Accuracy for 10 folds: {}\n'.format(cross_val))
	print('Area under curve: {}'.format(area))

	#Predict
	x = test.drop(['surveyid', 'depressed'], axis=1)
	y = rfc.predict(x)

	data = {'surveyid': test['surveyid'], 'depressed': y}
	final = pd.DataFrame.from_dict(data)

	depressed = len(final[final['depressed'] == 1])
	undepressed = len(final[final['depressed'] == 0])

	print(depressed)
	print('\n')
	print(undepressed)
	
	final.to_csv('data/test_predictions_new.csv', index=False)	
	









if __name__ == "__main__":
	main()