# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def cleandata(data):
	data.sex[data.sex == "male"] = 1
	data.sex[data.sex == "female"] = 0
	return data

def fillnadata(data):
	data.age = data.age.fillna(data.age.mean())
	data.fare = data.fare.fillna(data.fare.mean())
	return data

def random_tree_classifier(data, target):
	from sklearn.ensemble import RandomForestClassifier
	clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=1, random_state=0, n_jobs=-1)
	target = target.reshape(-1)
	clf.fit(data, target)
	return clf

def main():
	#read train data and test data by pandas
	traindata = pd.read_csv('../../data/titanic/train.csv')
	testdata = pd.read_csv('../../data/titanic/test.csv')

	traindata = cleandata(traindata)
	testdata = cleandata(testdata)

	traindata = fillnadata(traindata)
	testdata = fillnadata(testdata)

	#convert pd format to numpy array
	ntraindata = traindata[['pclass','sex','sibsp','parch','fare','age']].values
	ntargetdata = traindata[['survived']].values
	ntestdata = testdata[['pclass','sex','sibsp','parch','fare','age']].values

	#constract model
	clf = random_tree_classifier(ntraindata, ntargetdata)
	
	#model score
	print "-"* 40
	print "score:"
	print clf.score(ntraindata, ntargetdata)

	#predict
	predict = clf.predict(ntestdata)

	#output csv
	z = np.array(zip(np.arange(1,len(predict)+1), predict), dtype=[('int', int), ('str', '|S1')])
	np.savetxt('../..//result/predict_phase1.csv', z, fmt='"%i","%s"',header ='"","survived"',comments='')

if __name__ == '__main__':
    main()
