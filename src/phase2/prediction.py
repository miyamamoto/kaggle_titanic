# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def cleandata(data):
	data.sex[data.sex == "male"] = 1
	data.sex[data.sex == "female"] = 0

	return data

def fillnadata(train, test,  filename):
	import os
	tmp_out = train.append(test)
	file_out = filename+"_out.csv"
	file_in = filename+"_in.csv"
	
	tmp_out[['pclass','sex','sibsp','parch','fare','age']].to_csv(file_out)
	os.system("R --file=mi.r --args " + file_out + " " + file_in)
	
	tmp2 = pd.read_csv(file_in)

	rtrain = tmp2[:len(train)]
	rtest = tmp2[(len(train)):]
	
	return rtrain, rtest

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

	#
	traindata = cleandata(traindata)
	testdata = cleandata(testdata)

	ntargetdata = traindata[['survived']].values
	
	#fill na data
	traindata,testdata = fillnadata(traindata,testdata,"tmp")

	#convert pd format to numpy array
	ntraindata = traindata[['pclass','sex','sibsp','parch','fare','age']].values
	ntestdata = testdata[['pclass','sex','sibsp','parch','fare','age']].values

	#
	print "-"*40
	print len(ntraindata)
	print len(ntargetdata)
	print len(ntestdata)
	#constract model
	clf = random_tree_classifier(ntraindata, ntargetdata)
	
	#model score
	print clf.score(ntraindata, ntargetdata)

	#predict
	predict = clf.predict(ntestdata)

	#output csv
	z = np.array(zip(np.arange(1,len(predict)+1), predict), dtype=[('int', int), ('str', '|S1')])
	np.savetxt('../..//result/predict_phase2.csv', z, fmt='"%i","%s"',header ='"","survived"',comments='')

if __name__ == '__main__':
    main()
