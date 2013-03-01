# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy as sp
import math 
import os
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation


def cleandata(data):
	data.sex[data.sex == "male"] = 1
	data.sex[data.sex == "female"] = 0

	return data

def fillnadata(train, test,  filename):

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
	clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=1, random_state=1, n_jobs=-1)
	target = target.reshape(-1)
	clf.fit(data, target)

	return clf

#https://www.kaggle.com/wiki/LogarithmicLoss
def llfun(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

def n_folds_size(l):
	return int(math.ceil(1 + (math.log(len(l), 10) / math.log(2, 10))))

def rfKFold(train,target):
    #In this case we'll use a random forest, but this could be any classifier
	cfr = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=1, random_state=1, n_jobs=-1)
	target = target.reshape(-1)

	#Simple K-Fold cross validation. 5 folds.
	print "n_fold size:" + str(n_folds_size(train))
	cv = cross_validation.KFold(len(train), n_folds=n_folds_size(train), indices=False)

	results = []
	for traincv, testcv in cv:
		probas = cfr.fit(train[traincv], target[traincv]).predict_proba(train[testcv])
		results.append(llfun(target[testcv], [x[1] for x in probas]) )
	
	print "Results: " + str(np.array(results).mean())
	return np.array(results).mean()

def main():
	
	#read train data and test data by pandas.
	traindata = pd.read_csv('../../data/titanic/train.csv')
	testdata = pd.read_csv('../../data/titanic/test.csv')

	#clean data.
	traindata = cleandata(traindata)
	testdata = cleandata(testdata)

	for i in range(100):
		#fill na data by multiple imputation(R)
		filltraindata,filltestdata = fillnadata(traindata,testdata,"../../tmp/tmp_phase2_"+str(i))

		#convert pd format to numpy array
		ntargetdata = traindata[['survived']].values
		ntraindata = filltraindata[['pclass','sex','sibsp','parch','fare','age']].values
		ntestdata = filltestdata[['pclass','sex','sibsp','parch','fare','age']].values

		#K-fold cross-validation
		score = rfKFold(ntraindata,ntargetdata)

		#constract model
		clf = random_tree_classifier(ntraindata, ntargetdata)

		#predict
		predict = clf.predict(ntestdata)

		#output csv
		z = np.array(zip(np.arange(1,len(predict)+1), predict), dtype=[('int', int), ('str', '|S1')])
		np.savetxt('../..//result/predict_phase2_'+str(score)+'_'+str(i)+'.csv', z, fmt='"%i","%s"',header ='"","survived"',comments='')

if __name__ == '__main__':
	main()