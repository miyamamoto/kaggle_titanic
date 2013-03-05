# -*- coding: utf-8 -*-
import pandas as pd
import os
import math
import numpy as np
import scipy as sp

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

def cleandata(data):
	data.sex[data.sex == "male"] = 1
	data.sex[data.sex == "female"] = 0

	data.embarked[data.embarked == "C"] = 0
	data.embarked[data.embarked == "Q"] = 1
	data.embarked[data.embarked == "S"] = 2

	data.ticket = data.ticket.str.replace('^.*\s', '')
	data["cabin_level"] = data.cabin

	data.cabin_level = data.cabin_level.str.replace('^A.*',1)
	data.cabin_level = data.cabin_level.str.replace('^B.*',2)
	data.cabin_level = data.cabin_level.str.replace('^C.*',3)
	data.cabin_level = data.cabin_level.str.replace('^D.*',4)
	data.cabin_level = data.cabin_level.str.replace('^E.*',5)
	data.cabin_level = data.cabin_level.str.replace('^F.*',6)
	data.cabin_level = data.cabin_level.str.replace('^G.*',7)
	data.cabin_level = data.cabin_level.str.replace('^T.*',8)

	print data.head(100)
	a = []

	for i in data["ticket"].values:
		try:
			a.append(int(i))
		except:
			print "except"
			print i
		 	a.append(None)

	data["cabin_leftright"] = pd.DataFrame(a)

	return data

def n_folds_size(l):
	return int(math.ceil(1 + (math.log(len(l), 10) / math.log(2, 10))))

def llfun(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

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

def random_tree_classifier(data, target):
	clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=1, random_state=1, n_jobs=-1)
	target = target.reshape(-1)
	clf.fit(data, target)

	return clf

def main():
	
	traindata = pd.read_csv('../../data/titanic/train.csv')
	testdata = pd.read_csv('../../data/titanic/test.csv')
	clean_traindata = cleandata(traindata)
	clean_testdata = cleandata(testdata)
	tmp_out = clean_traindata.append(clean_testdata)

	for i in range(1):
		filename = "../../tmp/tmp_phase4" 
		file_out = filename+"_"+str(i)+"_out.csv"
		file_in = filename+"_"+str(i)+"_in.csv"

		tmp_out[['pclass','sex','age','sibsp','parch','ticket','fare','cabin','embarked','cabin_level','cabin_leftright']].to_csv(file_out)

#		os.system("R --file=mi.r --args " + file_out + " " + file_in)
		
		alldata = pd.read_csv(file_in)
		alldata[:len(traindata)].to_csv(filename+"train_"+str(i)+".csv")
		alldata[(len(traindata)):].to_csv(filename+"test_"+str(i)+".csv")
	
		ntargetdata = traindata[['survived']].values
		ntraindata = alldata[:len(traindata)].values
		ntestdata = alldata[(len(traindata)):].values

		print ntestdata
		#K-fold cross-validation
		score = rfKFold(ntraindata,ntargetdata)

		#constract model
		clf = random_tree_classifier(ntraindata, ntargetdata)

		#predict
		predict = clf.predict(ntestdata)

		#output csv
		z = np.array(zip(np.arange(1,len(predict)+1), predict), dtype=[('int', int), ('str', '|S1')])
		np.savetxt('../..//result/predict_phase4C_'+str(score)+'_'+str(i)+'.csv', z, fmt='"%i","%s"',header ='"","survived"',comments='')	

if __name__ == '__main__':
	main()