# -*- coding: utf-8 -*-
import pandas as pd
import os
import math
import numpy as np
import scipy as sp
#less predict_phase4_0.439166090393_0.csv 

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

	data.cabin_level = data.cabin_level.str.replace('^A.*',"1")
	data.cabin_level = data.cabin_level.str.replace('^B.*',"2")
	data.cabin_level = data.cabin_level.str.replace('^C.*',"3")
	data.cabin_level = data.cabin_level.str.replace('^D.*',"4")
	data.cabin_level = data.cabin_level.str.replace('^E.*',"5")
	data.cabin_level = data.cabin_level.str.replace('^F.*',"6")
	data.cabin_level = data.cabin_level.str.replace('^G.*',"7")
	data.cabin_level = data.cabin_level.str.replace('^T.*',"8")

	a = []
	for i in data["ticket"].values:
		try:
			a.append(int(i) % 2)
		except:
			print "except"
		 	a.append(None)

	data["left"] = pd.DataFrame(a)

	return data

def n_folds_size(l):
	return int(math.ceil(1 + (math.log(len(l), 10) / math.log(2, 10))))

def KFold(train,target):

	result =pd.DataFrame()

	for estimator in [1,20,40,60,80,100]:
#	for estimator in [100]:
		for depth in  [5,6,7,8,9,10,11,12,13]:
			sum_score = 0
		    #In this case we'll use a random forest, but this could be any classifier
			cfr = RandomForestClassifier(n_estimators=estimator, oob_score=True, max_depth=depth, min_samples_split=1, random_state=1, n_jobs=-1,compute_importances=True)
			target = target.reshape(-1)

			#Simple K-Fold cross validation. 5 folds.
			nfoldsize = n_folds_size(train)
			print "n_fold size:" + str(nfoldsize)
			cv = cross_validation.KFold(len(train), n_folds=nfoldsize, indices=False)

			for traincv, testcv in cv:
				probas = cfr.fit(train[traincv], target[traincv]).predict(train[testcv])
				score = sum(probas==target[testcv])
				sum_score += score


			print "="*40    
			print "estimator = "+str(estimator)
			print "depth = "+str(depth)
			print "average score in %0.4f" % (sum_score/nfoldsize)    
			print "="*40 
	
			result = result.append(pd.DataFrame({"estimator":[estimator],"depth":[depth],"score":[score]}),ignore_index=True)   

	result = result.sort_index(by="score",ascending=0).reset_index(drop=True)
	print result
	result.to_csv('../../tmp/'+str(estimator)+'_'+str(depth)+'.csv')
	return result.ix[0]["estimator"], result.ix[0]["depth"], result.ix[0]["score"]

def random_tree_classifier(data, target,estimator,depth):

	clf = RandomForestClassifier(n_estimators=estimator,oob_score=True, max_depth=depth,min_samples_split=1, random_state=1, n_jobs=-1,compute_importances=True)
	target = target.reshape(-1)
	clf.fit(data, target)
	return clf

def KFold_svm(train,target):
	from sklearn import svm
	sum_score = 0
    #In this case we'll use a random forest, but this could be any classifier
	clf = svm.SVC()
	target = target.reshape(-1)

	#Simple K-Fold cross validation. 5 folds.
	nfoldsize = n_folds_size(train)
	print "n_fold size:" + str(nfoldsize)
	cv = cross_validation.KFold(len(train), n_folds=nfoldsize, indices=False)

	for traincv, testcv in cv:
		probas = clf.fit(train[traincv], target[traincv]).predict(train[testcv])
		score = sum(probas==target[testcv])
		sum_score += score
	
	print "="*40    
	print "average score in %0.4f" % (sum_score/nfoldsize)    
	print "="*40 
	    
	return (sum_score/nfoldsize) 

def svm(data, target):
	from sklearn import svm
	clf = svm.SVC()
	target = target.reshape(-1)
	clf.fit(data, target)
	return clf

def main():

	for i in range(100):
		traindata = pd.read_csv('../../data/titanic/train.csv')
		testdata = pd.read_csv('../../data/titanic/test.csv')
		clean_traindata = cleandata(traindata)
		clean_testdata = cleandata(testdata)
		tmp_out = clean_traindata.append(clean_testdata)

		filename = "../../tmp/tmp_phase4" 
		file_out = filename+"_out.csv"
		file_in = filename+"_in.csv"

		tmp_out[['pclass','sex','age','sibsp','parch','fare','cabin','embarked','cabin_level','left']].to_csv(file_out)
		os.system("R --file=mi.r --args " + file_out + " " + file_in)
		alldata = pd.read_csv(file_in).reset_index()

		#minor or not
		alldata["minor"] = 0
		alldata.minor[18 <alldata.age][alldata.age<=30] = 1 
		alldata.minor[30 <alldata.age][alldata.age<=40] = 2 
		alldata.minor[40 <alldata.age][alldata.age<=50] = 3 
		alldata.minor[50 <alldata.age][alldata.age<=60] = 4 
		alldata.minor[60 <alldata.age ] = 5

		alldata["fare_lank"] = 0
		alldata.fare_lank[0 <alldata.fare][alldata.fare<=30] = 1 
		alldata.fare_lank[30 <alldata.fare][alldata.fare<=60] = 2 
		alldata.fare_lank[60 <alldata.fare][alldata.fare<=90] = 3 
		alldata.fare_lank[90 <alldata.fare][alldata.fare<=120] = 4 
		alldata.fare_lank[120 <alldata.fare][alldata.fare<=150] = 5 
		alldata.fare_lank[150 <alldata.fare][alldata.fare<=180] = 6
		alldata.fare_lank[180 <alldata.fare] = 7

		#for debug
		alldata[:len(traindata)].to_csv(filename+"train.csv")
		alldata[(len(traindata)):].to_csv(filename+"test.csv")

		#convert pandas to array
		ntargetdata = traindata[['survived']].values
		ntraindata = alldata[["age","sex","pclass","minor","left","parch","fare",'sibsp',"fare_lank","cabin_level"]][:len(traindata)].values
		ntestdata = alldata[["age","sex","pclass","minor","left","parch","fare",'sibsp',"fare_lank","cabin_level"]][(len(traindata)):].values

		#K-fold cross-validation
		#estimator,depth,score = KFold(ntraindata,ntargetdata)
		estimator,depth,score = KFold(ntraindata,ntargetdata)
		print "max estimator:"
		print estimator
		print "depth"
		print depth

		#constract model
		clf = random_tree_classifier(ntraindata, ntargetdata, estimator, depth)
		print clf.feature_importances_

		#predict
		predict = clf.predict(ntestdata)
		predict_proba = clf.predict_proba(ntestdata)
		
		sex = alldata[["sex"]][(len(traindata)):].values

		for i in range(0,len(sex)):
			print predict_proba[i,0]
			print predict_proba[i,1]
			if sex[i] == 0:
				predict_proba[i,1] = pow(predict_proba[i,1],0.33)
				predict_proba[i,0] = 1-predict_proba[i,1]
			else:
				predict_proba[i,0] = pow(predict_proba[i,0],0.33)
				predict_proba[i,1] = 1-predict_proba[i,0]


		predict_bias = []
		for proba in predict_proba:
			if proba[1] > 0.5:
				predict_bias.append(1)
			else:
				predict_bias.append(0)

	#	print predict_bias

		#output csv
		z = np.array(zip(np.arange(1,len(predict)+1), predict), dtype=[('int', int), ('str', '|S1')])
		np.savetxt('../..//result/predict_phase4_'+str(score)+'_'+str(estimator)+'_'+str(depth)+'.csv', z, fmt='"%i","%s"',header ='"","survived"',comments='')	


		#output csv
		z = np.array(zip(np.arange(1,len(predict_bias)+1), predict_bias), dtype=[('int', int), ('str', '|S1')])
		np.savetxt('../..//result/predict_phase4_'+str(score)+'_'+str(estimator)+'_'+str(depth)+'_bias.csv', z, fmt='"%i","%s"',header ='"","survived"',comments='')	
	
if __name__ == '__main__':
	main()