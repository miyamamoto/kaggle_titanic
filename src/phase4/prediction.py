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
			print i
		 	a.append(None)

	data["cabin_leftright"] = pd.DataFrame(a)

	print data.head(10)

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

def KFold(train,target):

	result = pd.DataFrame()

	for estimator in range(100,101):
		for depth in range(10,15):
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
	    
	print result.sort_index(by="score",ascending=0)

	return result.ix[0]["estimator"], result.ix[0]["depth"], result.ix[0]["score"]
	#	return np.array(results).mean()


# def rfKFold(train,target):

# 	for i in range(100,101):
# 		for j in range(10,11):
# 		    #In this case we'll use a random forest, but this could be any classifier
# 			cfr = RandomForestClassifier(n_estimators=i, oob_score=True, max_depth=j, min_samples_split=1, random_state=1, n_jobs=-1,compute_importances=True)
# 			target = target.reshape(-1)

# 			#Simple K-Fold cross validation. 5 folds.
# 			print "n_fold size:" + str(n_folds_size(train))
# 			cv = cross_validation.KFold(len(train), n_folds=n_folds_size(train), indices=False)

# 			results = []
# 			for traincv, testcv in cv:
# 				probas = cfr.fit(train[traincv], target[traincv]).predict_proba(train[testcv])
# 				results.append(llfun(target[testcv], [x[1] for x in probas]) )


# 			print "i = "+ str(i)
# 			print "j = "+ str(j)
# 			print "Results: " + str(np.array(results).mean())

	#	return np.array(results).mean()

def random_tree_classifier(data, target,estimator,depth):
	clf = RandomForestClassifier(n_estimators=estimator, oob_score=True,max_depth=depth, min_samples_split=1, random_state=1, n_jobs=-1,compute_importances=True)
	target = target.reshape(-1)
	clf.fit(data, target)
	return clf


def main():
	traindata = pd.read_csv('../../data/titanic/train.csv')
	testdata = pd.read_csv('../../data/titanic/test.csv')
	clean_traindata = cleandata(traindata)
	clean_testdata = cleandata(testdata)
	tmp_out = clean_traindata.append(clean_testdata)

	for i in range(10):
		filename = "../../tmp/tmp_phase4" 
		file_out = filename+"_"+str(i)+"_out.csv"
		file_in = filename+"_"+str(i)+"_in.csv"

		tmp_out[['pclass','sex','age','sibsp','parch','fare','cabin','embarked','cabin_level','cabin_leftright']].to_csv(file_out)
		os.system("R --file=mi.r --args " + file_out + " " + file_in)
		alldata = pd.read_csv(file_in)
		
		#for debug
		alldata[:len(traindata)].to_csv(filename+"train_"+str(i)+".csv")
		alldata[(len(traindata)):].to_csv(filename+"test_"+str(i)+".csv")
	
		#convert pandas to array
		ntargetdata = traindata[['survived']].values
		ntraindata = alldata[:len(traindata)].values
		ntestdata = alldata[(len(traindata)):].values

		#K-fold cross-validation
#		score = rfKFold(ntraindata,ntargetdata)
		estimator,depth,score = KFold(ntraindata,ntargetdata)

		#constract model
		clf = random_tree_classifier(ntraindata, ntargetdata,estimator,depth)

		#predict
		predict = clf.predict(ntestdata)

		#output csv
		z = np.array(zip(np.arange(1,len(predict)+1), predict), dtype=[('int', int), ('str', '|S1')])
		np.savetxt('../..//result/predict_phase4_'+str(score)+'_'+str(estimator)+'_'+str(depth)+'_'+str(i)+'.csv', z, fmt='"%i","%s"',header ='"","survived"',comments='')	

if __name__ == '__main__':
	main()