# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy as sp
import math 
import os
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation


# #/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# Created on Sun Oct 14 01:28:08 2012
# """

# import pandas as pd
# import re

# INITIAL_FEATURES = [
#     "survived",
#     "pclass",
#     "name",
#     "sex",
#     "age",
#     "sibsp",
#     "parch",
#     "ticket",
#     "fare",
#     "cabin",
#     "embarked"
#     ]

# ######################################################################
# # Add your feature extracting functions here
# ######################################################################


# def sex_code_func(s_sex):
#     if s_sex == "male":
#         return 1
#     elif s_sex == "female":
#         return 0
#     else:
#         raise Exception("Unknown sex value (%s)" % s_sex)


# def sex_code(data):
#     return pd.DataFrame.from_dict({"sex_code":
#         data["sex"].apply(sex_code_func)})


# def embarked_code_func(s_embarked):
#     if s_embarked == "C":
#         return 0
#     elif s_embarked == "S":
#         return 1
#     elif s_embarked == "Q":
#         return 2
#     elif pd.isnull(s_embarked):
#         return 3  # if nan
#     else:
#         raise Exception("Unknown embarked value (%s)" % s_embarked)


# def embarked_code(data):
#     return pd.DataFrame.from_dict({"embarked_code":
#         data["embarked"].apply(embarked_code_func)})


# def ticket_number_func(s_ticket):
#     """ Extracts actual number of the ticket,
#         skipping the optional string prefix.
#         Examples:
#         >>> extract_digits("C.A./SOTON 34068")
#         34068
#         >>> extract_digits("34568")
#         36568
#     """
#     ends_with = s_ticket.split()[-1]
#     if ends_with.isdigit():
#         return int(ends_with)
#     else:
#         return 0


# def ticket_number(data):
#     df = pd.DataFrame.from_dict({"ticket_number":
#         data["ticket"].apply(ticket_number_func)})
#     # replace missing values with the mean
#     mean = df['ticket_number'].mean()
#     df = df['ticket_number'].apply(lambda i: i if i != 0 else mean)
#     return df


def cabin_code_func(s_cabin):
    if not pd.isnull(s_cabin):
        # extract cabin letters
        letters = re.findall("[A-G]+", s_cabin.upper())
        if letters:
            # assuming that all letters are the same (should we?)
            return ord(letters[0]) - 64  # 1 for 'A', 2 for 'B'...
        else:
            return 0  # no letters found
    else:
        return -1  # not a number




# def get_title_from_name(name):
#     [last, title_and_first] = name.split(", ")
#     title = title_and_first.split(" ")[0]
#     return title


# def title(data):
#     return pd.DataFrame.from_dict({"title":
#         data["name"].apply(get_title_from_name)})


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
	

	return data

def fillnadata(train, test,  filename):

	tmp_out = train.append(test)
	file_out = filename+"_out.csv"
	file_in = filename+"_in.csv"

	tmp_out[['pclass','sex','sibsp','parch','fare','age','ticket','embarked']].to_csv(file_out)	
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
		ntraindata = filltraindata[['pclass','sex','sibsp','parch','fare','age','ticket','embarked']].values
		ntestdata = filltestdata[['pclass','sex','sibsp','parch','fare','age','ticket','embarked']].values

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