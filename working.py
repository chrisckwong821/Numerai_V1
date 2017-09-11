import os.path
import numpy as np
import pandas as pd
from sklearn import preprocessing

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import re
import time
from sklearn.model_selection import GridSearchCV,train_test_split,ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.metrics import make_scorer,mean_squared_log_error,log_loss

import random
from numpy import random

from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC

from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier,ExtraTreeRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostRegressor,GradientBoostingClassifier


class Preprocess:
    def __init__(self,tournament=71):
        parentdir = os.path.join(os.path.abspath(os.getcwd()),os.pardir)
        path = os.path.join(parentdir,"T{}/".format(tournament))
        self.training_data = pd.read_csv(path + "numerai_training_data.csv",header=0)
        self.test_data = pd.read_csv(path + "numerai_tournament_data.csv",header=0)
        self.X = self.training_data[[f for f in list(self.training_data) if "feature" in f]]
        self.Y = self.training_data['target']
        self.x_prediction = self.test_data[[f for f in list(self.training_data) if "feature" in f]]
        self.validation = self.test_data['target'][:16686]
        self.ids_test = self.test_data['id']
        self.ids_train = self.training_data['id']
        self.era = self.training_data['era']
    def eraStandardize(self,dataset,model):
        placeholder = pd.DataFrame()
        era = set([''.join(x for x in element if x.isdigit()) for element in dataset['era']])
        era.discard('')
        era = [int(i) for i in era]
        maxera,minera = max(era)+1,min(era)
        for i in range(minera,maxera):
            data = dataset[dataset['era']=='era{}'.format(i)][[f for f in list(dataset) if "feature" in f]]
            placeholder = placeholder.append(pd.DataFrame(model(data)))
        return placeholder
    def eraStandardize_random(self,data,iter,reduction):
        #data = id, era, feature
        placeholder = pd.DataFrame()
        era = set([''.join(x for x in element if x.isdigit()) for element in dataset['era']])
        era.discard('')
        era = [int(i) for i in era]
        maxera,minera = max(era)+1,min(era)
        for i in range(minera,maxera):
            data = dataset[dataset['era']=='era{}'.format(i)][[f for f in list(dataset) if "feature" in f]]
            placeholder = placeholder.append(pd.DataFrame(model(data)))
        return placeholder
    def advisory_screen(self,portion,train_x):
        model = RandomForestClassifier(n_estimators=20)
        X_test = self.x_prediction
        sample_size_test = X_test.shape[0]
        idholder = pd.DataFrame()
        for i in range(1,97):
            time.sleep(2)
            print(i)
            X_train = train_x[train_x['era']=='era{}'.format(i)][[f for f in list(train_x) if "feature" in f]]
            X_train_id = train_x[train_x['era']=='era{}'.format(i)].id.reset_index()
            sample_size_train = X_train.shape[0]
            X_data = pd.concat([X_train, X_test])
            Y_data = np.array(sample_size_train*[0] + sample_size_test*[1])
            model.fit(X_data,Y_data)
    #model.predict_proba(extract(x_prediction))[:,1]
            pre_train = pd.DataFrame(data={'wrong-score':model.predict_proba(X_train)[:,1]})
            pre_test = pd.DataFrame(data={'right-score':model.predict_proba(X_test)[:,1]})
            num_data = round(portion * X_train.shape[0])
            print(num_data)
            test_alike_data = pd.concat([X_train_id,pre_train],axis=1)
            test_alike_data = test_alike_data.sort_values(by='wrong-score',ascending=False)[:num_data]
            #test_class = self.ids_test[16686:].reset_index().join(pre_test).sort_values(by='right-score',ascending=False)
        ####control
            print('out of {0} training sample and {1} testing sample'.format(sample_size_train,sample_size_test))
            print('correct for training: {}'.format(sum([1 for i in model.predict_proba(X_train)[:,1] if i<0.5])))
            #print('correct for validation: {}'.format(sum([1 for i in model.predict_proba(X_test)[:,1] if i>0.5])))
            #print(pd.concat([test_alike_data.head(n=5),test_alike_data.tail(n=5)]))
            #print(pd.concat([test_class.head(n=5),test_class.tail(n=5)]))
        #########################
            idholder = idholder.append(pd.DataFrame(test_alike_data.id),ignore_index=True)
        return train_x[train_x.id.isin(idholder.id)]

    def logconversion(self,x):
        return x.apply(lambda k:np.log(1+k),axis=1)
    def unitNormalizer(self,x):
        model = preprocessing.Normalizer()
        return model.fit_transform(x)
    def RobustScaler(self,x):
        model = preprocessing.RobustScaler(quantile_range=(20.0, 70.0))
        return model.fit_transform(x)
    def Binarizer(self,x):
        model = preprocessing.Binarizer(threshold=0.6)
        return model.fit_transform(x)
    def StandardScaler(self,x,std=False,mean=True): #to unit variance
        model = preprocessing.StandardScaler(copy=False,with_std=std,with_mean=mean)
        return model.fit_transform(x)
    def PolyFeature(self,x):
        model = preprocessing.PolynomialFeatures(interaction_only=False)
        return pd.DataFrame(model.fit_transform(x))
    def KernelCenterer(self,x): #only demean
        model = preprocessing.KernelCenterer()
        return pd.DataFrame(model.fit_transform(x))

class models:
    def __init__(self,tournament=71):
        parentdir = os.path.join(os.path.abspath(os.getcwd()),os.pardir)
        path = os.path.join(parentdir,"T{}/".format(tournament))
        self.training_data = Preprocess().training_data
        self.test_data = Preprocess().test_data
        #self.X_train = self.training_data[[f for f in list(self.training_data) if "feature" in f]]
        #self.y_train = self.training_data['target']
        self.X_prediction = self.test_data[[f for f in list(self.training_data) if "feature" in f]]
        self.X_test = self.X_prediction[:16686]
        self.y_test = self.test_data['target'][:16686]
        #####feature engineering in this line #########
        self.X = self.training_data
        while self.X.shape[0] >= 90000:
            self.X = Preprocess().advisory_screen(0.8,self.X)
            print(self.X.shape[0])
            print('end1loop')
        self.X_train = self.X[[f for f in list(self.training_data) if "feature" in f]]
        self.y_train = self.X['target']
        #self.X_train = Preprocess().eraStandardize(self.training_data, Preprocess().StandardScaler)
        #self.X_test = Preprocess().eraStandardize(self.test_data, Preprocess().StandardScaler)
        #self.X_prediction = Preprocess().StandardScaler(self.X_prediction)
        ############################################
        print('init')
    def DTC(self):
        model = DecisionTreeClassifier()
        p = [{'min_samples_split':[[2]],'max_features':[['log2'],['auto']],'max_depth':[[5]]}]
        return model,p
    def RFC(self):
        model = RandomForestClassifier()
        p = [{'n_estimators':[[10000]],'min_samples_split':[[0.001]],'max_features':[['log2']],'oob_score':[[True]]}]
        return model,p
    def ABC(self):
        model = AdaBoostClassifier()
        p = [{'learning_rate':[[0.01,0.05]],'n_estimators':[[10,30,100,300]]}]
        return model,p
    def GBC(self):
        model = GradientBoostingClassifier()
        p = [{'learning_rate':[[0.01,0.02,0.05]],'n_estimators':[[100,300,500,1000]],'max_features':[['auto']],'max_depth':[[4]]}]
        return model,p
    def LR(self):
        model = LogisticRegression()
        p = [{'max_iter':[[1000]],'tol':[[0.00001]]}]
        return model,p
    def kernel(self,model,p):
        #score = make_scorer(neg_log_loss,greater_is_better=False)
        parameter = ParameterGrid(p)
        clf = GridSearchCV(model, parameter, cv=5, scoring='neg_log_loss',n_jobs=2)
        time.sleep(5)
        clf.fit(self.X_train,self.y_train)
        print('after fit')
        #print(clf.score(self.X_train,self.y_train))
        #print(clf.score(self.X_test,self.y_test))
        proba = clf.predict_proba(self.X_test)[:,1]
        error = log_loss(self.y_test,proba,normalize=True)
        print(error)
        print(clf.best_params_)
        result = pd.DataFrame(Preprocess().test_data['id']).join(pd.DataFrame(data={'probability':clf.predict_proba(self.X_prediction)[:,1]}))
        return result,error
    def model_stack(self,df):
        placeholder = df.DataFrame()
        return placeholder.join(df)


if __name__ == '__main__':
    a = models()
    model,p = a.RFC()
    result,error = a.kernel(model,p)
    result.to_csv('%.4f.AI_submission.csv'%(error),index=False)
