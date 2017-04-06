from sklearn import neighbors, svm, linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score


class ML:
    def __init__(self, pred, min_shift = 0.01):
        self.pred = pred
        self.indicators = {}
        self.model = []
        self.min_shift = min_shift

    def randomForset_learn(self, n, ndays, p={'meanfractal', 'loc_vol'}, y_colums=None,forward_look=0):
        pred = self.pred
        if y_colums is None:
            pred["switch"] = np.where((pred.make_splits(5, inplace=False).shift(1) / pred.make_splits(5, inplace=False))
                                  > (1.0025 / 0.9975) + self.min_shift, 1, 0)
        else:
            pred["switch"] = pred[y_colums]
        self.indicators = p
        clf = RandomForestClassifier(n_estimators=n)
        results = pd.DataFrame()
        accuracy = []
        for i in range(ndays, len(pred.index)-forward_look):
            # We perform a 80/20 split on the data
            ind = int(np.round(ndays*0.8))
            X_TRAIN = pred.ix[(i - ndays):(i - ndays + ind),p]
            Y_TRAIN = pred.switch.ix[pred.ix[(i - ndays + forward_look):(i - ndays + ind + forward_look)].index]
            X_TEST = pred.ix[(i - ndays + ind):i,p]
            Y_TEST = pred.switch.ix[pred.ix[(i - ndays + ind + forward_look):(i + forward_look)].index]
            clf.fit(X_TRAIN, Y_TRAIN)
            predicted = clf.predict(X_TEST)
            a = clf.score(X_TEST, Y_TEST)
            accuracy.append(a)
            X_TEST["REAL"] = Y_TEST
            X_TEST["PRED"] = predicted
            X_TEST["ACC"] = a
            results = results.append(X_TEST.iloc[0])
        self.model = clf
        print("RandomForest model accuracy: " + str(np.mean(accuracy)))
        return results

    def knn_learn(self, n, ndays, p={'meanfractal', 'loc_vol'}, y_colums=None,forward_look=0):
        pred = self.pred
        if y_colums is None:
            pred["switch"] = np.where((pred.make_splits(5, inplace=False).shift(1) / pred.make_splits(5, inplace=False))
                                  > (1.0025 / 0.9975) + self.min_shift, 1, 0)
        else:
            pred["switch"] = pred[y_colums]
        self.indicators = p
        clf = neighbors.KNeighborsClassifier(n_neighbors=n)
        results = pd.DataFrame()
        accuracy = []
        for i in range(ndays, len(pred.index)-forward_look):
            # We perform a 80/20 split on the data
            ind = int(np.round(ndays*0.8))
            X_TRAIN = pred.ix[(i - ndays):(i - ndays + ind),p]
            Y_TRAIN = pred.switch.ix[pred.ix[(i - ndays + forward_look):(i - ndays + ind + forward_look)].index]
            X_TEST = pred.ix[(i - ndays + ind):i,p]
            Y_TEST = pred.switch.ix[pred.ix[(i - ndays + ind + forward_look):(i + forward_look)].index]
            clf.fit(X_TRAIN, Y_TRAIN)
            predicted = clf.predict(X_TEST)
            a = clf.score(X_TEST, Y_TEST)
            accuracy.append(a)
            X_TEST["REAL"] = Y_TEST
            X_TEST["PRED"] = predicted
            X_TEST["ACC"] = a
            # results = results.append([X_TEST.iloc[x] for x in range(0,len(X_TEST))])
            results = results.append(X_TEST.iloc[0])
        self.model = clf
        print("KNN model accuracy: " + str(np.mean(accuracy)))
        return results

    def linear_learn(self, c, ndays, p={'meanfractal', 'loc_vol'},y_colums=None,forward_look=0):
        pred = self.pred
        if y_colums is None:
            pred["switch"] = np.where((pred.make_splits(5, inplace=False).shift(1) / pred.make_splits(5, inplace=False))
                                  > (1.0025 / 0.9975) + self.min_shift, 1, 0)
        else:
            pred["switch"] = pred[y_colums]
        self.indicators = p
        clf = linear_model.LogisticRegression(C=c)
        results = pd.DataFrame()
        accuracy = []
        for i in range(ndays, len(pred.index)-forward_look):
            # We perform a 80/20 split on the data
            ind = int(np.round(ndays*0.8))
            X_TRAIN = pred.ix[(i - ndays):(i - ndays + ind),p]
            Y_TRAIN = pred.switch.ix[pred.ix[(i - ndays + forward_look):(i - ndays + ind + forward_look)].index]
            X_TEST = pred.ix[(i - ndays + ind):i,p]
            Y_TEST = pred.switch.ix[pred.ix[(i - ndays + ind + forward_look):(i + forward_look)].index]
            clf.fit(X_TRAIN, Y_TRAIN)
            predicted = clf.predict(X_TEST)
            a = clf.score(X_TEST, Y_TEST)
            accuracy.append(a)
            X_TEST["REAL"] = Y_TEST
            X_TEST["PRED"] = predicted
            X_TEST["ACC"] = a
            results = results.append(X_TEST.iloc[0])
        self.model = clf
        print("Linear model accuracy: " + str(np.mean(accuracy)))
        return results

    def SVC_learn(self, ndays, c = 1, kernel='rbf', p={'meanfractal', 'loc_vol'},y_colums=None,forward_look=0):
        pred = self.pred
        if y_colums is None:
            pred["switch"] = np.where((pred.make_splits(5, inplace=False).shift(1) / pred.make_splits(5, inplace=False))
                                  > (1.0025 / 0.9975) + self.min_shift, 1, 0)
        else:
            pred["switch"] = pred[y_colums]
        self.indicators = p
        clf = svm.SVC(kernel=kernel, gamma=0.7, C=c,probability=True)
        results = pd.DataFrame()
        accuracy = []
        for i in range(ndays, len(pred.index)-forward_look):
            # We perform a 80/20 split on the data
            ind = int(np.round(ndays*0.8))
            X_TRAIN = pred.ix[(i - ndays):(i - ndays + ind),p]
            Y_TRAIN = pred.switch.ix[pred.ix[(i - ndays + forward_look):(i - ndays + ind + forward_look)].index]
            X_TEST = pred.ix[(i - ndays + ind):i,p]
            Y_TEST = pred.switch.ix[pred.ix[(i - ndays + ind + forward_look):(i + forward_look)].index]
            clf.fit(X_TRAIN, Y_TRAIN)
            predicted = clf.predict(X_TEST)
            # proba = clf.predict_proba(X_TEST)
            a = clf.score(X_TEST, Y_TEST)
            accuracy.append(a)
            X_TEST["REAL"] = Y_TEST
            X_TEST["PRED"] = predicted
            # X_TEST["PROB"] = proba
            X_TEST["ACC"] = a
            results = results.append(X_TEST.iloc[0])
        self.model = clf
        print("SVC model accuracy: " + str(np.mean(accuracy)))
        return results


    def GaussianNB(self, ndays, p={'meanfractal', 'loc_vol'},y_colums=None,forward_look=0):
        pred = self.pred
        if y_colums is None:
            pred["switch"] = np.where((pred.make_splits(5, inplace=False).shift(1) / pred.make_splits(5, inplace=False))
                                  > (1.0025 / 0.9975) + self.min_shift, 1, 0)
        else:
            pred["switch"] = pred[y_colums]
        self.indicators = p
        clf =  GaussianNB()
        results = pd.DataFrame()
        accuracy = []
        for i in range(ndays, len(pred.index)-forward_look):
            # We perform a 80/20 split on the data
            ind = int(np.round(ndays*0.8))
            X_TRAIN = pred.ix[(i - ndays):(i - ndays + ind),p]
            Y_TRAIN = pred.switch.ix[pred.ix[(i - ndays + forward_look):(i - ndays + ind + forward_look)].index]
            X_TEST = pred.ix[(i - ndays + ind):i,p]
            Y_TEST = pred.switch.ix[pred.ix[(i - ndays + ind + forward_look):(i + forward_look)].index]
            clf.fit(X_TRAIN, Y_TRAIN)
            predicted = clf.predict(X_TEST)
            a = clf.score(X_TEST, Y_TEST)
            accuracy.append(a)
            X_TEST["REAL"] = Y_TEST
            X_TEST["PRED"] = predicted
            X_TEST["ACC"] = a
            results = results.append(X_TEST.iloc[0])
        self.model = clf
        print("GaussianNB model accuracy: " + str(np.mean(accuracy)))
        return results


    def setup_xgboost_model(self,ndays,actual):
        pred = self.pred
        results = pd.DataFrame()
        accuracy = []
        cv_params = {'max_depth': [3, 5, 7, 9, 11], 'min_child_weight': [1, 3, 5, 7, 9]}
        ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.8,
                      'objective': 'binary:logistic'}
        optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
                                     cv_params,
                                     scoring='accuracy', cv=5, n_jobs=-1)
        j = 0
        for i in range(ndays, len(pred.index)):
            # We perform a 80/20 split on the data
            ind = int(np.round(ndays*0.8))
            X_TRAIN = pred.ix[(i - ndays):(i - ndays+ind)]
            Y_TRAIN = actual.ix[X_TRAIN.index]
            X_TEST  = pred.ix[(i - ndays+ind):i]
            Y_TEST  = actual.ix[X_TEST.index]
            if j == 0:
                optimized_GBM.fit(X_TRAIN, Y_TRAIN)
                me = [x[1] for x in optimized_GBM.grid_scores_]
                best = optimized_GBM.grid_scores_[me.index(max(me))][0]
                cv_params = {'learning_rate': [0.1, 0.01, 0.005], 'subsample': [0.7, 0.8, 0.9]}
                ind_params = {'n_estimators': 1000, 'seed': 0, 'colsample_bytree': 0.8,
                              'objective': 'binary:logistic', 'max_depth': best["max_depth"], 'min_child_weight': best["min_child_weight"]}
                optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
                                             cv_params,
                                             scoring='accuracy', cv=5, n_jobs=-1)
                optimized_GBM.fit(X_TRAIN, Y_TRAIN)
                me = [x[1] for x in optimized_GBM.grid_scores_]
                best = {**best, **optimized_GBM.grid_scores_[me.index(max(me))][0]}
                our_params = {'eta': best["learning_rate"], 'seed': 0, 'subsample': best["subsample"], 'colsample_bytree': 0.8,
                              'objective': 'binary:logistic', 'max_depth': best["max_depth"], 'min_child_weight': best["min_child_weight"]}
                print("Parameters Optimised...")
            xgdmat = xgb.DMatrix(X_TRAIN, Y_TRAIN)
            # Grid Search CV optimized settings
            cv_xgb = xgb.cv(params=our_params, dtrain=xgdmat, num_boost_round=3000, nfold=5,
                            metrics=['error'],  # Make sure you enter metrics inside a list or you may encounter issues!
                            early_stopping_rounds=100)  # Look for early stopping that minimizes error
            final_gb = xgb.train(our_params, xgdmat, num_boost_round=432)
            testdmat = xgb.DMatrix(X_TEST)
            y_pred = final_gb.predict(testdmat)  # Predict using our testdmat
            predicted = y_pred
            predicted[predicted > 0.5] = 1
            predicted[predicted <= 0.5] = 0
            X_TEST["REAL"] = Y_TEST
            X_TEST["PRED"] = predicted
            X_TEST["PROB"] = y_pred
            ret = accuracy_score(predicted, Y_TEST), 1 - accuracy_score(predicted, Y_TEST)
            accuracy.append(ret[0])
            results = results.append(X_TEST.iloc[0])
            j = j + 1
        self.model = our_params
        print("Xgboost model accuracy: " + str(np.mean(accuracy)))
        return results

    def short_xgboost_model(self, startdate, ndays, actual,forward_look=0):
        pred = self.pred
        cv_params = {'max_depth': [3, 5, 7, 9, 11], 'min_child_weight': [1, 3, 5, 7, 9]}
        ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.8,
                      'objective': 'binary:logistic'}
        optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
                                     cv_params,
                                     scoring='accuracy', cv=5, n_jobs=-1)
        ALL_X = pred.ix[startdate:].iloc[0:ndays]
        if forward_look > 0:
            ALL_Y = actual.shift(-1*forward_look)
            ALL_Y = ALL_Y.ix[startdate:].iloc[0:ndays]
        else:
            ALL_Y = actual.ix[startdate:].iloc[0:ndays]
        ind = int(np.round(ndays * 0.8))
        X_TRAIN = ALL_X.iloc[0:ind]
        Y_TRAIN = ALL_Y.iloc[0:ind]
        X_TEST = ALL_X.iloc[ind:]
        Y_TEST = ALL_Y.iloc[ind:]
        optimized_GBM.fit(X_TRAIN, Y_TRAIN)
        best = sorted(optimized_GBM.grid_scores_, key = lambda x: (x[1], -np.std(x[2]), -x.parameters['max_depth']))[-1].parameters
        cv_params = {'learning_rate': [0.1, 0.01, 0.005], 'subsample': [0.7, 0.8, 0.9]}
        ind_params = {'n_estimators': 1000, 'seed': 0, 'colsample_bytree': 0.8,
                      'objective': 'binary:logistic', 'max_depth': best["max_depth"],
                      'min_child_weight': best["min_child_weight"]}
        optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
                                     cv_params,
                                     scoring='accuracy', cv=5, n_jobs=-1)
        optimized_GBM.fit(X_TRAIN, Y_TRAIN)
        me = [x[1] for x in optimized_GBM.grid_scores_]
        best = {**best, **sorted(optimized_GBM.grid_scores_, key = lambda x: (x[1], -np.std(x[2]), x.parameters['subsample']))[-1].parameters}
        our_params = {'eta': best["learning_rate"], 'seed': 0, 'subsample': best["subsample"], 'colsample_bytree': 0.8,
                      'objective': 'binary:logistic', 'max_depth': best["max_depth"],
                      'min_child_weight': best["min_child_weight"]}
        xgdmat = xgb.DMatrix(X_TRAIN, Y_TRAIN)  # Grid Search CV optimized settings
        cv_xgb = xgb.cv(params=our_params, dtrain=xgdmat, num_boost_round=3000, nfold=5,
                    metrics=['error'],  # Make sure you enter metrics inside a list or you may encounter issues!
                    early_stopping_rounds=100)  # Look for early stopping that minimizes error
        final_gb = xgb.train(our_params, xgdmat, num_boost_round=432)
        testdmat = xgb.DMatrix(X_TEST)
        y_pred = final_gb.predict(testdmat)  # Predict using our testdmat
        predicted = y_pred
        predicted[predicted > 0.5] = 1
        predicted[predicted <= 0.5] = 0
        X_TEST["REAL"] = Y_TEST
        X_TEST["PRED"] = predicted
        X_TEST["PROB"] = y_pred
        ret = accuracy_score(predicted, Y_TEST), 1 - accuracy_score(predicted, Y_TEST)
        X_TEST["ACC"] = ret[0]
        self.model = our_params
        print("Xgboost model accuracy: %s" % np.round(ret[0],4))
        return X_TEST

    def eval_xboost_model(self,X_TRAIN,Y_TRAIN,X_PREDICT):
        our_params = self.model
        xgdmat = xgb.DMatrix(X_TRAIN, Y_TRAIN)
        final_gb = xgb.train(our_params, xgdmat, num_boost_round=432)
        testdmat = xgb.DMatrix(X_PREDICT)
        y_pred = final_gb.predict(testdmat)  # Predict using our testdmat
        predicted = y_pred
        predicted[predicted > 0.5] = 1
        predicted[predicted <= 0.5] = 0
        X_PREDICT["PRED"] = predicted
        X_PREDICT["PROB"] = y_pred
        return X_PREDICT