import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    recall_score,
    precision_score,
    classification_report,
    cohen_kappa_score
)
from sklearn.feature_selection import (
    SelectKBest,
    RFE,
    RFECV
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

AREA_CODES = {
    0 : 'Maicao', 
    1 : 'Riohacha', 
    2 : 'Uribia', 
    3 : 'Arauca1', 
    4 : 'Cucuta',
    5 : 'Arauquita', 
    6 : 'Tibu',
}
VALUE_CODES = {
    'Informal settlement': 1, 
    'Formal settlement': 2, 
    'Unoccupied land': 3
}

def get_cv_iterator(data):
    cv_iterator = []
    
    for area in data.area.unique():
        train_indices = data[data.area != area].index.values.astype(int)
        test_indices = data[data.area == area].index.values.astype(int)
        cv_iterator.append( (train_indices, test_indices) )
    
    return cv_iterator

def hyperparameter_optimization(data, features, label, clf, param_grid, verbose=0, scoring='f1'):
    cv_iterator = get_cv_iterator(data)
    
    X = data[features]
    y = data[label]
        
    pipe_clf = Pipeline([
        ('scaler',  MinMaxScaler()),
        ('classifier', clf)
    ])

    cv = GridSearchCV(
        estimator=pipe_clf, 
        param_grid=param_grid,
        cv=cv_iterator, 
        verbose=verbose, 
        scoring=scoring,
        n_jobs=-1
    )
    cv.fit(X, y)
    
    print('Best Paramaters: {}'.format(cv.best_params_))
    
    return cv

def rfecv_feature_selection(clf, data, features, label, scoring='f1', step=10, verbose=0):
    X = data[features]
    y = data[label]
    
    cv_iterator = get_cv_iterator(data)
    rfe_selector = RFECV(clf, step=step, cv=cv_iterator, scoring=scoring, verbose=verbose, n_jobs=-1)
    rfe_selector = rfe_selector.fit(X, y)
    
    rfe_support = rfe_selector.support_
    rfe_features = X.loc[:, rfe_support].columns.tolist()
    
    return rfe_features

def evaluate_model(model, X_test, y_test, area, scaler=None, verbose=0):
    if scaler != None:
        X_test = scaler.transform(X_test)
        
    y_pred = model.predict(X_test)
  
    f1_score_ = f1_score(y_test, y_pred, pos_label=1, average='binary')  
    precision = precision_score(y_test, y_pred, pos_label=1, average='binary') 
    recall = recall_score(y_test, y_pred, pos_label=1, average='binary') 
    accuracy = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
  
    if verbose > 1:
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred)) 
        print('{} Results: '.format(area))
        print('- F1 Score: {:.4f}'.format(f1_score_))
        print('- Kappa Statistics: {:.4f}'.format(kappa))
        print('- Precision: {:.4f}'.format(precision))
        print('- Recall: {:.4f}'.format(recall))
        print('- Accuracy: {:.4f}'.format(accuracy))

    return accuracy, f1_score_, precision, recall, kappa

def geospatialcv(data, features, label, clf, scale=False, verbose=0):
    
    data = data.fillna(0)
    accuracies, f1_scores, precisions, recalls, kappas = [], [], [], [], []
    classifiers = []
    
    for area in data.area.unique():
        
        area_str = AREA_CODES[area].upper()
        if verbose > 1:
            print('\nTest set: {}'.format(area_str))
        
        # Split into training and test sets
        train = data[data.area != area]
        test = data[data.area == area]
        
        X_train, X_test = train[features], test[features]
        y_train, y_test = train[label], test[label]
        
        # Scale/transform features
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        
        # Fit classifier to training set
        clf.fit(X_train, y_train)
        
        # Evaluate model
        acc, f1, prec, rec, kappa = evaluate_model(
            clf, 
            X_test, 
            y_test, 
            area_str, 
            scaler=scaler, 
            verbose=verbose
        )
        
        # Save results
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
        kappas.append(kappa)
        classifiers.append(clf)
    
    results = {
        'avg_accuracy' : np.mean(accuracies),
        'avg_f1_score' : np.mean(f1_scores),
        'avg_precision' : np.mean(precisions),
        'avg_recall' : np.mean(recalls),
        'avg_kappa' : np.mean(kappas)
    }
    
    if verbose > 0:
        print()
        print('Average F1 Score: {:.4f}'.format(results['avg_f1_score']))
        print('Average Kappa statistic: {:.4f}'.format(results['avg_kappa']))
        print('Average Precision: {:.4f}'.format(results['avg_precision']))
        print('Average Recall: {:.4f}'.format(results['avg_recall']))
        print('Average Accuracy: {:.4f}'.format(results['avg_accuracy']))
        print()
    
    return results, classifiers

def resample(data, num_neg_samples, neg_dist, random_state):    
    data_area = []
    
    for area in data['area'].unique():
        neg_samples = data[
            (data['area'] == area) 
            & (data['target'] != 1)
        ]
        for value in neg_dist:
            samples = neg_samples[
                data['target'] == VALUE_CODES[value]
            ]
            
            n_samples = int(
                num_neg_samples*neg_dist[value]
            )
            if len(samples) < n_samples:
                n_samples = len(samples)
                
            samples = samples.sample(
                n_samples, 
                replace=False, 
                random_state=random_state
            )
            data_area.append(samples)

    pos_samples = data[data['target'] == 1]
    data_area.append(pos_samples)
    data = pd.concat(data_area)
    data = data.reset_index(drop=True)
    
    return data