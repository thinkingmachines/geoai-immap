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
    RFE
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV

def get_best_features(
    clf, 
    data, 
    features, 
    label, 
    metric='avg_f1_score',
    scale=True, 
    calibrate=False,
    range_=[100, 0, -10],
    verbose=0, 
    plot=False, 
):
    feat_dict = feature_selection(
        clf, 
        data, 
        features, 
        label, 
        scale=scale, 
        range_=range_,
        calibrate=calibrate,
        verbose=verbose
    )
    
    x, y = [], []
    for num_features in feat_dict:
        x.append(num_features) 
        y.append(feat_dict[num_features]['results'][metric])
    
    index = y.index(max(y))
    best_num_features = x[index]
    
    if plot:
        plt.plot(x, y);
        plt.xlabel('Number of Features')
        plt.ylabel('F1 Score')
        plt.show()
    
    best_features = feat_dict[best_num_features]['rfe_features']
    
    if verbose > 0:
        print('Best {} Features: {}'.format(best_num_features, best_features))
    
    return best_features, feat_dict

def feature_selection(
    clf, 
    data, 
    features, 
    label, 
    scale=False, 
    calibrate=False, 
    range_=[100, 0, -10],
    verbose=0
):
    feat_dict = {}
    
    # Define feature matrix and target vector
    X = data[features]
    y = data[label]
    
    # Iterate over range
    start, stop, step = range_[0], range_[1], range_[2]
    for num_features in range(start, stop, step):   
        
        # Print number of features at each step
        if verbose > 0:
            print('-'*20)
            print("| NUM FEATURES: {} |".format(num_features))
            print('-'*20)
        
        # Recursive feature elimination
        rfe_features = get_rfe_features(X, y, clf, num_features, verbose)
        
        # Calibration allows the model to output the class probability 
        # We use this when the model does not have attribute .predict_proba()
        # Source : https://machinelearningmastery.com/calibrated-classification-model-in-scikit-learn/
        if calibrate:
            clf_cv = CalibratedClassifierCV(clf)
        else:
            clf_cv = clf
        
        # Commence leave-one-area-out cross validation
        results, _ = geospatialcv(
            data, rfe_features, label, clf_cv, scale=scale, verbose=verbose
        )
        
        # Record results in dictionary
        feat_dict[num_features] = {}
        feat_dict[num_features]['results'] = results
        feat_dict[num_features]['rfe_features'] = rfe_features
    
    return feat_dict

def get_rfe_features(X, y, clf, num_features, verbose):
    """Implements Recursive Feature Elimination."""
    
    # Instantiate RFE selector
    rfe_selector = RFE(estimator=clf, n_features_to_select=num_features, step=10, verbose=verbose)
    
    # Normalize feature matrix
    X_norm = MinMaxScaler().fit_transform(X)
    
    # Fit RFE Selector
    rfe_selector.fit(X_norm, y)
    
    # Get list of selected features
    rfe_support = rfe_selector.get_support()
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
    
    area_code = {0: 'maicao', 1:'riohacha', 2:'uribia'}
    for area in data.area.unique():
        
        area_str = area_code[area].upper()
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