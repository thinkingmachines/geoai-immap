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
    cohen_kappa_score,
    roc_auc_score
)
from pandas_ml import ConfusionMatrix
from sklearn.feature_selection import (
    SelectKBest,
    RFE,
    RFECV
)
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

AREA_CODES = {
    0 : 'Maicao', 
    1 : 'Riohacha', 
    2 : 'Uribia', 
    3 : 'Arauca', 
    4 : 'Cucuta',
    5 : 'Tibu',
    6 : 'Arauquita', 
    7 : 'Soacha',
    8 : 'Bogota'
}
VALUE_CODES = {
    1 : 'Informal settlement', 
    2 : 'Formal settlement', 
    3 : 'Unoccupied land'
}

def get_cv_iterator(data):
    
    # Get list of unique areas
    areas = list(data.area.unique())
    
    # Split the dataset into train test indices
    cv_iterator = []
    for area in areas:
        train_indices = data[data.area != area].index.values.astype(int)
        test_indices = data[data.area == area].index.values.astype(int)
        cv_iterator.append( (train_indices, test_indices) )
            
    return cv_iterator, [AREA_CODES[x] for x in areas]

def rfecv_feature_selection(clf, X, y, cv, scoring='f1', step=10, verbose=0):    
     
    # Instantiate RFE feature selector
    rfe_selector = RFECV(
        clf, step=step, cv=cv, scoring=scoring, verbose=verbose, n_jobs=-1
    )
    
    # Fit RFE feature selector
    rfe_selector = rfe_selector.fit(X, y)
    
    # Get selected features
    rfe_support = rfe_selector.support_
    rfe_features = X.loc[:, rfe_support].columns.tolist()
    
    return rfe_features

def evaluate_model(clf, X_test, y_test, verbose=0):
    
    # Predict on test set
    y_pred = clf.predict(X_test)
    
    # Calculate metrics
    y_test, y_pred = list(y_test), list(y_pred)
    f1_score_ = f1_score(y_test, y_pred, pos_label=1, average='binary')  
    precision = precision_score(y_test, y_pred, pos_label=1, average='binary') 
    recall = recall_score(y_test, y_pred, pos_label=1, average='binary') 
    accuracy = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred)
  
    if verbose > 1:
        print(ConfusionMatrix(y_test, y_pred))
        print('\n', classification_report(y_test, y_pred)) 
        print('F1 Score: {:.4f}'.format(f1_score_))
        print('Kappa Statistics: {:.4f}'.format(kappa))
        print('Precision: {:.4f}'.format(precision))
        print('Recall: {:.4f}'.format(recall))
        print('Accuracy: {:.4f}'.format(accuracy))
        print('ROC AUC: {:.4f}'.format(auc_score))
        print()

    return accuracy, f1_score_, precision, recall, kappa, auc_score

def nested_spatial_cv(
    clf, 
    X, 
    y, 
    splits, 
    param_grid, 
    search_type='grid', 
    feature_selection=True, 
    verbose=0,
    random_state=42
):
    
    scores = {
        'f1_score' : [],
        'kappa' : [],
        'precision' : [],
        'recall' : [],
        'accuracy' : [],
        'roc_auc' : []
    }
    
    outer_cv, areas = get_cv_iterator(splits)
    for (train_indices, test_indices), area in zip(outer_cv, areas):
        
        splits_train = splits.loc[train_indices].reset_index(drop=True)    
        X_train = X.loc[train_indices].reset_index(drop=True)
        X_test = X.loc[test_indices].reset_index(drop=True)
        y_train = y.loc[train_indices].reset_index(drop=True)
        y_test = y.loc[test_indices].reset_index(drop=True)
        
        inner_cv, _ = get_cv_iterator(splits_train)
        
        if feature_selection is True:
            best_features = rfecv_feature_selection(
                clf, X_train, y_train, inner_cv, scoring='f1', step=10, verbose=0
            )
            X_train = X_train[best_features]
            X_test = X_test[best_features]
        
        pipe_clf = Pipeline([
            ('scaler',  MinMaxScaler()),
            ('classifier', clf)
        ])
        
        best_estimator = pipe_clf
        if search_type is not None:
            if search_type == 'grid':
                cv = GridSearchCV(
                    estimator=pipe_clf, 
                    param_grid=param_grid,
                    cv=inner_cv, 
                    verbose=0, 
                    scoring='f1',
                    n_jobs=-1
                )
            elif search_type == 'random':
                cv = RandomizedSearchCV(
                    estimator=pipe_clf, 
                    param_distributions=param_grid,
                    n_iter=10,
                    cv=inner_cv, 
                    verbose=0, 
                    scoring='f1',
                    n_jobs=-1,
                    random_state=random_state
                )
            cv.fit(X_train, y_train)
            best_estimator = cv.best_estimator_
            
        best_estimator.fit(X_train, y_train)
        
        if verbose > 0: print("Test Set: {}".format(area))
        accuracy, f1_score_, precision, recall, kappa, auc_score = evaluate_model(
            best_estimator, X_test, y_test, verbose=verbose
        )
        
        # Save results
        scores['f1_score'].append(f1_score_)
        scores['kappa'].append(kappa)
        scores['precision'].append(precision)
        scores['recall'].append(recall)
        scores['accuracy'].append(accuracy)
        scores['roc_auc'].append(auc_score)
    
    if verbose > 0:
        print()
        print('Mean F1 Score: {:.4f}'.format(np.mean(scores['f1_score'])))
        print('Mean Kappa statistic: {:.4f}'.format(np.mean(scores['kappa'])))
        print('Mean Precision: {:.4f}'.format(np.mean(scores['precision'])))
        print('Mean Recall: {:.4f}'.format(np.mean(scores['recall'])))
        print('Mean Accuracy: {:.4f}'.format(np.mean(scores['accuracy'])))
        print('Mean ROC AUC: {:.4f}'.format(np.mean(scores['roc_auc'])))
        print()
    
    return scores

def spatial_cv(clf, X, y, splits, verbose=0):
    
    scores = {
        'f1_score' : [],
        'kappa' : [],
        'precision' : [],
        'recall' : [],
        'accuracy' : [],
        'roc_auc' : []
    }
    
    cv, areas = get_cv_iterator(splits)
    for (train_indices, test_indices), area in zip(cv, areas):
        
        # Split into train and test splits
        X_train, X_test = X.loc[train_indices], X.loc[test_indices]
        y_train, y_test = y.loc[train_indices], y.loc[test_indices]
        
        # Fit classifier to training set
        pipe_clf = Pipeline([
            ('scaler',  MinMaxScaler()),
            ('classifier', clf)
        ])
        pipe_clf.fit(X_train, y_train)
        
        # Evaluate model
        if verbose > 0: print('\nTest Set: {}'.format(area))
        accuracy, f1_score_, precision, recall, kappa, auc_score = evaluate_model(
            pipe_clf, X_test, y_test, verbose=verbose
        )
        
        # Save results
        scores['f1_score'].append(f1_score_)
        scores['kappa'].append(kappa)
        scores['precision'].append(precision)
        scores['recall'].append(recall)
        scores['accuracy'].append(accuracy)
        scores['roc_auc'].append(auc_score)
        
    if verbose > 0:
        print()
        print('Mean F1 Score: {:.4f}'.format(np.mean(scores['f1_score'])))
        print('Mean Kappa statistic: {:.4f}'.format(np.mean(scores['kappa'])))
        print('Mean Precision: {:.4f}'.format(np.mean(scores['precision'])))
        print('Mean Recall: {:.4f}'.format(np.mean(scores['recall'])))
        print('Mean Accuracy: {:.4f}'.format(np.mean(scores['accuracy'])))
        print('Mean ROC AUC: {:.4f}'.format(np.mean(scores['roc_auc'])))
        print()
    
    return scores

def resample(data, num_neg_samples, random_state):    

    neg_dist = {
        'Formal settlement': 0.4, 
        'Unoccupied land': 0.6
    }
    
    data_area = []
    for area in data['area'].unique():
        neg_samples = data[
            (data['area'] == area) 
            & (data['target'] != 1)
        ]
        
        # Formal Settlements
        formal_samples = neg_samples[
            data['target'] == 2
        ]
            
        n_formal_samples = int(
            num_neg_samples*neg_dist[VALUE_CODES[2]]
        )
        if len(formal_samples) < n_formal_samples:
            n_formal_samples = len(formal_samples)
            
        formal_samples = formal_samples.sample(
            n_formal_samples, 
            replace=False, 
            random_state=random_state
        )
        data_area.append(formal_samples)
        
        # Unoccupied Land
        land_samples = neg_samples[
            data['target'] == 3
        ]
        
        n_land_samples = num_neg_samples - n_formal_samples
        land_samples = land_samples.sample(
            n_land_samples, 
            replace=False, 
            random_state=random_state
        )
        data_area.append(land_samples)

    pos_samples = data[data['target'] == 1]
    data_area.append(pos_samples)
    data = pd.concat(data_area)
    data = data.reset_index(drop=True)
    
    return data