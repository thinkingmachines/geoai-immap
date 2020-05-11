import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    recall_score,
    precision_score,
    classification_report,
    cohen_kappa_score,
    roc_auc_score,
    precision_recall_curve
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
from scipy import stats

from matplotlib.lines import Line2D

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

def plot_precision_recall(results, classifiers=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted'][:len(results)]
    for result, linestyle in zip(results, linestyles):
        proportion = (result.index+1)/len(result)
        ax.plot(proportion, result['recall'], linestyle=linestyle, color='#F3A258')
        ax.plot(proportion, result['precision'], linestyle=linestyle, color='#5268B4')

    ax.tick_params(labelright=True)
    ax2.tick_params(labelright=False)
    ax2.grid(False)

    ax.set_ylabel('Precision', color='#5268B4')
    h = ax2.set_ylabel('Recall', color='#F3A258', labelpad=45)
    h.set_rotation(-90)
    
    if classifiers != None:
        legend_elements = [
            Line2D([0], [0], color='black', linestyle=linestyle, label=classifier)
            for linestyle, classifier in zip(linestyles, classifiers)
        ]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.6, 1))

    ax.set_xlabel('Proportion of Grids')
    plt.title('Precision-Recall at X Proportion of Grids')
    plt.show()

def calculate_precision(y_test):
    precision = sum(y_test)/len(y_test)
    return precision

def calculate_recall(y_test, total_pos):
    recall = sum(y_test)/total_pos
    return recall

def calculate_precision_recall(results):
    results = results.sort_values('y_pred', ascending=False)
    results = results.reset_index(drop=True)

    total_pos = sum(results['y_test'])
    results['precision'] = results['y_test'].expanding().apply(calculate_precision) 
    results['recall'] = results['y_test'].expanding().apply(lambda x: calculate_recall(x, total_pos))
        
    return results

def get_grid_level_results(results):
    def get_mode(x):
        return(stats.mode(x)[0])

    def get_top_percentile(x):
        percentile = 0.10
        top_n = int(percentile*len(x))
        x = sorted(x, reverse=True)[:top_n]
        return(np.mean(x))
    
    # Remove grids with less than 10 pixels
    grids = list(results[results['y_test'] == 1]['grid_id'].unique())
    counts = results[results['y_test'] != 1]['grid_id'].value_counts() 
    grids.extend(list(counts[counts > 10].index))
    results = results[results['grid_id'].isin(grids)]
    
    results_grid = results.groupby('grid_id')[['grid_id', 'y_pred', 'y_test']].agg({
        'grid_id': get_mode,
        'y_pred': get_top_percentile,
        'y_test': get_mode
    })
    
    results_grid = calculate_precision_recall(results_grid)
    plot_precision_recall([results_grid])
    
    return results_grid

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

def nested_spatial_cv(
    clf, 
    X, 
    y, 
    splits, 
    grids,
    param_grid, 
    search_type='grid', 
    feature_selection=True, 
    verbose=0,
    random_state=42
):
    results = {
        'grid_id': [],
        'y_pred': [],
        'y_test': []
    }
        
    outer_cv, areas = get_cv_iterator(splits)
    for (train_indices, test_indices), area in tqdm(zip(outer_cv, areas), total=len(areas)):
        
        splits_train = splits.loc[train_indices].reset_index(drop=True)    
        X_train = X.loc[train_indices].reset_index(drop=True)
        X_test = X.loc[test_indices].reset_index(drop=True)
        y_train = y.loc[train_indices].reset_index(drop=True)
        y_test = y.loc[test_indices].reset_index(drop=True)
        y_grids = grids.loc[test_indices].values
        
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
                    n_iter=5,
                    cv=inner_cv, 
                    verbose=0, 
                    scoring='f1',
                    n_jobs=-1,
                    random_state=random_state
                )
            cv.fit(X_train, y_train)
            best_estimator = cv.best_estimator_
            
        best_estimator.fit(X_train, y_train)
        
        try:
            y_pred = best_estimator.predict_proba(X_test)[:, 1]
        except:
            d = best_estimator.decision_function(X_test)
            y_pred = np.exp(d) / np.sum(np.exp(d))
            
        y_test, y_pred = list(y_test), list(y_pred)
        
        # Concatenate all out-of-fold predictions
        results['grid_id'].extend(y_grids)
        results['y_test'].extend(y_test)
        results['y_pred'].extend(y_pred)
            
    results = pd.DataFrame(results)
    return results

def spatial_cv(clf, X, y, splits, grids, verbose=0):
    
    results = {
        'grid_id': [],
        'y_pred': [],
        'y_test': []
    }
    
    cv, areas = get_cv_iterator(splits)
    for (train_indices, test_indices), area in tqdm(zip(cv, areas), total=len(areas)):
        
        # Split into train and test splits
        X_train, X_test = X.loc[train_indices], X.loc[test_indices]
        y_train, y_test = y.loc[train_indices], y.loc[test_indices]
        y_grids = grids.loc[test_indices].values
        
        # Fit classifier to training set
        pipe_clf = Pipeline([
            ('scaler',  MinMaxScaler()),
            ('classifier', clf)
        ])
        pipe_clf.fit(X_train, y_train)
            
        # Predict on test set
        y_pred = pipe_clf.predict_proba(X_test)[:, 1]
        y_test, y_pred = list(y_test), list(y_pred)
        
        # Concatenate all out-of-fold predictions
        results['grid_id'].extend(y_grids)
        results['y_test'].extend(y_test)
        results['y_pred'].extend(y_pred)
    
    results = pd.DataFrame(results)
    return results

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