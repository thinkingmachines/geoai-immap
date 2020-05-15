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

def evaluate_per_area(results):
    areas = results[0]['areas'].unique()
    
    output = {
        area : {
            'labels' : [],
            'pixel_preds' : [],
            'grid_preds' : [],
            'pixel_metrics' : [],
            'grid_metrics' : []
        }
        for area in areas
    }
    
    for area in areas:
        for results in results:
            subdata = result[result['area'] == area]

            pixel_preds = subdata['y_pred']
            grid_preds = get_grid_level_results(pixel_preds)

            pixel_metrics = calculate_precision_recall(pixel_preds)
            grid_metrics = calculate_precision_recall(grid_preds)

            output[area]['labels'].append(label)
            output[area]['pixel_preds'].append(pixel_preds)
            output[area]['grid_preds'].append(grid_preds)
            output[area]['pixel_metrics'].append(pixel_metrics)
            output[area]['grid_metrics'].append(grid_metrics)
            
    return output

def evaluate_model(models, labels, X, y, splits, grids, verbose=2):
    output = {
        'labels' : [],
        'pixel_preds' : [],
        'grid_preds' : [],
        'pixel_metrics' : [],
        'grid_metrics' : []
    }
    
    for model, label in tqdm(zip(models, labels), total=len(models)):
        pixel_preds = spatial_cv(model, X, y, splits=splits, grids=grids, verbose=2)
        grid_preds = get_grid_level_results(pixel_preds)

        pixel_metrics = calculate_precision_recall(pixel_preds)
        grid_metrics = calculate_precision_recall(grid_preds)
        
        output['labels'].append(label)
        output['pixel_preds'].append(pixel_preds)
        output['grid_preds'].append(grid_preds)
        output['pixel_metrics'].append(pixel_metrics)
        output['grid_metrics'].append(grid_metrics)
        
    return output
    
def plot_precision_recall(results, labels=None, level='grid'):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    
    linestyles_named = ['solid', 'dashed', 'dashdot', 'dotted']
    linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),
     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
    
    linestyles = linestyles_named + [x[1] for x in linestyle_tuple]
    linestyles = linestyles[:len(results)]
    
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
    
    if labels != None:
        legend_elements = [
            Line2D([0], [0], color='black', linestyle=linestyle, label=label)
            for linestyle, label in zip(linestyles, labels)
        ]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.6, 1))

    ax.set_xlabel('Proportion of {}s'.format(level.title()))
    plt.title('Precision-Recall at X Proportion of {}s'.format(level.title()))
    plt.show()

def calculate_precision(y_test):
    precision = sum(y_test)/len(y_test)
    return precision

def calculate_recall(y_test, total_pos):
    recall = sum(y_test)/total_pos
    return recall

def calculate_precision_recall(results):
    metrics = {
        'precision' : [],
        'recall' : []
    }
    
    results = results.sort_values('y_pred', ascending=False)
    results = results.reset_index(drop=True)

    total_pos = sum(results['y_test'])
    
    for x in range(1, 100):
        top_n = int(len(results)*(x/100))
        subdata = results.iloc[:top_n, :]
        y_test = subdata['y_test']
        
        precision = calculate_precision(y_test)
        recall = calculate_recall(y_test, total_pos)
        
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
    
    metrics = pd.DataFrame(metrics)
    
    return metrics

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

def spatial_cv(clf, X, y, splits, grids, verbose=0):
    
    results = {
        'grid_id': [],
        'area': [],
        'y_pred': [],
        'y_test': []
    }
    
    cv, areas = get_cv_iterator(splits)
    for (train_indices, test_indices), area in zip(cv, areas):
        
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
        area_list = [area for x in range(len(y_test))]
        results['area'].extend(area_list)
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