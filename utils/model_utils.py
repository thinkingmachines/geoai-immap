import os
import itertools
from operator import itemgetter

import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import pandas as pd
import numpy as np
import string

from matplotlib.lines import Line2D
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

import random
SEED = 42

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

def resample(data, num_neg_samples, random_state):  
    """
    Resamples the dataset as follows:
        Positive samples: Due to the sparsity of postive data samples,
            we included all positive samples per municipality.
        Negative samples: We resample n negative samples, 40% being 
            formal settlements and 60% being unoccupied land areas.
    
    Args:
        data (pd.DataFrame) : The dataframe containing the data to be resampled
        num_neg_samples (int) : The number of negative samples 
        random_state (int) : The random state or seed for reproducibility
        
    Returns:
        data (pd.DataFrame) : The resulting resampled pandas dataframe 
    """

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

def get_hyperparameters(model):
    """
    Generates the models with different hyperparameters to be trained and evaluated
    using spatial cross validation.
    
    Args:
        model (str) : A string indicating the model or classifier to fetch hyperparameters
                      for. Supported models include 'logistic_regression', 'random_forest',
                      and 'linear_svc'.
        
    Returns:
        models (list) : A list of models, where each model is instantiated using different
                        hyperparameter settings.
        labels (list) : A list of labels indicating the corresponding model hyperparameters
                        in string format. The labels are used for plotting charts and file
                        naming schemes.
    """
    
    if model == 'logistic_regression':
        param_grid = {
            'penalty' : ['l2', 'l1'],
            'C' : [0.001, 0.01, 0.1, 1]
        }
        params = list(itertools.product(*[param_grid[param] for param in param_grid]))
        models, labels = [], []
        for param in params:
            models.append(LogisticRegression(penalty=param[0], C=param[1]))
            labels.append('penalty={}, C={:.3f}'.format(param[0], param[1]))
            
        return models, labels
    
    if model == 'linear_svc':
        param_grid = {
            'C' : [0.001, 0.01, 0.1, 1],
        }
        params = list(itertools.product(*[param_grid[param] for param in param_grid]))

        models, labels = [], []
        for param in params:
            models.append(CalibratedClassifierCV(LinearSVC(C=param[0], random_state=SEED)))
            labels.append('C={:.3f}'.format(param[0]))
            
        return model, labels
    
    if model == 'random_forest':
        param_grid = {
            'n_estimators': [100, 300, 500, 800, 1200],
            'max_depth': [5, 8, 12],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 5, 10] 
        }
        params = list(itertools.product(
            *[param_grid[param] for param in param_grid]
        ))
        
        # Randomly sample 5 parameter settings due to 
        # the large number of combinations
        random.seed(SEED) 
        params = random.sample(params, 5)

        models, labels = [], []
        for param in params:
            models.append(
                RandomForestClassifier(
                    n_estimators=param[0], 
                    max_depth=param[1], 
                    min_samples_split=param[2],
                    min_samples_leaf=param[3],
                    random_state=SEED
                )
            )
            labels.append(
                'n_estimators={}, max_depth={}, min_samples_split={}, min_samples_leaf={}'.format(
                    param[0], param[1], param[2], param[3]
                )
            )
            
        return models, labels

def spatial_cv(clf, X, y, splits, grids, label=None):
    """
    Implements spatial cross validation (CV) in the form of a 
    leave-one-municipality-out CV scheme. 
    
    Args:
        clf (sklearn classifier) : The machine learning classifier to be evaluated
        X (pd.DataFrame) : A pandas dataframe containing the features as columns
        y (pd.DataFrame) : A pandas dataframe containing the target variable
        splits (pd.DataFrame) : A pandas dataframe having column 'area' which 
                                contains the municipalities to split by
        grids (pd.Series) : A pandas series containing the grid unique id ('uid').
                            This is used for calculating settlement-level performance.
        label (str) : A string describing more information about the model, e.g.
                      model hyperparameters or model name. The label is printed
                      in the progress bar (default None)
        
    Returns:
        results (pd.DataFrame) : A pandas dataframe containing the y_preds and y_tests,
                                 along with the grid id and the corresponding area of 
                                 corresponding pixel.
    """
    
    results = {
        'grid_id': [],
        'area': [],
        'y_pred': [],
        'y_test': []
    }
    
    areas = list(splits.area.unique())
    area_names = [AREA_CODES[x] for x in areas]
    
    pbar = tqdm(zip(area_names, areas), total=len(areas))
    for area_name, area in pbar:    
        
        # Print progress bar
        if label != None:
            pbar.set_description(
                "Parameters: {} | Processing {}".format(label, area_name)
            )
        else:
            pbar.set_description("Processing {}".format(area_name))
        
        # Get train and test indices
        train_indices = splits[splits.area != area].index.values.astype(int)
        test_indices = splits[splits.area == area].index.values.astype(int)
        
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
        
        # Concatenate all predictions
        area_list = [area for x in range(len(y_test))]
        results['area'].extend(area_list)
        results['grid_id'].extend(y_grids)
        results['y_test'].extend(y_test)
        results['y_pred'].extend(y_pred)
    
    results = pd.DataFrame(results)
    
    return results       

def get_grid_level_results(results):
    """
    Calculates grid or settlement-level prediction by aggregating the pixels
    within each grid (for negative samples) or informal settlement polygon 
    (for positive samples). 
    
    Specifically, we calculte the mean of the top 10% pixels per grid or polygon.
    
    Args:
        results (pd.DataFrame) : A pandas dataframe containing the y_preds, y_tests,
                                 and grid id to be aggregated by. 
    Returns:
        results_grid (pd.DataFrame) : A pandas dataframe containing the predictions,
                                      aggregated by grid_id. 
    """
    
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
    
    results_grid = results.groupby('grid_id')[['grid_id', 'area', 'y_pred', 'y_test']].agg({
        'grid_id': get_mode,
        'area' : get_mode,
        'y_pred': get_top_percentile,
        'y_test': get_mode
    })
    
    return results_grid

def calculate_precision_recall(results):
    """
    Calculates the cumulative precision and recall at the top x% of predictions. 
    
    Args:
        results (pd.DataFrame) : A pandas dataframe containing the y_preds, y_tests,
                                 grid ids, and corresponding area ids per pixel.
    Returns:
        metrics (pd.DataFrame) : A pandas dataframe containing the precision and recall
                                 at every x% of predictions, sorted in descending order
    """
    
    def calculate_precision(y_test):
        if len(y_test) == 0: return 0
        precision = sum(y_test)/len(y_test)
        return precision

    def calculate_recall(y_test, total_pos):
        recall = sum(y_test)/total_pos
        return recall
        
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

def evaluate_model(models, labels, X, y, splits, grids):
    """
    Returns pixel-level and grid-level predictions and performance metrics.
    
    Args:
        models (list) : A list of models, where each model is instantiated using different
                        hyperparameter settings.
        labels (list) : A list of labels indicating the corresponding model hyperparameters
                        in string format. 
        X (pd.DataFrame) : A pandas dataframe containing the features as columns
        y (pd.DataFrame) : A pandas dataframe containing the target variable
        splits (pd.DataFrame) : A pandas dataframe having column 'area' which 
                                contains the municipalities to split by
        grids (pd.Series) : A pandas series containing the grid unique id ('uid').
                            This is used for calcuating settlement-level performance.
    
    Returns:
        output (Python dict) : A Python dictionary containing the following keys and values:
            - 'labels' : A list of string labels describing each model
            - 'pixel_preds': A list of pandas dataframes containing the pixel-level 
                             predictions per model
            - 'grid_preds' : A list of pandas dataframes containing grid-level 
                             predictions per model
            - 'pixel_metrics' : A list of pandas dataframes containing pixel-level 
                                metrics per model
            - 'grid_metrics' : A list of pandas dataframes containing the grid-level
                               metrics per model
    """    
    
    output = {
        'labels' : [],
        'pixel_preds' : [],
        'grid_preds' : [],
        'pixel_metrics' : [],
        'grid_metrics' : []
    }
    
    for model, label in zip(models, labels):
        pixel_preds = spatial_cv(
            model, 
            X, 
            y, 
            splits=splits, 
            grids=grids, 
            label=label
        )
        grid_preds = get_grid_level_results(pixel_preds)

        pixel_metrics = calculate_precision_recall(pixel_preds)
        grid_metrics = calculate_precision_recall(grid_preds)
        
        output['labels'].append(label)
        output['pixel_preds'].append(pixel_preds)
        output['grid_preds'].append(grid_preds)
        output['pixel_metrics'].append(pixel_metrics)
        output['grid_metrics'].append(grid_metrics)
        
    return output  

def plot_precision_recall(
    results, 
    labels=None, 
    indexes=None, 
    level='grid', 
    legend_title=None, 
    subtitle=''
):
    """
    Plots the precision-recall curves for each model output in the list of results. 
    
    Args:
        results (list) : A list of model results (pandas dataframes) to plot 
        labels (list) : A list of labels that correspond with each model 
                        result (default: None)
        indexes (list) : A list of integers specifying the result at the specified 
                         indices to plot (default: None)
        level (str) : A string specifying the level of aggregation. 
                      Values can be either 'pixel' or 'grid'
        legend_title (str) : A string specifying the legend title (default: None)
        subtitle (str) : A string indicating a subtitle for the chart.
        
    Returns:
        None
    """
    
    if indexes is not None:
        results = itemgetter(*indexes)(results)
        labels = itemgetter(*indexes)(labels)
    
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    
    linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
    linestyles = linestyles[:len(results)][::-1]
    
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
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.1, 1), title=legend_title)

    ax.set_xlabel('Proportion of {}s'.format(level.title()))
    plt.suptitle('Precision-Recall at X Proportion of {}s'.format(level.title()), fontsize=12)
    plt.title(subtitle,fontsize=11)
    plt.show()

def evaluate_model_per_area(results_):   
    """
    Generates results per municipality.
    
    Args:
        results (list) : A list of results (pandas dataframes) per model
    
    Results:
        output (Python dict) : A dictionary of dictionaries, where the keys are 
                               the municipalities and the values are the result 
                               dictionaries containing 'labels', 'pixel_preds', 
                               'grid_preds', 'pixel_metrics', and 'grid_metrics'
    """
        
    output = {
        area : {
            'labels' : [],
            'pixel_preds' : [],
            'grid_preds' : [],
            'pixel_metrics' : [],
            'grid_metrics' : []
        }
        for area in AREA_CODES
    }
    
    results = results_['pixel_preds']
    for area in AREA_CODES:
        for result in results:
            pixel_preds = result[result['area'] == area]
            grid_preds = get_grid_level_results(pixel_preds)

            pixel_metrics = calculate_precision_recall(pixel_preds)
            grid_metrics = calculate_precision_recall(grid_preds)
            
            output[area]['labels'] = results_['labels']
            output[area]['pixel_preds'].append(pixel_preds)
            output[area]['grid_preds'].append(grid_preds)
            output[area]['pixel_metrics'].append(pixel_metrics)
            output[area]['grid_metrics'].append(grid_metrics)
            
    return output

def plot_precision_recall_per_area(
    results, 
    level, 
    legend_title, 
    indexes=None
):
    """
    Plots the precision-recall curves for each model output in the list of results. 
    
    Args:
        results (list) : A list of model results (pandas dataframes) to plot 
        level (str) : A string specifying the level of aggregation. 
                      Values can be either 'pixel' or 'grid'
        legend_title (str) : A string specifying the legend title (default: None)
        indexes (list) : A list of integers specifying the result at the specified 
                         indices to plot (default: None)
        
    Results:
        area_results (dict) : A dictionary of dictionaries, where the keys are 
                              the municipalities and the values are the result 
                              dictionaries containing 'labels', 'pixel_preds', 
                              'grid_preds', 'pixel_metrics', and 'grid_metrics'
    """
    
    area_results = evaluate_model_per_area(results)
    
    if indexes is None:
        indexes = [x for x in range(len(results['labels']))]
    
    for area in AREA_CODES:
        
        plot_precision_recall(
            itemgetter(*indexes)(
                area_results[area]['{}_metrics'.format(level)]
            ), 
            labels=itemgetter(*indexes)(results['labels']), 
            level=level, 
            legend_title=legend_title,
            subtitle= '{} Municipality'.format(AREA_CODES[area])
        )
    
    return area_results

def save_results(results, output_dir, model_prefix):
    """
    Saves each dataframe in the results dictionary as a CSV file.
    
    Args:
        results: The Python dictionary that contains the results
        output_dir (str) : The output directory
        model_prefix : The model name as a prefix
        
    Returns:
        None
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for level in ['grid', 'pixel']:
        for type_ in ['preds', 'metrics']:
            
            labels = results['labels']
            preds_label = '{}_{}'.format(level, type_)
            preds = results[preds_label]
            
            for pred, label in zip(preds, labels) :
                
                param_label = label.translate(
                    str.maketrans('', '', string.punctuation.replace('.', ''))
                ).replace(' ', '_').lower()
                filename = '{}{}_{}_{}.csv'.format(
                    output_dir, 
                    model_prefix, 
                    preds_label, 
                    param_label
                )
                pred.to_csv(filename, index=False)

def load_results(labels, output_dir, model_prefix):    
    """
    Loads the CSV results as pandas dataframes
    
    Args:
        labels (list) : A list of string labels for each unique model
                        The label typically indicates the hyperparameters 
        output_dir (str) : The output directory
        model_prefix : The model name as a prefix
        
    Returns:
        results: The Python dictionary that contains the results
    """
    
    results = {
        'pixel_preds' : [],
        'grid_preds' : [],
        'pixel_metrics' : [],
        'grid_metrics' : []
    }
    results['labels'] = labels
                
    for level in ['grid', 'pixel']:
        for type_ in ['preds', 'metrics']:
            
            preds_label = '{}_{}'.format(level, type_)   
            results[preds_label] = []

            for label in labels:
                param_label = label.translate(
                    str.maketrans('', '', string.punctuation.replace('.', ''))
                ).replace(' ', '_').lower()

                filename = '{}{}_{}_{}.csv'.format(
                        output_dir, 
                        model_prefix,
                        preds_label, 
                        param_label
                    )
                preds = pd.read_csv(filename)
                results[preds_label].append(preds)
                
    return results