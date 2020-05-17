import os
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

def spatial_cv(clf, X, y, splits, grids, label=None, verbose=0):
    
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

def calculate_precision(y_test):
    if len(y_test) == 0: return 0
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

def evaluate_model(models, labels, X, y, splits, grids, verbose=2):
    output = {
        'labels' : [],
        'pixel_preds' : [],
        'grid_preds' : [],
        'pixel_metrics' : [],
        'grid_metrics' : []
    }
    
    for model, label in zip(models, labels):
        pixel_preds = spatial_cv(model, X, y, splits=splits, grids=grids, label=label, verbose=2)
        grid_preds = get_grid_level_results(pixel_preds)

        pixel_metrics = calculate_precision_recall(pixel_preds)
        grid_metrics = calculate_precision_recall(grid_preds)
        
        output['labels'].append(label)
        output['pixel_preds'].append(pixel_preds)
        output['grid_preds'].append(grid_preds)
        output['pixel_metrics'].append(pixel_metrics)
        output['grid_metrics'].append(grid_metrics)
        
    return output  

def plot_precision_recall(results, labels=None, indexes=None, level='grid', legend_title=None, subtitle=''):
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

def evaluate_model_per_area(results, areas):    
    
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
        for result in results:
            pixel_preds = result[result['area'] == area]
            grid_preds = get_grid_level_results(pixel_preds)

            pixel_metrics = calculate_precision_recall(pixel_preds)
            grid_metrics = calculate_precision_recall(grid_preds)

            output[area]['pixel_preds'].append(pixel_preds)
            output[area]['grid_preds'].append(grid_preds)
            output[area]['pixel_metrics'].append(pixel_metrics)
            output[area]['grid_metrics'].append(grid_metrics)
            
    return output

def plot_precision_recall_per_area(results, areas, level, legend_title, indexes=None):
    area_results = evaluate_model_per_area(results['pixel_preds'], areas)
    
    if indexes is None:
        indexes = [x for x in range(len(results['labels']))]
    
    for area in areas:
        
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
    results = {
        'labels' : [],
        'pixel_preds' : [],
        'grid_preds' : [],
        'pixel_metrics' : [],
        'grid_metrics' : []
    }
    
    for label in labels:
        results['labels'].append(label)
                
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