import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    recall_score,
    precision_score,
    classification_report,
    balanced_accuracy_score,
    cohen_kappa_score
)
from sklearn.feature_selection import (
    SelectKBest,
    RFE
)
from sklearn.preprocessing import MinMaxScaler

def get_rfe_features(X, y, clf, num_features):
    rfe_selector = RFE(estimator=clf, n_features_to_select=num_features, step=10, verbose=5)

    X_norm = MinMaxScaler().fit_transform(X)
    rfe_selector.fit(X_norm, y)
    rfe_support = rfe_selector.get_support()
    rfe_features = X.loc[:, rfe_support].columns.tolist()
    
    return rfe_features

def evaluate_model(model, X_test, y_test, scaler=None):
    if scaler != None:
        X_test = scaler.transform(X_test)
        
    y_pred = model.predict(X_test)
  
    f1_score_ = f1_score(y_test, y_pred, pos_label=1, average='binary')  
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1, average='binary') 
    recall = recall_score(y_test, y_pred, pos_label=1, average='binary') 
    kappa = cohen_kappa_score(y_test, y_pred)
  
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred)) 
    print('Accuracy: {:.4f}'.format(accuracy))
    print('F1 Score: {:.4f}'.format(f1_score_))
    print('Precision: {:.4f}'.format(precision))
    print('Recall: {:.4f}'.format(recall))
    print('Kappa Statistics: {:.4f}'.format(kappa))

    return accuracy, f1_score_, precision, recall, kappa

def geospatialcv(data, features, label, clf, scale=False):
    data = data.fillna(0)
    accuracies, f1_scores, precisions, recalls, kappas = [], [], [], [], []
    classifiers = []
    
    area_code = {0: 'maicao', 1:'riohacha', 2:'uribia'}
    for area in data.area.unique():
        print('\nTest set: {}'.format(area_code[area].upper()))
        train = data[data.area != area]
        test = data[data.area == area]
        
        X_train, X_test = train[features], test[features]
        y_train, y_test = train[label], test[label]
        
        scaler = None
        if scale:
            scaler = MinMaxScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            clf.fit(X_train, y_train)
        else:
            clf.fit(X_train, y_train)
        
        acc, f1, prec, rec, kappa = evaluate_model(clf, X_test, y_test, scaler=scaler)
        
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
        kappas.append(kappa)
        classifiers.append(clf)
    
    print()
    print('Average Accuracy: {:.4f}'.format(np.mean(accuracies)))
    print('Average F1 Score: {:.4f}'.format(np.mean(f1_scores)))
    print('Average Precision: {:.4f}'.format(np.mean(precisions)))
    print('Average Recall: {:.4f}'.format(np.mean(recalls)))
    print('Average Kappa statistic: {:.4f}'.format(np.mean(kappas)))
    return classifiers