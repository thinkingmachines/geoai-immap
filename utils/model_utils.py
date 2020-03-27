import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    recall_score,
    precision_score,
    classification_report,
    balanced_accuracy_score
)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
  
    f1_score_ = f1_score(y_test, y_pred, pos_label=1, average='binary')  
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1, average='binary') 
    recall = recall_score(y_test, y_pred, pos_label=1, average='binary') 
  
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred)) 

    return accuracy, f1_score_, precision, recall

def geospatialcv(data, features, label, clf):
    data = data.fillna(0)
    accuracies, f1_scores, precisions, recalls = [], [], [], []
    classifiers = []
    
    area_code = {0: 'maicao', 1:'riohacha', 2:'uribia'}
    for area in data.area.unique():
        print('Testing on: {}'.format(area_code[area]))
        train = data[data.area != area]
        test = data[data.area == area]
        
        X_train, X_test = train[features], test[features]
        y_train, y_test = train[label], test[label]
        clf.fit(X_train, y_train)
        
        acc, f1, prec, rec = evaluate_model(clf, X_test, y_test)
        
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
        classifiers.append(clf)
    
    print('Average Accuracy: {:.2f}'.format(np.mean(accuracies)))
    print('Average F1 Score: {:.2f}'.format(np.mean(f1_scores)))
    print('Average Precision: {:.2f}'.format(np.mean(precisions)))
    print('Average Recall: {:.2f}'.format(np.mean(recalls)))
    return classifiers