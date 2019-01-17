# Hack Aoteroa Workshop
# Introduction to Machine Learning in Healthcare
# Workshop - helper.py

import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from shutil import copyfile

import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.externals.joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

#import yaml
#config = yaml.load(open('config.yaml'))


def get_train_test_split(config):
    """
    This function performs the following steps:
    - Reads in the data from data/labels.csv and data/files/*.csv (keep only the first 2,500 examples)
    - Generates a feature vector for each example
    - Aggregates feature vectors into a feature matrix (features are sorted alphabetically by name)
    - Performs imputation and normalization with respect to the population
    
    After all these steps, it splits the data into 80% train and 20% test. 
    
    The binary labels take two values:
        -1: survivor
        +1: died in hospital
    
    Returns the features and labesl for train and test sets, followed by the names of features.
    """
    df_labels = pd.read_csv('data/labels_subset.csv')
    df_labels = df_labels[:2500]
    IDs = df_labels['RecordID'][:2500]
    raw_data = {}
    for i in tqdm(IDs, desc='Loading files from disk'):
        raw_data[i] = pd.read_csv('data/subset_files/{}.csv'.format(i))
    features = Parallel(n_jobs=16)(delayed(generate_feature_vector)(df, config) for _, df in tqdm(raw_data.items(), desc='Generating feature vectors'))
    df_features = pd.DataFrame(features).sort_index(axis=1)
    feature_names = df_features.columns.tolist()
    X, y = df_features.values, df_labels['In-hospital_death'].values
    X = impute_missing_values(X)
    X = normalize_feature_matrix(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=3)
    return X_train, y_train, X_test, y_test, feature_names



def generate_feature_vector(df, config):
    """
    Reads a dataframe containing all measurements for a single patient
    within the first 48 hours of the ICU admission, and convert it into
    a feature vector.
    
    Args:
        df: pd.Dataframe, with columns [Time, Variable, Value]
    
    Returns:
        a python dictionary of format {feature_name: feature_value}
        for example, {'Age': 32, 'Gender': 0, 'mean_HR': 84, ...}
    """
    static_variables = config['static']
    timeseries_variables = config['timeseries']
    
    # Replace unknow values
    df = df.replace({-1: np.nan})

    # Split time invariant and time series
    static, timeseries = df.iloc[0:5], df.iloc[5:]
    static = static.pivot('Time', 'Variable', 'Value')
    
    feature_dict = static.iloc[0].to_dict()
    for variable in timeseries_variables:
        ##creates the dictionary of features, where the key corresponds 
        ## the feature name and the value the average measurement
        measurements = timeseries[timeseries['Variable'] == variable].Value
        feature_dict['mean_' + variable] = np.mean(measurements)


    
    return feature_dict



def impute_missing_values(X):
    """
    For each feature column, impute missing values  (np.nan) with the 
    population mean for that feature.
    
    Args:
        X: np.array, shape (N, d). X could contain missing values
    Returns:
        X: np.array, shape (N, d). X does not contain any missing values
    """
    col_means=np.nanmean(X,axis=0)
    inds=np.where(np.isnan(X))
    X[inds]=np.take(col_means,inds[1])
    return X


def normalize_feature_matrix(X):
    """
    For each feature column, normalize all values to range [0, 1].

    Args:
        X: np.array, shape (N, d).
    Returns:
        X: np.array, shape (N, d). Values are normalized per column.
    """
    scaler=MinMaxScaler()
    X=scaler.fit_transform(X)
    
    return X



def get_classifier(kernel='linear', C=1.0, gamma=0.0):
    """
    Return a linear/rbf kernel SVM classifier based on the given
    penalty function and regularization parameter C.
    """

    if kernel == 'linear':
        return SVC(kernel='linear', C=C)
    elif kernel == 'rbf':
        return SVC(kernel='rbf', C=C, gamma=gamma)


def performance(clf_trained, X, y_true, metric='auroc'):
    """
    Calculates the performance metric as evaluated on the true labels
    y_true versus the predicted scores from clf_trained and X.
    Input:
        clf_trained: a fitted instance of sklearn estimator
        X : (n,d) np.array containing features
        y_true: (n,) np.array containing true labels
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
    Returns:
        the performance measure as a float
    """
    
    y_pred = clf_trained.predict(X)
    y_score = clf_trained.decision_function(X)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[-1,1]).ravel()
    if metric == 'accuracy':
        return metrics.accuracy_score(y_true, y_pred)
    elif metric == 'auroc':
        return metrics.roc_auc_score(y_true, y_score)
    elif metric == 'f1_score':
        return metrics.f1_score(y_true, y_pred)
    elif metric == 'precision':
        return metrics.precision_score(y_true, y_pred)
    elif metric == 'sensitivity':
        if tp + fn > 0:
            return tp / (tp+fn)
        else:
            return 0.0
    elif metric == 'specificity':
        if tn + fp > 0:
            return tn / (tn+fp)
        else:
            return 0.0


def plot_ROC_curve(clf_trained, X, y_true):
    """
    Computes and plots the ROC curve based on the predicted and actual labels.
    
    Args:
        clf_trained: a fitted instance of sklearn estimator
        X : (n,d) np.array containing features
        y_true: (n,) np.array containing true labels
    """
    
    y_score = clf_trained.decision_function(X)
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    roc_auc=metrics.roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.show()
