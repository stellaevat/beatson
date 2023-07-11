import streamlit as st
import numpy as np
import pandas as pd
import math
import spacy
from tqdm import tqdm
from scipy.sparse import lil_matrix, vstack
from skmultilearn.dataset import load_from_arff
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# TODO: Replace with appropriate model
nlp = spacy.load("en_core_web_sm")
vectorizer = TfidfVectorizer(tokenizer=lambda x:[token.lemma_.lower() for token in nlp(x) if not (token.is_stop or token.is_punct or token.is_space)])
## TODO: Is there any way to fix random state of OVR too?
clf = OneVsRestClassifier(estimator=SVC(kernel='rbf', probability=True, random_state=42))

LABELLED, UNLABELLED, TEST = 0.6, 0.2, 0.2
ITERATIONS = 10
SELECTION = 10
metrics = {'Accuracy': accuracy_score,
           'Precision': precision_score,
           'Recall': recall_score,
           'F1': f1_score}
predict_msg = "Running prediction algorithm..."


# MMC Algorithm

@st.cache_data(show_spinner=False)
def mmc_label_prediction(X_labelled, y_labelled, X_unlabelled):
    # Training data for label-number (n) prediction
    y_probability = clf.predict_proba(X_labelled)
    y_normalized = (y_probability.T / np.sum(y_probability, axis=1)).T
    n_X_train = np.sort(y_normalized, axis=1)
    n_y_train = np.sum(y_labelled, axis=1)

    # Data to predict for label-number (n) prediction
    y_probability = clf.predict_proba(X_unlabelled)
    y_normalized = (y_probability.T / np.sum(y_probability, axis=1)).T
    n_X_predict = np.sort(y_normalized, axis=1)

    # Indices of labels from most to least probable
    y_sorted_indices = np.argsort(y_normalized, axis=1)[:, ::-1]

    # Label-number (n) prediction
    label_clf = LogisticRegression(random_state=42)
    label_clf.fit(n_X_train, n_y_train)
    n_y_predicted = label_clf.predict(n_X_predict)

    # Keep n most probable labels for each sample
    total_labels = y_labelled.shape[1]
    total_unlabelled = X_unlabelled.shape[0]
    y_predicted = np.full((total_unlabelled, total_labels), -1)

    ## Can this be vectorised?
    for i in range(total_unlabelled):
        ln = n_y_predicted[i]
        for j in y_sorted_indices[i, :ln]:
            y_predicted[i, j] = 1

    return y_predicted

@st.cache_data(show_spinner=False)
def mmc_query_selection(X_labelled, y_labelled, X_unlabelled, selection):
    y_decision = clf.decision_function(X_unlabelled)
    y_predicted = mmc_label_prediction(X_labelled, y_labelled, X_unlabelled)

    expected_loss_reduction_score = np.argsort(np.sum((1 - y_predicted * y_decision)/2, axis=1))[::-1]
    to_label = expected_loss_reduction_score[:selection]
    not_to_label = expected_loss_reduction_score[selection:]
    return to_label, not_to_label
  
# Simplified MMC Algorithm 

@st.cache_data(show_spinner=False) 
def mmc_simplified_query_selection(X_unlabelled, selection):
    y_decision = clf.decision_function(X_unlabelled)
    y_predicted = clf.predict(X_unlabelled)
    y_predicted[y_predicted < 1] = -1

    expected_loss_reduction_score = np.argsort(np.sum((1 - y_predicted * y_decision)/2, axis=1))[::-1]
    to_label = expected_loss_reduction_score[:selection]
    not_to_label = expected_loss_reduction_score[selection:]
    return to_label, not_to_label
  
# BinMin Algorithm

@st.cache_data(show_spinner=False)
def binmin_query_selection(X_unlabelled, selection):
    y_decision = clf.decision_function(X_unlabelled)
    most_uncertain_label_score = np.argsort(np.min(np.abs(y_decision), axis=1))
    to_label = most_uncertain_label_score[:selection]
    not_to_label = most_uncertain_label_score[selection:]
    return to_label, not_to_label
 
    
# Testing version
    
@st.cache_data(show_spinner="Running algorithm on test data...")
def test_active_learning_classifier(clf, X_labelled, y_labelled, X_unlabelled, y_unlabelled, iterations=ITERATIONS, selection=SELECTION):
    clf.fit(X_labelled, y_labelled)
    for i in tqdm(range(iterations)):      
        to_label, not_to_label = binmin_query_selection(X_unlabelled, selection)

        # Pseudo-query for labels
        X_labelled = vstack((X_labelled, X_unlabelled[to_label]))
        y_labelled = np.vstack((y_labelled, y_unlabelled[to_label]))
        X_unlabelled = X_unlabelled[not_to_label]
        y_unlabelled = y_unlabelled[not_to_label]

        clf.fit(X_labelled, y_labelled)
    return clf

@st.cache_data(show_spinner="Pre-processing data...")
def test_preprocess_dataset(X, y):
    # TODO: Decide on split percentages
    X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=TEST)
    X_labelled, y_labelled, X_unlabelled, y_unlabelled = iterative_train_test_split(X_train, y_train, test_size=(LABELLED / (LABELLED + UNLABELLED)))
    
    # Vectorise samples
    X_train = vectorizer.fit_transform(X_train)
    X_labelled = vectorizer.transform(X_labelled)
    

    # Dense label arrays required for classifier
    y_labelled = y_labelled.toarray()
    y_unlabelled = y_unlabelled.toarray()
    
    return X_labelled, y_labelled, X_unlabelled, y_unlabelled, X_test, y_test

@st.cache_data(show_spinner=False)
def test():
    # TODO: Load from testing gsheet?        
    X, y = None, None
    
    X_labelled, y_labelled, X_unlabelled, y_unlabelled, X_test, y_test = preprocess_dataset(X, y)

    clf_trained = active_learning_classifier(clf, X_labelled, y_labelled, X_unlabelled, y_unlabelled, ITERATIONS, SELECTION)
    
    # TODO: get probability of each prediction too
    y_predicted = clf_trained.predict(X_unlabelled)
    
    for (metric, fn) in metrics.items():
        if fn == accuracy_score:
            print(f"{metric}: {fn(y_unlabelled, y_predicted):.3f}")
        else:
            print(f"{metric} (micro): {fn(y_unlabelled, y_predicted, average='micro'):.3f}")
            print(f"{metric} (macro): {fn(y_unlabelled, y_predicted, average='macro'):.3f}")
  
# Web-app version

@st.cache_data(show_spinner=False)
def active_learning_classifier(clf, X_labelled, y_labelled, X_unlabelled, iterations=ITERATIONS, selection=SELECTION):
    clf.fit(X_labelled, y_labelled)
    for i in tqdm(range(iterations)):      
        to_label, not_to_label = binmin_query_selection(X_unlabelled, selection)

        # Query researchers for labels
        new_labels = None
        
        y_labelled = np.vstack((y_labelled, new_labels))
        X_labelled = vstack((X_labelled, X_unlabelled[to_label]))
        X_unlabelled = X_unlabelled[not_to_label]

        clf.fit(X_labelled, y_labelled)
    return clf
    
@st.cache_data(show_spinner=predict_msg)
def predict(project_df, pub_df, text_columns, label_col, pmid_col, delimiter):
    # Need to send in only the labelled ones
    labels = set()
    for annotation in project_df[label_col]:
        labels.update(set(annotation.split(delimiter)))
    
    i = 0
    label_to_index = {}
    index_to_label = {}
    for label in labels:
        label_to_index[label] = i
        index_to_label[i] = label
        i += 1
    
    y_labelled = np.zeros((len(project_df), len(labels))
    for i, annotation in enumerate(project_df[label_col]):
        for label in annotation.split(delimiter):
            y_labelled[i, label_to_index[label]] = 1
            
    X_labelled = []
    for project in project_df:
        text = project_df[text_columns].str.join(" ")
        if project[label_col]:
            for pmid in project[label_col].split(delimiter):
                text += " " + pub_df.loc[pub_df[pub_col] == pmid].str.join(" ")
        X_labelled.append(text)
            
    # TODO: Ensure they are sparse matrices
    X_labelled = vectorizer.fit_transform(X_labelled)
    
    # TODO: Load from gsheet
    X_unlabelled = None
    X_unlabelled = vectorizer.transform(X_unlabelled)
    
    clf_trained = active_learning_classifier(clf, X_labelled, y_labelled, X_unlabelled, ITERATIONS, SELECTION)
    
    # Transform back to original labels (index_to_label)
    y_predicted = clf_trained.predict(X_unlabelled)
    # TODO: get probability/metrics of each prediction too (is it meant to be for each label separately?)
    y_probabilities = None
    
    return y_predicted, y_probabilities