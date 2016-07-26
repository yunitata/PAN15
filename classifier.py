from __future__ import division
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
import os
from collections import OrderedDict
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import ExtraTreeClassifier



target = []
feature_value = []
feature_value_test = []
ttrain = []


def classify(dict_tfidf, dict_target, model_path_dir):
    for key in dict_tfidf:
        name = dict_target.get(key, None)
        name2 = dict_tfidf.get(key, None)
        if name:
            target.append(name)
            feature_value.append(name2)
    for i in range(len(target)):
        if target[i] == 'Y':
            ttrain.append(1)
        else:
            ttrain.append(0)
    x = np.array(feature_value)
    print(x)
    y = np.array(ttrain)
    sv = SVC(C=1, probability=True, random_state=20)
    pipe = Pipeline([('feature_selection', ExtraTreeClassifier()), ('classification', sv)])
    param_grid = {'classification__C': [1, 10, 100, 1000], 'classification__gamma': [0.001, 0.0001],
    'classification__kernel': ['rbf', 'linear']}
    clf = GridSearchCV(pipe, param_grid=param_grid, cv=4, verbose=10)  # initially cv=4
    clf.fit(x, y)
    path_save = os.path.join(model_path_dir, 'model.pkl')
    joblib.dump(clf, path_save)
    path_model = os.path.abspath(path_save)
    return path_model


def classifier_predict(model_path, dict_test, output_path_dir):
    proba = []
    output = {}
    for key in dict_test:
        val = dict_test.get(key, None)
        if val:
            feature_value_test.append(val)

    z = np.array(feature_value_test)
    for subdir, dirs, files in os.walk(model_path):
        for file_ in files:
            if file_ == 'model.pkl':
                model_file = subdir + os.path.sep + file_
    estimator = joblib.load(model_file)
    pred = estimator.predict_proba(z)[:, 1]
    pred_accuracy = estimator.predict(z)
    '''
    for i in range(len(pred)):
        if pred[i] >= 0.48 and pred[i] <= 0.5:
            proba.append(0.6)
        else:
            proba.append(pred[i])
    '''
    k = 0
    for key in dict_test.keys():
        output[key] = pred[k]
        k += 1
    sorted_output = OrderedDict(sorted(output.items(), key=lambda s: s[0]))
    ofile = os.path.join(output_path_dir, "answers.txt")
    output_file = open(ofile, 'w')
    for key, value in sorted_output.iteritems():
        output_file.write(key + "\t" + str(round(value, 3)) + "\n")
    output_file.close()
    path_output = os.path.abspath("answers.txt")
    return path_output

