"""Helper functions"""
import csv
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import mutual_info_regression


def load_config():
	parent = Path(Path.cwd()).parent
	config_path = os.path.join(parent, 'project_2', 'config.json')
	with open(config_path, 'r') as f:
		config = json.load(f)
	return config


def make_mi_scores(X, y):
	X = X.copy()
	for colname in X.select_dtypes(["object", "category"]):
		X[colname], _ = X[colname].factorize()
	# All discrete features should now have integer dtypes
	discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
	mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
	mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
	mi_scores = mi_scores.sort_values(ascending=False)
	return mi_scores


def plot_mi_scores(scores):
	scores = scores.sort_values(ascending=True)
	width = np.arange(len(scores))
	ticks = list(scores.index)
	plt.barh(width, scores)
	plt.yticks(width, ticks)
	plt.title("Mutual Information Scores")


def drop_uninformative(df, mi_scores):
	return df.loc[:, mi_scores > 0.0]


def load_train_data():
	config = load_config()
	X_train = pd.read_csv(
		os.path.join(config['path']['train'], 'X_train.csv'),
		keep_default_na=False
	)
	y_train = pd.read_csv(
		os.path.join(config['path']['train'], 'y_train.csv'),
		keep_default_na=False
	)

	return X_train, y_train


def load_test_data():
	config = load_config()
	X_test = pd.read_csv(
		os.path.join(config['path']['test'], 'X_test.csv'),
		keep_default_na=False
	)
	y_test = pd.read_csv(
		os.path.join(config['path']['test'], 'y_test.csv'),
		keep_default_na=False
	)

	return X_test, y_test


def save_result(fields):
	config = load_config()
	with open(os.path.join(config['path']['result'], 'result.csv'), 'a') as f:
		writer = csv.DictWriter(f, fieldnames=['model', 'validation_best_RMSE', 'validation_best_params'])
		writer.writerow(fields)


def save_model(model_fit, model_name):
	config = load_config()
	joblib.dump(model_fit, os.path.join(config['path']['result'], f'{model_name}.pkl'))


def load_model(model_name):
	config = load_config()
	return joblib.load(os.path.join(config['path']['result'], f'{model_name}.pkl'))
