# Databricks notebook source
import pandas as pd
import xgboost as xgb
import mlflow
from sklearn import datasets

# Branch Exp1, v1
mlflow.autolog(disable=False)

iris = datasets.load_iris()
X = pd.DataFrame(iris.data[:, :2], columns=iris.feature_names[:2])
y = iris.target
dataset = xgb.DMatrix(X, y)
xgb.train({"num_class": 3, "eval_metric": "mlogloss"}, dataset, evals=[(dataset, "train")])
