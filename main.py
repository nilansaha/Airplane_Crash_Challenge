import os
import sys
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from mlens.ensemble import BlendEnsemble
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from forward_selection import ForwardSelection
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
warnings.filterwarnings('ignore')


data_folder = 'Data/'
train_file = data_folder + 'train.csv'
test_file = data_folder + 'test.csv'

results_folder = 'results/'

df = pd.read_csv(train_file)
df = df.drop(['Accident_ID'], axis = 1)

target_encoder = LabelEncoder()
df['Severity'] = target_encoder.fit_transform(df['Severity'])

X = df.drop(['Severity'], axis = 1)
y = df['Severity']

numeric_features = ['Safety_Score', 'Days_Since_Inspection', 'Control_Metric', 'Total_Safety_Complaints', 'Turbulence_In_gforces', 'Cabin_Temperature', 'Max_Elevation', 'Violations', 'Adverse_Weather_Metric']
categorical_features = ['Accident_Type_Code']

numeric_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'median'))])

categorical_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))])

preprocessor = ColumnTransformer(
    transformers = [
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])


transformed_X = preprocessor.fit_transform(X)
transformed_subset_X = transformed_X[:, [8, 2, 0, 1, 14]]

def build_submission(estimator, model_name):
    test_df = pd.read_csv(test_file)
    id_ = test_df['Accident_ID']
    test_X = test_df.drop(['Accident_ID'], axis = 1)
    estimator.fit(transformed_subset_X, y)
    predictions = estimator.predict(preprocessor.transform(test_X)[:, [8, 2, 0, 1, 14]]).astype(int)
    submission = pd.DataFrame()
    submission['Accident_ID'] = id_
    submission['Severity'] = target_encoder.inverse_transform(predictions)
    submission.to_csv(results_folder + '{}.csv'.format(model_name), index = False)

def build_ensemble(proba, **kwargs):
    """Return an ensemble."""
    estimators = [DecisionTreeClassifier(),
                  LGBMClassifier(learning_rate= 0.24188855846184307, max_depth= 19, n_estimators= 582)]

    ensemble = BlendEnsemble(**kwargs)
    ensemble.add(estimators, proba=proba)
    ensemble.add_meta(LogisticRegression())
    return ensemble

ensemble = build_ensemble(proba = True)
build_submission(ensemble, 'RF_LG_LR_Proba_Subset')



