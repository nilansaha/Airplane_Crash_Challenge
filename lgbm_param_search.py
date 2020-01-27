import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from hyperopt import fmin, tpe, hp, anneal, Trials
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
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

def object_function(params, random_state = 40, cv = 3, X = transformed_X, y = y):
    params = {'n_estimators': int(params['n_estimators']), 
              'max_depth': int(params['max_depth']), 
             'learning_rate': params['learning_rate']}

    estimator = LGBMClassifier(random_state = random_state, **params)
    error_score = 1 - np.mean(cross_val_score(estimator, X, y, cv = cv))
    return error_score

space={'n_estimators': hp.quniform('n_estimators', 100, 2000, 1),
       'max_depth' : hp.quniform('max_depth', 2, 20, 1),
       'learning_rate': hp.loguniform('learning_rate', -5, 0)
      }

trials = Trials()
random_state = 40

best = fmin(fn = object_function,
            space = space,
            algo = tpe.suggest,
            max_evals = 100,
            trials = trials,
            rstate = np.random.RandomState(random_state))

print('-' * 60)
print(best)