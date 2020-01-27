import numpy as np
from sklearn.model_selection import cross_val_score

class ForwardSelection:
    def __init__(self, model, 
                 min_features=None, 
                 max_features=None, 
                 scoring=None, 
                 cv=None):
        """
        ForwardSelection method for selecting the best features in a dataset in a forward stepwise manner
        
        Parameters
        ----------
        model: sklearn model that will fit the data
        min_features: minimum number of features to start with
        max_features: maximum number of features to consider
        scoring: the scoring function to use
        cv: number of cross-validation steps
        """
        if min_features is None:
            self.min_features = 1
        else:
            self.min_features = min_features
        self.max_features = max_features
        self.model = model
        self.scoring = scoring
        self.cv = cv
        self.ftr_ = []
        return
    
    def fit(self, X, y):
        """
        Method to fit the ForwardSelection model to a dataset
        
        Parameters
        ----------
        X: feature set
        y: prediction/output set
        """
        feature_list = list(range(X.shape[1]))
        global_best_score = float("-inf")
        for i in range(self.max_features + 1):
            print('Selecting {} Feature'.format(i + 1))
            print(self.ftr_)
            local_scores = []
            for idx in feature_list:
                temp_X = X[:, self.ftr_ + [idx]]
                score = np.mean(cross_val_score(self.model, temp_X, y, scoring = self.scoring, cv = self.cv))
                local_scores.append(score)
            best_local_feature = np.argmax(local_scores)
            if max(local_scores) < global_best_score and len(self.ftr_) >= self.min_features:
                break
            else:
                self.ftr_.append(feature_list[best_local_feature])
                feature_list.pop(best_local_feature)
                global_best_score = max(local_scores)
        
    def transform(self, X, y=None):
        """
        Method to transform the fitted dataset to a one with the best validation score
        
        Parameters
        ----------
        X: the feature set
        y: the prediction/output set
        """
        return X[:, self.ftr_]