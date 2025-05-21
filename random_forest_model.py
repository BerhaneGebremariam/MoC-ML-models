from __future__ import annotations
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,log_loss
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, GroupKFold
from data_split import data_splits
import warnings
warnings.filterwarnings('ignore')

class RandomForestModel:
    """
    This class main class for the Random Forest model. It contains the Random Forest model and the hyperparameters
    to be tuned. The class also contains the method to train the model and evaluate the model. the random_forest_model
    method is the main method to train the model and evaluate the model. The method takes in the split_type, metrics_type,
    in_sample_validation, and tuning_type as parameters. If finals returns dictionary of performance metrics, for different model settings

    """

    def __init__(self, cv:int = 5,n_iter : int = 20,random_state : int = 42, data_splits: list =data_splits()):

        self.cv = cv
        self.n_iter = n_iter
        self.random_split = data_splits[0]
        self.random_val_grouped_test = data_splits[1]
        self.grouped_split = data_splits[2]
        self.random_state = random_state
        self.param_grid = {
            'n_estimators': [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False],
            'criterion': ['gini', 'entropy'],
        }

    def random_forest_model(self, split_type='random', metrics_type='discrete', in_sample_validation=False, tuning_type='randomized search'):
        
        if split_type == 'random':
            X_train_val, X_train, X_val, X_test, y_train_val, y_train, y_val, y_test = self.random_split
        elif split_type == 'grouped':
            X_train_val, X_train, X_val, X_test, y_train_val, y_train, y_val, y_test = self.grouped_split
        else:
            X_train_val, X_train, X_val, X_test, y_train_val, y_train, y_val, y_test = self.random_val_grouped_test

        group_train_val = X_train_val['ResponseId']
        group_train = X_train['ResponseId']
        X_train_val_1 = X_train_val.drop(columns=['ResponseId'], axis=1)
        X_train_1 = X_train.drop(columns=['ResponseId'], axis=1)

        scoring = 'neg_log_loss' if metrics_type == 'continuous' else 'accuracy'   #f1_macro was considered but no improvement was observed
        # without walking mode also didn't improve the model considerably the accuracy was around 0.72 for the random split
        cv = GroupKFold(n_splits=self.cv) if split_type == 'grouped' else KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        rfc = RandomForestClassifier(random_state=self.random_state)
        if tuning_type == 'gridsearch':
            rf_random = GridSearchCV(estimator=rfc, param_grid=self.param_grid, cv=cv, verbose=1, n_jobs=1, scoring=scoring)
        else:
            rf_random = RandomizedSearchCV(estimator=rfc, param_distributions=self.param_grid, n_iter=self.n_iter, cv=cv, verbose=1, random_state=self.random_state, n_jobs=1, scoring=scoring)

        if in_sample_validation:
            if split_type == 'grouped':
                rf_random.fit(X_train_val_1, y_train_val, groups=group_train_val)
            else:
                rf_random.fit(X_train_val_1, y_train_val)
        else:
            if split_type == 'grouped':
                rf_random.fit(X_train_1, y_train, groups=group_train)
            else:
                rf_random.fit(X_train_1, y_train)

        best_params = rf_random.best_params_
        print('Best parameters:', best_params)
        best_score = rf_random.best_score_
        print('Best score:', best_score)
        
        rf = rf_random.best_estimator_

        if metrics_type == 'continuous':
            y_pred_test = rf.predict_proba(X_test)
            y_pred_train = rf.predict_proba(X_train_1)
            y_pred_val = rf.predict_proba(X_val)
            y_neg_log_loss_test = -log_loss(y_test, y_pred_test)
            y_neg_log_loss_val = -log_loss(y_val, y_pred_val)

        else:
            y_pred_test = rf.predict(X_test)
            y_pred_train = rf.predict(X_train_1)
            y_pred_val = rf.predict(X_val)

            conf_matrix_test = confusion_matrix(y_test, y_pred_test)
            conf_matrix_train = confusion_matrix(y_train, y_pred_train)
            conf_matrix_val = confusion_matrix(y_val, y_pred_val)
    
            performance_test = classification_report(y_test, y_pred_test, output_dict=True)
            performance_train = classification_report(y_train, y_pred_train, output_dict=True)
            performance_val = classification_report(y_val, y_pred_val, output_dict=True)
    
            accuracy_test = accuracy_score(y_test, y_pred_test)
            accuracy_train = accuracy_score(y_train, y_pred_train)
            accuracy_val = accuracy_score(y_val, y_pred_val)

            
            
        if metrics_type == 'continuous':
            to_return = {
                'y_pred_test': y_pred_test,
                'y_pred_train': y_pred_train,
                'y_pred_val': y_pred_val,
                'rf': rf,
                'best_params': best_params,
                'best_score': best_score,
                'loss_test': y_neg_log_loss_test,
                'loss_val': y_neg_log_loss_val

            }
        else:
            to_return = {
                'conf_matrix_test': conf_matrix_test,
                'conf_matrix_train': conf_matrix_train,
                'conf_matrix_val': conf_matrix_val,
                'performance_test': performance_test,
                'performance_train': performance_train,
                'performance_val': performance_val,
                'y_pred_test': y_pred_test,
                'y_pred_train': y_pred_train,
                'y_pred_val': y_pred_val,
                'rf': rf,
                'best_params': best_params,
                'best_score': best_score,
                'accuracy_test': accuracy_test,
                'accuracy_train': accuracy_train,
                'accuracy_val': accuracy_val
            }
            
        return to_return

def main():
    rf = RandomForestModel()
    rf.random_forest_model()


if __name__ == '__main__':
    main()