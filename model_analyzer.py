from __future__ import annotations
import pandas as pd
import numpy as np
from random_forest_model import RandomForestModel
from deep_learning_model import NeuralNetwork
import tensorflow as tf
import random


class ResultAnalysis:
    split_type = ['random', 'random_val', 'grouped']
    metrics_type = ['discrete', 'continuous']
    in_sample_validation = [False, True]
    modes = ['No Moco\nPt', 'No Moco\nAuto', 'No Moco\nBike', 'No Moco\nWalk', 'Moco Pt', 'Moco Auto', 'Moco Bike',
             'Moco Walk']

    def __init__(self, cv: int = 5, n_iter: int = 20, random_state: int = 42, n_features: int = 63,
                 early_stop_patience: int = 10):
        self.cv = cv
        self.n_iter = n_iter
        self.random_state = random_state
        self.n_features = n_features
        self.early_stop_patience = early_stop_patience

        tf.random.set_seed(self.random_state)
        random.seed(self.random_state)
        np.random.seed(self.random_state)

        self.rf = RandomForestModel(cv=cv, n_iter=n_iter, random_state=random_state)
        self.nn = NeuralNetwork(cv=cv, n_iter=n_iter, n_features=n_features, early_stop_patience=early_stop_patience,
                                random_state=random_state)
        self.RF_Models = {}
        self.ANN_Models = {}


    def create_rf_models(self) -> None:
        tf.random.set_seed(self.random_state)
        random.seed(self.random_state)
        np.random.seed(self.random_state)

        model = 1
        for split in self.split_type:
            for metrics in self.metrics_type:
                self.RF_Models[f"M{model}RF"] = self.rf.random_forest_model(split_type=split, metrics_type=metrics,
                                                                            in_sample_validation=
                                                                            self.in_sample_validation[0],
                                                                            tuning_type='randomized search')
                self.RF_Models[f"M{model}RF"]['Model ID'] = f"M{model}RF"
                self.RF_Models[f"M{model}RF"]['split_type'] = split
                self.RF_Models[f"M{model}RF"]['metrics_type'] = metrics
                self.RF_Models[f"M{model}RF"]['in_sample_validation'] = self.in_sample_validation[0]

                model += 1
        for split in self.split_type:
            for metrics in self.metrics_type:
                self.RF_Models[f"M{model}RF"] = self.rf.random_forest_model(split_type=split, metrics_type=metrics,
                                                                            in_sample_validation=
                                                                            self.in_sample_validation[1],
                                                                            tuning_type='randomized search')
                self.RF_Models[f"M{model}RF"]['Model ID'] = f"M{model}RF"
                self.RF_Models[f"M{model}RF"]['split_type'] = split
                self.RF_Models[f"M{model}RF"]['metrics_type'] = metrics
                self.RF_Models[f"M{model}RF"]['in_sample_validation'] = self.in_sample_validation[1]

                model += 1
    def create_ann_models(self) -> None:

        tf.random.set_seed(self.random_state)
        random.seed(self.random_state)
        np.random.seed(self.random_state)

        model = 1
        for split in self.split_type:
            for metrics in self.metrics_type:


                self.ANN_Models[f"M{model}ANN"] = self.nn.nn_model(split_type=split, metrics_type=metrics,
                                                                   in_sample_validation=self.in_sample_validation[0])
                self.ANN_Models[f"M{model}ANN"]['Model ID'] = f"M{model}ANN"
                self.ANN_Models[f"M{model}ANN"]['split_type'] = split
                self.ANN_Models[f"M{model}ANN"]['metrics_type'] = metrics
                self.ANN_Models[f"M{model}ANN"]['in_sample_validation'] = self.in_sample_validation[0]
                model += 1
        for split in self.split_type:
            for metrics in self.metrics_type:


                self.ANN_Models[f"M{model}ANN"] = self.nn.nn_model(split_type=split, metrics_type=metrics,
                                                                   in_sample_validation=self.in_sample_validation[1])
                self.ANN_Models[f"M{model}ANN"]['Model ID'] = f"M{model}ANN"
                self.ANN_Models[f"M{model}ANN"]['split_type'] = split
                self.ANN_Models[f"M{model}ANN"]['metrics_type'] = metrics
                self.ANN_Models[f"M{model}ANN"]['in_sample_validation'] = self.in_sample_validation[1]
                model += 1

    def discrete_continuous_metrics(self) -> pd.DataFrame:

        model_types = ['RF', 'ANN']
        models_df = []

        for model_type in model_types:

            df = pd.DataFrame()
            models = []
            for i in range(1, len(self.RF_Models) + 1):
                if 'RF' == model_type:
                    models.append(self.RF_Models[f"M{i}RF"])

                else:
                    models.append(self.ANN_Models[f"M{i}ANN"])

            for model in models:
                if model['metrics_type'] == 'continuous':
                    df.loc[model['Model ID'], 'Train'] = round(model['best_score'], 3)
                    df.loc[model['Model ID'], 'Validation'] = round(model['loss_val'], 3)
                    df.loc[model['Model ID'], 'Test'] = round(model['loss_test'], 3)
                else:
                    df.loc[model['Model ID'], 'Train'] = round(model['best_score'], 4) * 100
                    df.loc[model['Model ID'], 'Validation'] = round(model['accuracy_val'], 4) * 100
                    df.loc[model['Model ID'], 'Test'] = round(model['accuracy_test'], 4) * 100
                df.loc[model['Model ID'], 'split_type'] = model['split_type']
                df.loc[model['Model ID'], 'metrics_type'] = model['metrics_type']
                df.loc[model['Model ID'], 'in_sample_validation'] = model['in_sample_validation']
                df.loc[model['Model ID'], 'Model'] = model_type
            models_df.append(df)

        return pd.concat(models_df, axis=0)

    def compare_discrete_continuous_metrics(self) -> pd.DataFrame:
        model_types = ['RF', 'ANN']
        models_df = []

        for model_type in model_types:
            df_list = []
            models = []
            if model_type == 'RF':
                for i in range(1, len(self.RF_Models) + 1):
                    models.append(self.RF_Models[f"M{i}RF"])
            else:
                for i in range(1, len(self.ANN_Models) + 1):
                    models.append(self.ANN_Models[f"M{i}ANN"])

            modes = self.modes
            index_ = [f'{i}_true' for i in modes]
            columns_ = [f'{i}_pred' for i in modes]

            for i in range(0, len(models), 2):
                df_test = pd.DataFrame(models[i]['conf_matrix_test'], index=index_, columns=columns_)
                df_val = pd.DataFrame(models[i]['conf_matrix_val'], index=index_, columns=columns_)
                df_test['Actual Mode Share Test'] = df_test.sum(axis=1)
                df_test.loc['Discrete Mode Share Test'] = df_test.sum()
                df_val['Actual Mode Share Val'] = df_val.sum(axis=1)
                df_val.loc['Discrete Mode Share Val'] = df_val.sum()

                df_prob_test = pd.DataFrame(models[i + 1]['y_pred_test'], columns=columns_)
                df_prob_val = pd.DataFrame(models[i + 1]['y_pred_val'], columns=columns_)
                df_prob_val.loc['Pro. Mode Share Val'] = df_prob_val.sum()
                df_prob_test.loc['Pro. Mode Share Test'] = df_prob_test.sum()

                col_1 = pd.DataFrame(df_test['Actual Mode Share Test'].values, columns=['Actual Mode\nShare Test'])

                col_2 = pd.DataFrame(df_prob_test.loc['Pro. Mode Share Test'].values, columns=['Pro. Mode\nShare Test'])

                col_3 = pd.DataFrame(df_test.loc['Discrete Mode Share Test'].values,
                                     columns=['Discrete Mode\nShare Test'])

                col_4 = pd.DataFrame(df_val['Actual Mode Share Val'].values, columns=['Actual Mode\nShare Val'])
                col_5 = pd.DataFrame(df_prob_val.loc['Pro. Mode Share Val'].values, columns=['Pro. Mode\nShare Val'])

                col_6 = pd.DataFrame(df_val.loc['Discrete Mode Share Val'].values, columns=['Discrete Mode\nShare Val'])

                df_concat = pd.concat([col_1, col_2, col_3, col_4, col_5, col_6], axis=1)
                df_concat.index = modes + ['Total']
                df_concat.drop('Total', axis=0, inplace=True)
                df_concat['Model Comparison'] = f"{models[i]['Model ID']} vs {models[i + 1]['Model ID']}"
                df_concat['Model'] = model_type

                df_list.append(df_concat)

            models_df.append(pd.concat(df_list, axis=0))

        return pd.concat(models_df, axis=0)

def main():
    ra = ResultAnalysis(cv=2, n_iter=1, random_state=42, n_features=63, early_stop_patience=5)
    discrete_continuous_metrics_rf = ra.discrete_continuous_metrics()
    compare_discrete_continuous_metrics_rf = ra.compare_discrete_continuous_metrics()
    print(discrete_continuous_metrics_rf)
    print(compare_discrete_continuous_metrics_rf)



if __name__ == '__main__':
    main()
