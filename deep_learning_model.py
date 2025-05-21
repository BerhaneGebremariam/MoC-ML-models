from __future__ import annotations
import pandas as pd
import numpy as np
import typing as tp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,log_loss
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from data_split import data_splits
import tensorflow as tf
from explanatory_analysis import SpDataVisualizer
import os
import random
import warnings
warnings.filterwarnings('ignore')

class NeuralNetwork:
    """
    Similar to the RandomForest class this is also the main class to train tune and evaluate a neural network model.
    This class has nn_model which takes different model settings to train and randomly search for the best model. it finally
    returns the best model and its performance on the validation and test set.
    """
    
    def __init__(self, cv:int=5,n_iter:int=20,n_features:int= 63,
                 early_stop_patience:int=10, random_state : int = 42,
                 data_splits: tp.List[tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]]=data_splits()):
        self.cv = cv
        self.n_iter = n_iter
        self.n_features = n_features
        self.patience = early_stop_patience
        self.random_split = data_splits[0]
        self.random_val_grouped_test = data_splits[1]
        self.grouped_split = data_splits[2]
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.params = {'n_hidden_layers': [1, 2, 3],
                       'learning_rate': [0.1, 0.01, 0.001],
                       'batch_size': [128,256,500,1000],
                       'n_epochs': [25,45,60,100],
                       'dropout': [0,0.1,0.15,0.3,0.5],
                       'n_neurons':[32, 64, 128, 256]}
        
        
    def nn_model(self, split_type='random', metrics_type='discrete', in_sample_validation=False):

        model_name = self.model_name(split_type=split_type, metrics_type=metrics_type, in_sample_validation=in_sample_validation)
        if split_type == 'random':
            X_train_val, X_train, X_val, X_test, y_train_val, y_train, y_val, y_test = self.random_split
        elif split_type == 'grouped':
            X_train_val, X_train, X_val, X_test, y_train_val, y_train, y_val, y_test = self.grouped_split
        else:
            X_train_val, X_train, X_val, X_test, y_train_val, y_train, y_val, y_test = self.random_val_grouped_test
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
        metrics_value_continuous = np.inf
        metrics_value_discrete = 0
        best_model = None
        model_loss = None
        best_params = {'n_hidden_layers': None, 'n_neurons': None, 'learning_rate': None, 'batch_size': None, 'n_epochs': None, 'dropout': None}
            
        for i in range(self.n_iter):

            np.random.seed(i)
            n_hidden_layers = np.random.choice(self.params['n_hidden_layers'])
            n_neurons = np.random.choice(self.params['n_neurons'])
            learning_rate = np.random.choice(self.params['learning_rate'])
            batch_size = np.random.choice(self.params['batch_size'])
            n_epochs = np.random.choice(self.params['n_epochs'])
            dropout = np.random.choice(self.params['dropout'])
            
            model = Sequential(name=model_name)
            model.add(Dense(self.n_features, activation='relu'))
            model.add(Dropout(dropout,seed=i))
            for _ in range(n_hidden_layers):
                model.add(Dense(n_neurons, activation='relu'))
                model.add(Dropout(dropout,seed=i))
            model.add(Dense(8, activation='softmax'))
            
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            if metrics_type == 'discrete':
                early_stop = EarlyStopping(monitor='val_accuracy', mode='max', verbose=0, patience=self.patience)
            else:
                early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=self.patience)
                
            temp_discrete_metrics = []
            temp_continuous_metrics = []
            model_loss_temp = []
            
            if in_sample_validation:
                if split_type == 'grouped':
                    inds = X_train_val['ResponseId']
                    individuals= list(X_train_val['ResponseId'].unique())
                    y_train_val_1 = pd.concat([inds, y_train_val], axis=1)
                    np.random.seed(i)
                    np.random.shuffle(individuals)
                    ind_folds = np.array_split(individuals, self.cv)
                    ind_folds = [list(ind_fold) for ind_fold in ind_folds]
                    for ind_fold in ind_folds:
                        X_to_val = X_train_val[X_train_val['ResponseId'].isin(ind_fold)]
                        X_to_train = X_train_val[~X_train_val['ResponseId'].isin(ind_fold)]
                        y_to_val = y_train_val_1[y_train_val_1['ResponseId'].isin(ind_fold)]
                        y_to_train = y_train_val_1[~y_train_val_1['ResponseId'].isin(ind_fold)]
                        X_to_val = X_to_val.drop(columns=['ResponseId'])
                        X_to_train = X_to_train.drop(columns=['ResponseId'])
                        y_to_val = y_to_val.drop(columns=['ResponseId'])
                        y_to_train = y_to_train.drop(columns=['ResponseId'])
                        X_to_train = X_to_train.values
                        X_to_val = X_to_val.values
                        y_to_train = y_to_train.values
                        y_to_val = y_to_val.values
                        X_to_train_scaled = self.scaler.fit_transform(X_to_train)
                        X_to_val_scaled = self.scaler.transform(X_to_val)
                        model.fit(X_to_train_scaled, y_to_train, validation_data=(X_to_val_scaled, y_to_val), epochs=n_epochs, batch_size=batch_size, callbacks=[early_stop],verbose=0)
                        model_loss_ = pd.DataFrame(model.history.history)
                        val_loss = model_loss_['val_loss'].iloc[-1]
                        val_accuracy = model_loss_['val_accuracy'].iloc[-1]
                        temp_discrete_metrics.append(val_accuracy)
                        temp_continuous_metrics.append(val_loss)
                        model_loss_temp.append(model_loss_)
                else:
                    X_train_val_1 = X_train_val.drop(columns=['ResponseId'])
                    for _ in range(self.cv):
                        X_to_train, X_to_val, y_to_train, y_to_val = train_test_split(X_train_val_1, y_train_val, test_size=0.2, random_state=i, shuffle=True)
                        X_to_train = X_to_train.values
                        X_to_val = X_to_val.values
                        y_to_train = y_to_train.values
                        y_to_val = y_to_val.values
                        X_to_train_scaled = self.scaler.fit_transform(X_to_train)
                        X_to_val_scaled = self.scaler.transform(X_to_val)
                        model.fit(X_to_train_scaled, y_to_train, validation_data=(X_to_val_scaled, y_to_val), epochs=n_epochs, batch_size=batch_size, callbacks=[early_stop],verbose=0)
                        model_loss_ = pd.DataFrame(model.history.history)
                        val_loss = model_loss_['val_loss'].iloc[-1]
                        val_accuracy = model_loss_['val_accuracy'].iloc[-1]
                        temp_discrete_metrics.append(val_accuracy)
                        temp_continuous_metrics.append(val_loss)
                        model_loss_temp.append(model_loss_)
            else:
                if split_type == 'grouped':
                    inds = X_train['ResponseId']
                    individuals= list(X_train['ResponseId'].unique())
                    y_train_1 = pd.concat([inds, y_train], axis=1)

                    np.random.seed(i)
                    np.random.shuffle(individuals)
                    ind_folds = np.array_split(individuals, self.cv)
                    ind_folds = [list(ind_fold) for ind_fold in ind_folds]
                    for ind_fold in ind_folds:
                        X_to_val = X_train[X_train['ResponseId'].isin(ind_fold)]
                        X_to_train = X_train[~X_train['ResponseId'].isin(ind_fold)]
                        y_to_val = y_train_1[y_train_1['ResponseId'].isin(ind_fold)]
                        y_to_train = y_train_1[~y_train_1['ResponseId'].isin(ind_fold)]
                        X_to_val = X_to_val.drop(columns=['ResponseId'])
                        X_to_train = X_to_train.drop(columns=['ResponseId'])
                        y_to_val = y_to_val.drop(columns=['ResponseId'])
                        y_to_train = y_to_train.drop(columns=['ResponseId'])
                        X_to_train = X_to_train.values
                        X_to_val = X_to_val.values
                        y_to_train = y_to_train.values
                        y_to_val = y_to_val.values
                        X_to_train_scaled = self.scaler.fit_transform(X_to_train)
                        X_to_val_scaled = self.scaler.transform(X_to_val)
                        model.fit(X_to_train_scaled, y_to_train, validation_data=(X_to_val_scaled, y_to_val), epochs=n_epochs, batch_size=batch_size, callbacks=[early_stop],verbose=0)
                        model_loss_ = pd.DataFrame(model.history.history)
                        val_loss = model_loss_['val_loss'].iloc[-1]
                        val_accuracy = model_loss_['val_accuracy'].iloc[-1]
                        temp_discrete_metrics.append(val_accuracy)
                        temp_continuous_metrics.append(val_loss)
                        model_loss_temp.append(model_loss_)
                    
                else:
                    X_train_1 = X_train.drop(columns=['ResponseId'])
                    for _ in range(self.cv):
                        X_to_train, X_to_val, y_to_train, y_to_val = train_test_split(X_train_1, y_train, test_size=0.2, random_state=i, shuffle=True)
                        X_to_train = X_to_train.values
                        X_to_val = X_to_val.values
                        y_to_train = y_to_train.values
                        y_to_val = y_to_val.values
                        X_to_train_scaled = self.scaler.fit_transform(X_to_train)
                        X_to_val_scaled = self.scaler.transform(X_to_val)
                        model.fit(X_to_train_scaled, y_to_train, validation_data=(X_to_val_scaled, y_to_val), epochs=n_epochs, batch_size=batch_size, callbacks=[early_stop],verbose=0)
                        model_loss_ = pd.DataFrame(model.history.history)
                        val_loss = model_loss_['val_loss'].iloc[-1]
                        val_accuracy = model_loss_['val_accuracy'].iloc[-1]
                        temp_discrete_metrics.append(val_accuracy)
                        temp_continuous_metrics.append(val_loss)
                        model_loss_temp.append(model_loss_)
                    
            if split_type == 'grouped':

                mean_val_accuracy = np.min(temp_discrete_metrics)
                mean_val_loss = np.max(temp_continuous_metrics)
            else:
                mean_val_accuracy = np.mean(temp_discrete_metrics)
                mean_val_loss = np.mean(temp_continuous_metrics)

            
                    
            if metrics_type == 'discrete':
                if mean_val_accuracy > metrics_value_discrete:
                    metrics_value_discrete = mean_val_accuracy
                    best_model = model
                    best_params['n_hidden_layers'] = n_hidden_layers
                    best_params['n_neurons'] = n_neurons
                    best_params['learning_rate'] = learning_rate
                    best_params['batch_size'] = batch_size
                    best_params['n_epochs'] = n_epochs
                    best_params['dropout'] = dropout
                    model_loss = model_loss_temp
        
            else:
                if mean_val_loss < metrics_value_continuous:
                    metrics_value_continuous = mean_val_loss
                    best_model = model
                    best_params['n_hidden_layers'] = n_hidden_layers
                    best_params['n_neurons'] = n_neurons
                    best_params['learning_rate'] = learning_rate
                    best_params['batch_size'] = batch_size
                    best_params['n_epochs'] = n_epochs
                    best_params['dropout'] = dropout
                    model_loss = model_loss_temp
            
            temp_discrete_metrics = []
            temp_continuous_metrics = []
            model_loss_temp = []
                    

        
        X_val = X_val.values
        y_val = y_val.values
        X_test = X_test.values
        y_test = y_test.values
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        if metrics_type == 'discrete':
            y_pred_val = np.argmax(best_model.predict(X_val_scaled), axis=-1)
            accuracy_val = accuracy_score(y_val, y_pred_val)

            y_pred_test = np.argmax(best_model.predict(X_test_scaled), axis=-1)
            accuracy_test = accuracy_score(y_test, y_pred_test)

            
            performance_test = classification_report(y_test, y_pred_test,output_dict=True)

            performance_val = classification_report(y_val, y_pred_val,output_dict=True)
            
            conf_matrix_test = confusion_matrix(y_test, y_pred_test)
            conf_matrix_val = confusion_matrix(y_val, y_pred_val)
            
            to_return = {'model': best_model, 'best_params': best_params, 'accuracy_val': accuracy_val,
                         'accuracy_test': accuracy_test, 'performance_test': performance_test,
                         'performance_val': performance_val, 'conf_matrix_test': conf_matrix_test,
                         'conf_matrix_val': conf_matrix_val,'best_score': metrics_value_discrete,
                         'model_loss': model_loss}
        else:

            y_pred_val = best_model.predict(X_val_scaled)

            y_pred_test = best_model.predict(X_test_scaled)

            y_loss_test= log_loss(y_test, y_pred_test)

            y_loss_val= log_loss(y_val, y_pred_val)

            
            to_return = {'model': best_model, 'best_params': best_params,
                         'best_score': metrics_value_continuous,
                         'y_pred_val': y_pred_val,
                         'y_pred_test': y_pred_test,
                        'loss_val': y_loss_val,
                         'loss_test': y_loss_test,
                         'model_loss': model_loss}

        index = None

        for i in range(len(model_loss)):

            model_ = model_loss[i]
            
            min_diff = []
            if split_type == 'grouped':
                if metrics_type == 'discrete':
                    min_diff.append(model_['val_accuracy'].iloc[-1])

                else:
                    min_diff.append(model_['val_loss'].iloc[-1])

            else:
                if metrics_type == 'discrete':
                    min_diff.append(np.abs(model_['val_accuracy'].iloc[-1] - model_['val_accuracy'].iloc[-2]))

                else:
                    min_diff.append(np.abs(model_['val_loss'].iloc[-1] - model_['val_loss'].iloc[-2]))


            if split_type == 'grouped':
                if metrics_type == 'discrete':
                    index = min_diff.index(min(min_diff))
                else:
                    index = min_diff.index(max(min_diff))
            else:
                index = min_diff.index(min(min_diff))


        model_ = model_loss[index]
        SpDataVisualizer().general_styling()
        plt.close()
        if metrics_type == 'discrete':
            plt.plot(model_['accuracy'], label='accuracy',linewidth=2)
            plt.plot(model_['val_accuracy'], label='val_accuracy',linewidth=2)
            plt.title(f'{model_name} Model accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig(os.path.join('plots',f'{model_name}_accuracy.png'))
            plt.close()
        else:
            plt.plot(model_['loss'], label='loss',linewidth=2)
            plt.plot(model_['val_loss'], label='val_loss',linewidth=2)
            plt.title(f'{model_name} Model loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join('plots',f'{model_name}_loss.png'))
            plt.close()

        best_model.save(os.path.join('models',f'{model_name}_{self.cv}_{self.n_iter}.h5'))
        return to_return

    def model_name(self, split_type:str='random',metrics_type:str='discrete',in_sample_validation:bool= False):
        if split_type == 'random':
            if metrics_type == 'discrete':
                if not in_sample_validation:
                    return 'M1ANN'
                else:
                    return 'M7ANN'
            else:
                if not in_sample_validation:
                    return 'M2ANN'
                else:
                    return 'M8ANN'
        elif split_type == 'grouped':
            if metrics_type == 'discrete':
                if not in_sample_validation:
                    return 'M5ANN'
                else:
                    return 'M11ANN'
            else:
                if not in_sample_validation:
                    return 'M6ANN'
                else:
                    return 'M12ANN'
        else:
            if metrics_type == 'discrete':
                if not in_sample_validation:
                    return 'M3ANN'
                else:
                    return 'M9ANN'
            else:
                if not in_sample_validation:
                    return 'M4ANN'
                else:
                    return 'M10ANN'

def main():
    nn = NeuralNetwork(cv=2,n_iter=2,n_features= 63,early_stop_patience=5,random_state = 42)
    nn.nn_model()

if __name__ == '__main__':

    main()
