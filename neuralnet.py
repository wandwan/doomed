import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras import layers, callbacks
from keras_tuner.tuners import RandomSearch

def balanced_log_loss(y_true, y_pred):
    N_0 = np.sum(1 - y_true)
    N_1 = np.sum(y_true)
    w_0 = 1 / N_0
    w_1 = 1 / N_1
    p_1 = np.clip(y_pred, 1e-15, 1 - 1e-15)
    p_0 = 1 - p_1
    log_loss_0 = -np.sum((1 - y_true) * np.log(p_0))
    log_loss_1 = -np.sum(y_true * np.log(p_1))
    balanced_log_loss = 2*(w_0 * log_loss_0 + w_1 * log_loss_1) / (w_0 + w_1)
    return balanced_log_loss/(N_0+N_1)

def balanced_logarithmic_loss_new(y_true, y_pred):
    N_1 = np.sum(y_true == 1, axis=0)
    N_0 = np.sum(y_true == 0, axis=0)
    y_pred = np.maximum(np.minimum(y_pred, 1 - 1e-15), 1e-15)
    loss_numerator = - (1/N_0) * np.sum((1 - y_true) * np.log(1-y_pred)) - (1/N_1) * np.sum(y_true * np.log(y_pred))
    return loss_numerator / 2

def balanced_log_loss_between(y_true, y_pred):
    return (balanced_log_loss(y_true, y_pred)+balanced_logarithmic_loss_new(y_true, y_pred))/2

class MyModel:
    def __init__(self, train_size, validation_size, early_stopping_monitor, early_stopping_patience, verbose, max_trials, executions_per_trial, max_tunning_epochs):
        self.train_size = train_size
        self.validation_size = validation_size
        self.early_stopping_monitor = early_stopping_monitor
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose
        self.max_trials = max_trials
        self.executions_per_trial = executions_per_trial
        self.max_tunning_epochs = max_tunning_epochs
        self.X_train = None
        self.y_train = None
        self.X_validation = None
        self.y_validation = None
        self.X_test = None
        self.y_test = None
        self.experimentsDf = pd.DataFrame(columns=['Iteration', 'loss', 'accuracy', 'log_loss1',  'log_loss2', 'log_loss3', 'optimizer', 'learning_rate', 'hp'])
        self.best_model = None
        self.best_hp = None
        self.best_log_loss = np.inf

    def create(self):
        def read_data(self):
            raw_data = pd.read_csv('data.csv')
            test_data = pd.read_csv('test.csv')
            test_data['EJ'] = test_data['EJ'].replace({'A': 0, 'B': 1})
            X = raw_data.drop('Class', axis=1)
            y = raw_data['Class']
            X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=1-self.train_size, random_state=42, stratify=y)
            X_validation, X_test, y_validation, y_test = train_test_split(X_other, y_other, test_size=self.validation_size, random_state=42, stratify=y_other)
            scaler = MinMaxScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_validation = scaler.transform(X_validation)
            X_test = scaler.transform(X_test)
            self.X_train, self.y_train, self.X_validation, self.y_validation, self.X_test, self.y_test = X_train, y_train, X_validation, y_validation, X_test, y_test

        def build_model(self, hp):
            model = keras.Sequential()
            activation  = hp.Choice('activation', values=['relu', 'sigmoid', 'tanh'])
            hp_optimizer = hp.Choice('hp_optimizer', values=['Adam'])
            model.add(layers.Dense(hp.Int("layer_1", min_value=290, max_value=290, step=40), activation=activation))
            model.add(layers.Dropout(hp.Float("dropuout_rate_1", min_value=0, max_value=0.3,step=0.1)))
            model.add(layers.Dense(hp.Int("layer_2", min_value=10, max_value=300, step=40), activation=activation))
            model.add(layers.Dropout(hp.Float("dropuout_rate_2", min_value=0, max_value=0,step=0.1)))
            model.add(layers.Dense(1,  activation='sigmoid'))    
            model.compile(loss=keras.losses.BinaryCrossentropy(),
                            optimizer = hp_optimizer,
                            metrics=['accuracy'])
            return model

        read_data(self)
        early_stoping_callback = callbacks.EarlyStopping(
            monitor=self.early_stopping_monitor, 
            patience=self.early_stopping_patience, 
            verbose=self.verbose, 
            restore_best_weights=True
        )
        tuner = RandomSearch(
            hypermodel=build_model,
            objective="val_accuracy",
            max_trials=self.max_trials,
            executions_per_trial=self.executions_per_trial,
            overwrite=True,
            directory="my_dir",
            project_name="ICR")
        tuner.search(x=self.X_train, y=self.y_train, epochs=self.max_tunning_epochs, validation_data=(self.X_validation, self.y_validation), callbacks=[early_stoping_callback])
        trials_count = len(tuner.get_best_models(99999))
        top_models = tuner.get_best_models(trials_count)
        best_hps = tuner.get_best_hyperparameters(trials_count)
        for n in range(trials_count):
            n_model = top_models[n]
            n_hp = best_hps[n]
            loss, accuracy = n_model.evaluate(self.X_test, self.y_test)
            tmp_pred = n_model.predict(self.X_test)
            log_loss1 = balanced_log_loss(self.y_test, np.squeeze(tmp_pred, axis=1))
            log_loss2 = balanced_logarithmic_loss_new(self.y_test, np.squeeze(tmp_pred, axis=1))
            log_loss3 = balanced_log_loss_between(self.y_test, np.squeeze(tmp_pred, axis=1))
            new_row = {
                'Iteration': n,
                'loss' : loss,
                'accuracy' : accuracy,
                'log_loss1' : log_loss1,
                'log_loss2' : log_loss2,
                'log_loss3' : log_loss3,
                'optimizer' : n_model.optimizer.name,
                'learning_rate' : n_model.optimizer.learning_rate.read_value().numpy(), 
                'hp' : n_hp.values
            }
            self.experimentsDf = pd.concat([self.experimentsDf, pd.DataFrame.from_records([new_row])])
            if(log_loss3<self.best_log_loss):
                self.best_model = n_model
                self.best_hp = n_hp
                self.best_log_loss = log_loss3

    def inference(self, X):
        if self.best_model is None:
            raise ValueError("Model has not been created yet. Please call the create method first.")
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        return self.best_model.predict(X)