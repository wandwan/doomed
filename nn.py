import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import keras_tuner
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers

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

def read_data():
    raw_data   = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/train.csv')
    greek_data = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/greeks.csv')
    test_data  = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/test.csv')
    raw_test_data  = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/test.csv')
    sample     = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/sample_submission.csv')
    raw_data = raw_data.fillna(raw_data.median())
    raw_data = raw_data.ffill()
    test_data = test_data.fillna(test_data.median())
    test_data = test_data.ffill()
    raw_data.drop('Id', axis=1, inplace=True)
    test_data.drop('Id', axis=1, inplace=True)
    raw_data['EJ'] = raw_data['EJ'].replace({'A': 0, 'B': 1})
    test_data['EJ'] = test_data['EJ'].replace({'A': 0, 'B': 1})
    X = raw_data.drop('Class', axis=1)
    y = raw_data['Class']
    X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=1-train_size, random_state=42, stratify=y)
    X_validation, X_test, y_validation, y_test = train_test_split(X_other, y_other, test_size=validation_size, random_state=42, stratify=y_other)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_validation = scaler.transform(X_validation)
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_validation, y_validation, X_test, y_test

def experiment_parameters():
    quick_run = False
    train_size = 0.7
    validation_size = 0.5
    early_stopping_patience = 20
    early_stopping_monitor = 'val_loss'
    executions_per_trial = 2
    if quick_run:
        verbose = 0
        max_tunning_epochs = 100
        max_trials = 2
    else:
        verbose = 0
        max_tunning_epochs = 2000
        max_trials = 100
    return verbose, max_tunning_epochs, max_trials, early_stopping_patience, early_stopping_monitor, executions_per_trial

def read_data():
    raw_data = pd.read_csv('data.csv')
    test_data = pd.read_csv('test.csv')
    test_data['EJ'] = test_data['EJ'].replace({'A': 0, 'B': 1})
    X = raw_data.drop('Class', axis=1)
    y = raw_data['Class']
    X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=1-train_size, random_state=42, stratify=y)
    X_validation, X_test, y_validation, y_test = train_test_split(X_other, y_other, test_size=validation_size, random_state=42, stratify=y_other)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_validation = scaler.transform(X_validation)
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_validation, y_validation, X_test, y_test

def experiment_parameters():
    quick_run = False
    train_size = 0.7
    validation_size = 0.5
    early_stopping_patience = 20
    early_stopping_monitor = 'val_loss'
    executions_per_trial = 2
    if quick_run:
        verbose = 0
        max_tunning_epochs = 100
        max_trials = 2
    else:
        verbose = 0
        max_tunning_epochs = 2000
        max_trials = 100
    return verbose, max_tunning_epochs, max_trials, early_stopping_patience, early_stopping_monitor, executions_per_trial

def main():
    X_train, y_train, X_validation, y_validation, X_test, y_test = read_data()
    verbose, max_tunning_epochs, max_trials, early_stopping_patience, early_stopping_monitor, executions_per_trial = experiment_parameters()
    print("X length {}".format(len(X)))
    print("Train length {}".format(len(X_train)))
    print("Validation length {}".format(len(X_validation)))
    print("Test length {}".format(len(X_test)))

def build_model(hp):
    model = tf.keras.Sequential()
    
    activation  = hp.Choice('activation', values=['relu', 'sigmoid', 'tanh'])
    hp_optimizer = hp.Choice('hp_optimizer', values=['Adam'])
    
    model.add(tf.keras.layers.Dense(hp.Int("layer_1", min_value=290, max_value=290, step=40), activation=activation))
    model.add(tf.keras.layers.Dropout(hp.Float("dropuout_rate_1", min_value=0, max_value=0.3,step=0.1)))
    
    model.add(tf.keras.layers.Dense(hp.Int("layer_2", min_value=10, max_value=300, step=40), activation=activation))
    model.add(tf.keras.layers.Dropout(hp.Float("dropuout_rate_2", min_value=0, max_value=0,step=0.1)))
    
    model.add(tf.keras.layers.Dense(1,  activation='sigmoid'))    

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer = hp_optimizer,
                    metrics=['accuracy'])
    
    return model

early_stoping_callback = tf.keras.callbacks.EarlyStopping(
    monitor=early_stopping_monitor, 
    patience=early_stopping_patience, 
    verbose=verbose, 
    restore_best_weights=True
)

tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=max_trials,
    executions_per_trial=executions_per_trial,
    overwrite=True,
    directory="my_dir",
    project_name="ICR")

tuner.search(x=X_train, y=y_train, epochs=max_tunning_epochs, validation_data=(X_validation, y_validation), callbacks=[early_stoping_callback])
trials_count = len(tuner.get_best_models(99999))
top_models = tuner.get_best_models(trials_count)
best_hps = tuner.get_best_hyperparameters(trials_count)
best_log_loss = 9999
experimentsDf = df = pd.DataFrame(columns=['Iteration', 'loss', 'accuracy', 'log_loss1',  'log_loss2', 'log_loss3', 'optimizer', 'learning_rate', 'hp'])
for n in range(trials_count):
    n_model = top_models[n]
    n_hp = best_hps[n]
    loss, accuracy = n_model.evaluate(X_test, y_test)
    tmp_pred = n_model.predict(X_test)
    log_loss1 = balanced_log_loss(y_test, np.squeeze(tmp_pred, axis=1))
    log_loss2 = balanced_logarithmic_loss_new(y_test, np.squeeze(tmp_pred, axis=1))
    log_loss3 = balanced_log_loss_between(y_test, np.squeeze(tmp_pred, axis=1))

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

    experimentsDf = pd.concat([experimentsDf, pd.DataFrame.from_records([new_row])])
   

    if(log_loss3<best_log_loss):
        best_model = n_model
        best_hp = n_hp
        best_log_loss = log_loss3
        print(f"""
        ------------------------------ Model summary ------------------------------
        balanced_log_loss1: {log_loss1} 
        balanced_log_loss2: {log_loss2}
        balanced_log_loss3: {log_loss3}
        loss: {loss} 
        accuracy: {accuracy} 
        hp : {best_hp.values}
        Model optimizer: {best_model.optimizer.name}
        Model optimizer learning_rate: {best_model.optimizer.learning_rate.read_value().numpy()}
        Model summary: {best_model.summary()}
        ------------------------------ Model summary end ------------------------------
        """)