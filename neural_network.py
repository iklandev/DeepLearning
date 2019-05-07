import logging
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from math import sqrt
from sklearn.metrics import mean_squared_error
from statistics import mean

#Global config
activation_function = 'sigmoid';
optimizer = 'adam';
loss = 'mse';
metrics=['mae'];
model_root_folder = "NeuralNetworks/";
percentage_training = 0.8;
epochs = 1;
save_actual_and_forecast_data = 1;

PROCESS_COLUMNS = ['Date', 'TimeID','DayType', 'FreeParking','TotalJamINRadius_1','TotalJamOUTRadius_1','TotalJamINRadius_2','TotalJamOUTRadius_2','TotalJamINRadius_3','TotalJamOUTRadius_3'];
REINDEX_COLUMNS = ['FreeParking', 'DayType', 'TotalJamINRadius_1','TotalJamOUTRadius_1','TotalJamINRadius_2','TotalJamOUTRadius_2','TotalJamINRadius_3','TotalJamOUTRadius_3'];
ENTRY_PARAM_COLUMNS = ['TotalJamINRadius_1','TotalJamOUTRadius_1','TotalJamINRadius_2','TotalJamOUTRadius_2','TotalJamINRadius_3','TotalJamOUTRadius_3'];

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, filename='neural_network.log');

def parse(x):
    return datetime.strptime(x, '%d/%m/%Y %H:%M')

def convert_time(X):
    return float("{0:.2f}".format(X.hour+get_minute_decimal_value(X.minute))); 

def get_minute_decimal_value (Y):
    if Y==0: return 0.00;
    if Y==4: return 0.07;
    if Y==8: return 0.13;
    if Y==12: return 0.20;
    if Y==16: return 0.27;
    if Y==20: return 0.33;
    if Y==24: return 0.40;
    if Y==28: return 0.47;
    if Y==32: return 0.54;
    if Y==36: return 0.60;    
    if Y==40: return 0.67;
    if Y==44: return 0.73;
    if Y==48: return 0.80;
    if Y==52: return 0.87;
    if Y==56: return 0.93;
    
    return 0;

def save_model (model, name):
    
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_root_folder+name+".json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights(model_root_folder+name+".h5")
    
def load_model (name):
    
    # load json and create model
    json_file = open(model_root_folder+name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_root_folder+name+".h5")
    loaded_model.compile(loss=loss, optimizer=optimizer, metrics=metrics);
    
    return loaded_model

#Read the CSV file and prepare data
#name - name of the CSV file
#nr_time_ids - number of time slots - 15 is 1h
#day_types - 1=>week day; 2=> weekend;
#use_free_parking - use free parking spaces as entry in network
#include_time - use time as entry in network
#include_day_type - use day type as entry in network
def read_and_process_csv (name, nr_time_ids, day_types, use_free_parking, include_time, include_day_type=0):
    
    
    logging.info('read_and_process_csv=>name: {0}; nr_time_ids: {1}; day_types: {2}; use_free_parking: {3}; include_time: {4};'
                 .format(name, nr_time_ids, day_types, use_free_parking, include_time));
    
    input_data = list();
    output_data = list();
    
    optional_entry_columns = [];
    if use_free_parking == 1:
        optional_entry_columns.append('FreeParking'); 
    
    # Read the data from CSV file 
    dataset = pd.read_csv(name, sep=';', usecols=PROCESS_COLUMNS, parse_dates = [['Date', 'TimeID']], index_col=0, date_parser=parse);
    dataset.index.name = 'date'
    dataset = dataset.reindex(columns = REINDEX_COLUMNS);
    
    
    if day_types == 1:
        #Get only Mon to Fri
        dataset = dataset.loc[dataset['DayType'].isin([1, 2, 3, 4, 5])];
    else:
        #Get only Sat and Sun
        dataset = dataset.loc[dataset['DayType'].isin([6,7])];
    
    #Remove day type from entry params
    if include_day_type == 0:
        dataset = dataset.drop(['DayType'], axis=1);
    else:
        optional_entry_columns.append('DayType');
        
    all_entry_data = optional_entry_columns + ENTRY_PARAM_COLUMNS;

    for index, _ in dataset.iterrows():
        forward_date = index.to_datetime();
        back_date = index.to_datetime() - timedelta(minutes=4*nr_time_ids);
        
        in_data = list();
        out_data = list();
        
        if(include_time):
            in_data.append(convert_time(index));
        
        for _ in range(nr_time_ids): 
            if (forward_date in dataset.index) and (back_date in dataset.index):
                
                out_data.append(dataset.loc[forward_date, 'FreeParking']);
                in_data += dataset.loc[back_date, all_entry_data].values.tolist();
                
                forward_date+=timedelta(minutes=4);
                back_date+=timedelta(minutes=4);
                
            else: break;
        else: # only executed if the inner loop did NOT break
            input_data.append(in_data);
            output_data.append(out_data);
            continue;  
    
    train_X = np.asarray(input_data);
    train_Y = np.asarray(output_data);     
        
    n_train_data = int(train_X.shape[0]*percentage_training);
    
    test_X = train_X[n_train_data:, :];
    train_X = train_X[:n_train_data, :];
    
    test_Y = train_Y[n_train_data:, :];
    train_Y = train_Y[:n_train_data, :];  
        
    logging.info('read_and_process_csv: finish');    
    return train_X, train_Y, test_X, test_Y; 

def train (neurons, X_train, Y_train, output_dim):
    
    
    logging.info('train=>neurons: {0};'.format(neurons));
    input_dim = len(X_train[0]);
        
    #Initializing Neural Network
    model = Sequential();
     
    # Adding the input layer and the first hidden layer
    model.add(Dense(neurons, input_dim=input_dim, activation= activation_function))
    model.add(Dense(neurons//4, activation=activation_function))
    model.add(Dense(output_dim, activation=activation_function))

    # Compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(X_train, Y_train, epochs=epochs, batch_size=10, verbose=1, shuffle=False, validation_split = 0.2);
    
    logging.info('train=>finish');    
    return model;

def evaluate_model (model, X_test, Y_test, model_name):
    
    eval = list();
    arr2 = [];
    
    predicted = model.predict(X_test);
    for i in range(len(predicted)):
        # make forecast
        forecast = predicted[i];
        actual = Y_test[i];
        rmse = sqrt(mean_squared_error(actual, forecast));
        arr2.append(actual.tolist()+forecast.tolist());
        eval.append(rmse);
    
    if save_actual_and_forecast_data == 1:
        np.savetxt("{0}_detail_{1}.csv".format(csv_file_name.replace('DataCSV/', '').replace('Garages//', '').replace('.csv', ''), model_name), np.array(arr2), delimiter=";", fmt='%s');  
    
    return mean(eval);
    
    

def train_and_evaluate(array_neurons, file_name, nr_time_ids, day_types, use_free_parking, include_time, include_day_type):
    
    logging.info('train_and_evaluate=>start');
    train_X, train_Y, test_X, test_Y = read_and_process_csv(csv_file_name, nr_time_ids, day_types, use_free_parking, include_time, include_day_type);
    
    for neurons in array_neurons:
        logging.info('train_and_evaluate=>neurons: {0};'.format(neurons));
        model_name = "{0}{1}_{2}".format(use_free_parking, include_time, neurons);
        model = train(neurons, train_X, train_Y, nr_time_ids);
        save_model(model, model_name);
        mean_error = evaluate_model(model, test_X, test_Y, model_name);
        f = open('evaluates.csv','a')  # w : writing mode  /  r : reading mode  /  a  :  appending mode
        f.write('{0};{1}\n'.format(model_name, mean_error));
        f.close();
    
    logging.info('train_and_evaluate=>finish');
    return;

    
#Global config   
csv_file_name = "DataCSV/5_G17.csv"; 
nr_time_ids = 15;
day_types = 1;
include_day_type=0;
array_neurons = [10, 20];
#Config_1
use_free_parking = 0; 
include_time = 0;

train_and_evaluate(array_neurons, csv_file_name, nr_time_ids, day_types, use_free_parking, include_time, include_day_type);
