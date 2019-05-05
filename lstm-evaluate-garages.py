from datetime import datetime
from keras.layers import Dense, LSTM
from keras.models import Sequential, model_from_json
from math import sqrt
from matplotlib import pyplot
from pandas import concat
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from numpy import array
from statistics import mean
import os

nr_time_ids = 15;
np.random.seed(7);
optimizer = "adam";
loss = "mse";
metrics = ['mae'];
batch_size = 7;
model_root_folder = "NeuralNetworks/";
#use_free_parking_var, include_time

PROCESS_COLUMNS = ['Date', 'TimeID','DayType', 'FreeParking','TotalJamINRadius_1','TotalJamOUTRadius_1','TotalJamINRadius_2','TotalJamOUTRadius_2','TotalJamINRadius_3','TotalJamOUTRadius_3'];
REINDEX_COLUMNS = ['FreeParking', 'DayType', 'TotalJamINRadius_1','TotalJamOUTRadius_1','TotalJamINRadius_2','TotalJamOUTRadius_2','TotalJamINRadius_3','TotalJamOUTRadius_3'];


def get_all_garages():
    onlyfiles = [];
    for root, dirs, files in os.walk("Garages/"):  
        for filename in files:
            onlyfiles.append(root+"/"+filename); 
    return onlyfiles;

def parse(x):
    return datetime.strptime(x, '%d/%m/%Y %H:%M')

def convert_time(X):
    return X.hour+float("{0:.2f}".format(X.minute/60)); 

# Convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
 
#Read the csv file
def read_and_process_csv (name, include_time):
     
    # Read the data from CSV file 
    dataset = pd.read_csv(name, sep=';', usecols=PROCESS_COLUMNS, parse_dates = [['Date', 'TimeID']], index_col=0, date_parser=parse);
    dataset.index.name = 'date'
    dataset = dataset.reindex(columns = REINDEX_COLUMNS);
    
    if include_time == 1:
        dataset.reset_index(inplace=True);
        dataset['Time'] = dataset['date'].apply(convert_time)
        dataset.set_index('date', drop=True, inplace=True)

    
    #Get only Mon to Fri
    dataset = dataset.loc[dataset['DayType'].isin([1, 2, 3, 4, 5])];
    dataset = dataset.drop(['DayType'], axis=1)
        
    #dataset.to_csv("raw.csv", sep=';');
    return dataset;
 
#Prepare the data for LSTM networks
def get_data (name, use_free_parking_var, include_time):
      

    #Get the CSV file
    dataset = read_and_process_csv(name, include_time);
    values = dataset.values
     
    # ensure all data is float
    values = values.astype('float32')
    
    features_temp = 7;    
    if include_time == 1:
        features_temp = features_temp + 1;
   
    tem = np.array(values, copy=True);
    tem = tem[:, 0]

    reframed = series_to_supervised(values, nr_time_ids, nr_time_ids)

    #reframed.to_csv("origin.csv", sep=';');
    values = reframed.values
    
    #delete free parking from entry variable
    if use_free_parking_var == 0:
        indexs_to_remove = [];
        for i in range (0, nr_time_ids*features_temp, features_temp):
            indexs_to_remove.append(i);
        features_temp = features_temp-1;    
        values = np.delete(values, indexs_to_remove, axis=1)
        
    # split into train and test sets
    n_train_hours = int(values.shape[0]);
    train = values[:n_train_hours, :]
    
    # split into input and outputs
    n_obs = nr_time_ids * features_temp;
    n_obs_f = values.shape[1] - n_obs;
    
    result_features_temp = features_temp;
    if use_free_parking_var == 0:
        result_features_temp = result_features_temp+1;
    
    train_X, train_y = train[:, :n_obs], train[:, -n_obs_f:]
    train_y = np.array([x[0::result_features_temp] for x in train_y])

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], nr_time_ids, features_temp))    
    return train_X, train_y;

def load_model (name):
    
    # load json and create model
    json_file = open(model_root_folder+name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_root_folder+name+".h5")
    loaded_model.compile(loss=loss, optimizer=optimizer, metrics=metrics);
    
    return loaded_model;

# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch, use_free_parking_var, include_time):
    
    features_temp = 7;    
    if include_time == 1:
        features_temp = features_temp + 1;
        
    if use_free_parking_var == 0:
        features_temp = features_temp-1;    
    
    # reshape input pattern to [samples, timesteps, features]
    X = X.reshape((n_batch, nr_time_ids, features_temp))
    # make forecast
    forecast = model.predict(X, batch_size=n_batch)
    # convert to array
    return [x for x in forecast[0, :]]
 
def evaluate_model(csv_file_name, model_name, use_free_parking_var, include_time):
    
    test_X, test_y = get_data(csv_file_name, use_free_parking_var, include_time);
    model = load_model(model_name);
        
    forecasts = list()
    for i in range(len(test_X)):
        # make forecast
        if(i+6<len(test_X)):
            forecast = forecast_lstm(model, np.concatenate((test_X[i], test_X[i+1], test_X[i+2], test_X[i+3], test_X[i+4], test_X[i+5], test_X[i+6])), batch_size, use_free_parking_var, include_time)
            # store the forecast
            forecasts.append(forecast);
    
    eval = list()
    arr2 = [];
    for i in range(len(forecasts)):
        actual = test_y[i]
        predicted = forecasts[i]
        rmse = sqrt(mean_squared_error(actual, predicted))
        arr2.append(actual.tolist()+predicted);
        eval.append(rmse);
    
    np.savetxt("{0}_detail_{1}{2}.csv".format(csv_file_name.replace('Garages//', '').replace('.csv', ''), use_free_parking_var, include_time), np.array(arr2), delimiter=";", fmt='%s');    
    return mean(eval);  

def create_model_name (use_free_parking_var, include_time):        
    return  "{0}{1}".format(use_free_parking_var, include_time);

def create_log_message (garage, use_free_parking_var, include_time):
    return  "{0}: Evaluate garage {1} for {2}{3}\n".format(datetime.now(), garage, use_free_parking_var, include_time);

def evaluate_all ():
    garages = get_all_garages();
    
    arr = [];
    logger_my = [];
    for garage in garages:
        
        log_messages = list();
        stat_garage = list();
        stat_garage.append(garage);
        # Config 1
        use_free_parking_var = 0; 
        include_time = 0;
        log_messages.append(create_log_message (garage, use_free_parking_var, include_time));
        print(create_log_message (garage, use_free_parking_var, include_time));
        stat_garage.append(evaluate_model(garage, create_model_name(use_free_parking_var, include_time), use_free_parking_var, include_time));  
   
        #Config 2
        use_free_parking_var = 0; 
        include_time = 1;
        log_messages.append(create_log_message (garage, use_free_parking_var, include_time));
        print(create_log_message (garage, use_free_parking_var, include_time));
        stat_garage.append(evaluate_model(garage, create_model_name(use_free_parking_var, include_time), use_free_parking_var, include_time));

        #Config 3
        use_free_parking_var = 1; 
        include_time = 0;
        log_messages.append(create_log_message (garage, use_free_parking_var, include_time));
        print(create_log_message (garage, use_free_parking_var, include_time));
        stat_garage.append(evaluate_model(garage, create_model_name(use_free_parking_var, include_time), use_free_parking_var, include_time));
    
        #Config 4
        use_free_parking_var = 1; 
        include_time = 1
        log_messages.append(create_log_message (garage, use_free_parking_var, include_time));
        print(create_log_message (garage, use_free_parking_var, include_time));
        stat_garage.append(evaluate_model(garage, create_model_name(use_free_parking_var, include_time), use_free_parking_var, include_time));
        
        arr.append(stat_garage)
        logger_my.append(log_messages);
        np_arr = np.array(arr);
        np_arr_log = np.array(logger_my);
        np.savetxt("logs.txt", np_arr_log, delimiter=";", fmt='%s');
        np.savetxt("all_garages_eval.csv", np_arr, delimiter=";", fmt='%s');
    
   
    
evaluate_all ();