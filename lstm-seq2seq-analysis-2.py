from datetime import datetime
from keras.layers import Dense, LSTM
from keras.models import Sequential, model_from_json
from math import sqrt
from matplotlib import pyplot
from pandas import concat
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from numpy import array

 
#ALL_COLUMNS = ['Date','Time','TimeID','DayType','LotID','CityID','LotName','FreeParking','TotalJamINRadius_1','TotalJamOUTRadius_1','TotalJamINRadius_2','TotalJamOUTRadius_2','TotalJamINRadius_3','TotalJamOUTRadius_3','AvgINForRadius_1','AvgOUTForRadius_1','AvgINForRadius_2','AvgOUTForRadius_2','AvgINForRadius_3','AvgOUTForRadius_3'];
PROCESS_COLUMNS = ['Date', 'TimeID','DayType', 'FreeParking','TotalJamINRadius_1','TotalJamOUTRadius_1','TotalJamINRadius_2','TotalJamOUTRadius_2','TotalJamINRadius_3','TotalJamOUTRadius_3'];
REINDEX_COLUMNS = ['FreeParking', 'DayType', 'TotalJamINRadius_1','TotalJamOUTRadius_1','TotalJamINRadius_2','TotalJamOUTRadius_2','TotalJamINRadius_3','TotalJamOUTRadius_3'];


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, filename='app.log');
#Number of memory units in first LSTM layer
neorons = 1; 

#Don't use number of free spaces as entry parameter
use_free_parking_var = 111; 

#0 - Group by all days; 1 - Only for week days; 2 - Only for weekends 
#group_day_type = 222;

#Include time in entry variables
include_time = 333;

#number of previous time ID's. 15 Time ID's is equal to 1 hour 
nr_time_ids = 15;
    
# fix random seed for reproducibility
np.random.seed(7)

#config
csv_file_name = "DataCSV/5_G17.csv";
model_name = "Seattle_LSTM";
optimizer = "adam";
loss = "mse";
metrics = ['mae'];
epochs = 50;
batch_size = 7;
percentage_training = 0.7;
model_root_folder = "NeuralNetworks/";

 
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
def read_and_process_csv (name):
     
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
       
    
    #Get only Sat and Sun
    #if group_day_type == 2:
     #   dataset = dataset.loc[dataset['DayType'].isin([6,7])];
      #  dataset = dataset.drop(['DayType'], axis=1)
        
    #dataset.to_csv("raw.csv", sep=';');
    return dataset;
 
def set_len_to_be_devide_with_batch_size(len, batch_size_t):
    while len % batch_size_t != 0:
        len -= 1
    return len; 
#Prepare the data for LSTM networks
def get_data (name):
      
    scaler = None;
    #Get the CSV file
    dataset = read_and_process_csv(name);
    values = dataset.values
     
    # ensure all data is float
    values = values.astype('float32')
    
    features_temp = 7;    
    if include_time == 1:
        features_temp = features_temp + 1;
   
    tem = np.array(values, copy=True);
    tem = tem[:, 0]

    reframed = series_to_supervised(values, nr_time_ids, nr_time_ids)

    reframed.to_csv("origin.csv", sep=';');
    values = reframed.values
    
    #delete free parking from entry variable
    if use_free_parking_var == 0:
        indexs_to_remove = [];
        for i in range (0, nr_time_ids*features_temp, features_temp):
            indexs_to_remove.append(i);
        features_temp = features_temp-1;    
        values = np.delete(values, indexs_to_remove, axis=1)
        
    # split into train and test sets
    n_train_hours = int(values.shape[0]*percentage_training);
    n_train_hours = set_len_to_be_devide_with_batch_size(n_train_hours, batch_size);
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
     
    
    # split into input and outputs
    n_obs = nr_time_ids * features_temp;
    n_obs_f = values.shape[1] - n_obs;
    
    result_features_temp = features_temp;
    if use_free_parking_var == 0:
        result_features_temp = result_features_temp+1;
    
    train_X, train_y = train[:, :n_obs], train[:, -n_obs_f:]
    train_y = np.array([x[0::result_features_temp] for x in train_y])
    
    test_X, test_y = test[:, :n_obs], test[:, -n_obs_f:]
    test_y = np.array([x[0::result_features_temp] for x in test_y])
        
    
    #Separate on half and use the first half for validating 
    #each epoch, and second half for validating network
    val_len = test_X.shape[0]//2;
    
    #set val len to be divadible with  batch size
    val_len = set_len_to_be_devide_with_batch_size(val_len, batch_size)
    
        
    val_X = test_X[:val_len,:];
    test_X = test_X[val_len:, :];

    val_y = test_y[:val_len, :];
    test_y = test_y[val_len:, :];
    
    #np.savetxt("train_X.csv", train_X, delimiter=";");
 
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], nr_time_ids, features_temp))
    test_X = test_X.reshape((test_X.shape[0], nr_time_ids, features_temp))
    val_X = val_X.reshape((val_X.shape[0], nr_time_ids, features_temp))
    
    return train_X, train_y, test_X, test_y, scaler, val_X, val_y;
 

def save_model (model, name):
    
    name = "{0}{1}_{2}".format(use_free_parking_var, include_time, neorons);
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_root_folder+name+".json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights(model_root_folder+name+".h5")
    
def load_model (name):
    
    name = "{0}{1}_{2}".format(use_free_parking_var, include_time, neorons);
    # load json and create model
    json_file = open(model_root_folder+name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_root_folder+name+".h5")
    loaded_model.compile(loss=loss, optimizer=optimizer, metrics=metrics);
    
    return loaded_model


# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
    
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
 
def evaluate_model():
    
    _, _, test_X, test_y, scaler, _, _ = get_data(csv_file_name);
    model = load_model(model_name);
    
        
    forecasts = list()
    for i in range(len(test_X)):
        # make forecast
        if(i+6<len(test_X)):
            forecast = forecast_lstm(model, np.concatenate((test_X[i], test_X[i+1], test_X[i+2], test_X[i+3], test_X[i+4], test_X[i+5], test_X[i+6])) , batch_size)
            # store the forecast
            forecasts.append(forecast);
   
    file_basic_name = "{0}{1}_{2}".format(use_free_parking_var, include_time, neorons);
    #np.savetxt("p{0}.csv".format(file_basic_name), forecasts, delimiter=";");

    eval = list()
    for i in range(len(forecasts)):
        actual = test_y[i]
        predicted = forecasts[i]
        rmse = sqrt(mean_squared_error(actual, predicted))
        eval.append(rmse);
        
    np.savetxt("e{0}.csv".format(file_basic_name), eval, delimiter=";");
    #pyplot.plot(eval, color='blue');
    #pyplot.show();


    print("Model was evaluated");
     
def train_lstm():
     
    train_X, train_y, _, _, _, val_X, val_y = get_data(csv_file_name);

    # design network
    model = Sequential()
    model.add(LSTM(neorons, batch_input_shape=(batch_size, train_X.shape[1], train_X.shape[2]), stateful=True))
    model.add(Dense(50))
    model.add(Dense(train_y.shape[1]))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=False, validation_data=(val_X, val_y))
        
    save_model(model, model_name);
    logging.info('Training the model has finished');
    print("Training the model has finished");

    
print ("----START----");
logging.info('Start');
array_neurons = [15, 20, 24, 30, 38, 45, 52, 60, 68, 75];

# Config 1
use_free_parking_var = 1; 
include_time = 0;
for i in array_neurons:
    neorons = i;
    logging.info("{0}{1}: {2}".format(use_free_parking_var, include_time, neorons));
    print("{0}{1}: {2}".format(use_free_parking_var, include_time, neorons));
    train_lstm();
    evaluate_model();

   
#Config 2
use_free_parking_var = 0; 
include_time = 0;
for i in array_neurons:
    neorons = i;
    logging.info("{0}{1}: {2}".format(use_free_parking_var, include_time, neorons));
    print("{0}{1}: {2}".format(use_free_parking_var, include_time, neorons));
    train_lstm();
    evaluate_model();

    
#Config 3
use_free_parking_var = 1; 
include_time = 0;
for i in array_neurons:
    neorons = i;
    logging.info("{0}{1}: {2}".format(use_free_parking_var, include_time, neorons));
    print("{0}{1}: {2}".format(use_free_parking_var, include_time, neorons));
    train_lstm();
    evaluate_model();
    
#Config 4
use_free_parking_var = 1; 
include_time = 1;
for i in array_neurons:
    neorons = i;
    logging.info("{0}{1}: {2}".format(use_free_parking_var, include_time, neorons));
    print("{0}{1}: {2}".format(use_free_parking_var, include_time, neorons));
    train_lstm();
    evaluate_model();

logging.info('End');
print ("----END----");


