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
import os
 
ALL_COLUMNS = ['Date','Time','TimeID','DayType','LotID','CityID','LotName','FreeParking','TotalJamINRadius_1','TotalJamOUTRadius_1','TotalJamINRadius_2','TotalJamOUTRadius_2','TotalJamINRadius_3','TotalJamOUTRadius_3','AvgINForRadius_1','AvgOUTForRadius_1','AvgINForRadius_2','AvgOUTForRadius_2','AvgINForRadius_3','AvgOUTForRadius_3'];
PROCESS_COLUMNS = ['Date', 'TimeID','DayType', 'FreeParking','TotalJamINRadius_1','TotalJamOUTRadius_1','TotalJamINRadius_2','TotalJamOUTRadius_2','TotalJamINRadius_3','TotalJamOUTRadius_3'];
REINDEX_COLUMNS = ['FreeParking', 'DayType', 'TotalJamINRadius_1','TotalJamOUTRadius_1','TotalJamINRadius_2','TotalJamOUTRadius_2','TotalJamINRadius_3','TotalJamOUTRadius_3'];

#Don't use number of free spaces as entry parameter
#dont_use_free_parking_var = 1; 

#0 - Group by all days; 1 - Only for week days; 2 - Only for weekends 
#group_day_type = 1;

#Include time in entry variables
#include_time = 1;

#number of previous time ID's. 15 Time ID's is equal to 1 hour 
nr_time_ids = 15;
    
# fix random seed for reproducibility
np.random.seed(7)
use_scaler = 1;

#config
#csv_file_name = "DataCSV/8_8353.csv";
#model_name = "Seattle_LSTM568";
optimizer = "adam";
loss = "mse";
metrics = ['mae'];
epochs = 50;
batch_size = 1;
model_root_folder = "NeuralNetworks/";
only_evaluate = 1;

 
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
def read_and_process_csv (name, dont_use_free_parking_var, group_day_type, include_time):
     
    # Read the data from CSV file 
    dataset = pd.read_csv(name, sep=';', usecols=PROCESS_COLUMNS, parse_dates = [['Date', 'TimeID']], index_col=0, date_parser=parse);
    dataset.index.name = 'date'
    dataset = dataset.reindex(columns = REINDEX_COLUMNS);
    
    if include_time == 1:
        dataset.reset_index(inplace=True);
        dataset['Time'] = dataset['date'].apply(convert_time)
        dataset.set_index('date', drop=True, inplace=True)

    
    #Get only Mon to Fri
    if group_day_type == 1:
        dataset = dataset.loc[dataset['DayType'].isin([1, 2, 3, 4, 5])];
        dataset = dataset.drop(['DayType'], axis=1)
       
    
    #Get only Sat and Sun
    if group_day_type == 2:
        dataset = dataset.loc[dataset['DayType'].isin([6,7])];
        dataset = dataset.drop(['DayType'], axis=1)
        
    #dataset.to_csv("raw.csv", sep=';');
    return dataset;
 
#Prepare the data for LSTM networks
def get_data (name, dont_use_free_parking_var, group_day_type, include_time):
      
    scaler = None;
    #Get the CSV file
    dataset = read_and_process_csv(name, dont_use_free_parking_var, group_day_type, include_time);
    values = dataset.values
     
    # ensure all data is float
    values = values.astype('float32')
    
    features_temp = 8;
    if (group_day_type == 1 or group_day_type == 2):
        features_temp = features_temp - 1;
    
    if include_time == 1:
        features_temp = features_temp + 1;
   
    tem = np.array(values, copy=True);
    tem = tem[:, 0]

    
    reframed = None;
    if use_scaler == 1:
        # normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        reframed = series_to_supervised(scaled, nr_time_ids, nr_time_ids)
    else: 
        reframed = series_to_supervised(values, nr_time_ids, nr_time_ids)

    #reframed.to_csv("origin.csv", sep=';');
    values = reframed.values
    
    #delete free parking from entry variable
    if dont_use_free_parking_var == 1:
        indexs_to_remove = [];
        for i in range (0, nr_time_ids*features_temp, features_temp):
            indexs_to_remove.append(i);
        values = np.delete(values, indexs_to_remove, axis=1)
    
    if only_evaluate == 1:
        percentage_training = 0.0;
    else:
        percentage_training = 0.7;
    # split into train and test sets
    n_train_hours = int(values.shape[0]*percentage_training);
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
     
    n_features = calculate_n_features(dont_use_free_parking_var, group_day_type, include_time);
    # split into input and outputs
    n_obs = nr_time_ids * n_features;
    n_obs_f = nr_time_ids * features_temp;
    
    train_X, train_y = train[:, :n_obs], train[:, -n_obs_f:]
    train_y = np.array([x[0::features_temp] for x in train_y])
    
    test_X, test_y = test[:, :n_obs], test[:, -n_obs_f:]
    test_y = np.array([x[0::features_temp] for x in test_y])
        
    #np.savetxt("test_X_o.csv", test_X, delimiter=";");
    
    val_len = test_X.shape[0]//2;
    
    if only_evaluate == 1:
        val_len = 0;
        
    val_X = test_X[:val_len,:];
    test_X = test_X[val_len:, :];

    val_y = test_y[:val_len, :];
    test_y = test_y[val_len:, :];
 
    if use_scaler == 1:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(tem.reshape(-1,1));
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], nr_time_ids, n_features))
    test_X = test_X.reshape((test_X.shape[0], nr_time_ids, n_features))
    val_X = val_X.reshape((val_X.shape[0], nr_time_ids, n_features))

    return train_X, train_y, test_X, test_y, scaler, val_X, val_y;
 

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


# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch, dont_use_free_parking_var, group_day_type, include_time):
    # reshape input pattern to [samples, timesteps, features]
    n_features = calculate_n_features(dont_use_free_parking_var, group_day_type, include_time);
    X = X.reshape(1, nr_time_ids, n_features)
    # make forecast
    forecast = model.predict(X, batch_size=n_batch)
    # convert to array
    return [x for x in forecast[0, :]]
 
def inverse_transform(forecasts, scaler):

    inverted = list()
    for i in range(len(forecasts)):
        # create array from forecast
        forecast = array(forecasts[i])
        forecast = forecast.reshape(1, len(forecast))
        # invert scaling
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        # store
        inverted.append(inv_scale)
    return inverted

def evaluate_model(model_name, csv_file_name, dont_use_free_parking_var, group_day_type, include_time):
    
    print("Evaluate model: "+model_name+" for CSV file: "+csv_file_name);
    _, _, test_X, test_y, scaler, _, _ = get_data(csv_file_name, dont_use_free_parking_var, group_day_type, include_time);
    model = load_model(model_name);
    
    forecasts = list()
    for i in range(len(test_X)):
        # make forecast
        forecast = forecast_lstm(model, test_X[i], batch_size, dont_use_free_parking_var, group_day_type, include_time)
        # store the forecast
        forecasts.append(forecast);

    
    if use_scaler == 1:
        forecasts = inverse_transform(forecasts, scaler);
        test_y = inverse_transform(test_y, scaler);

    eval = list()
    for i in range(len(test_y)):
        actual = test_y[i]
        predicted = forecasts[i]
        rmse = sqrt(mean_squared_error(actual, predicted))
        eval.append(rmse);
        
    
    eval_name = csv_file_name.split("/")[-1].split(".")[0]
    np.savetxt("e{0}_{1}.csv".format(model_name, eval_name), eval, delimiter=";");
    #pyplot.plot(eval, color='blue');
    #pyplot.show();


    print("Model was evaluated");
     
def train_lstm(model_name, csv_file_name, dont_use_free_parking_var, group_day_type, include_time, neorons):
     
    train_X, train_y, _, _, _, val_X, val_y = get_data(csv_file_name, dont_use_free_parking_var, group_day_type, include_time);
   
    # design network
    model = Sequential()
    model.add(LSTM(neorons, batch_input_shape=(batch_size, train_X.shape[1], train_X.shape[2]), stateful=True))
    model.add(Dense(train_y.shape[1]))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    # fit network
    for i in range(epochs):
        print ("LSTM epoch: "+str(i+1)+"/"+str(epochs));
        model.fit(train_X, train_y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False, validation_data=(val_X, val_y))
        model.reset_states()

    save_model(model, model_name);
    print("Training the model has finished");

def calculate_n_features(dont_use_free_parking_var, group_day_type, include_time):
    n_features = 7;
    if dont_use_free_parking_var == 1:
        n_features = n_features - 1;
    
    if (group_day_type == 1 or group_day_type == 2):
        n_features = n_features - 1;
    
    if include_time == 1:
        n_features = n_features + 1;
    
    return n_features;
 
def create_model_name (dont_use_free_parking_var, group_day_type, include_time, neorons):
    
    name = "{0}{1}{2}".format(dont_use_free_parking_var, group_day_type, include_time);
    if neorons != 0:
        name = name+"_"+str(neorons);
        
    return name;

def train_and_evaluate (array_neurons, csv_file_name, dont_use_free_parking_var, group_day_type, include_time): 
    
    
    for i in array_neurons:
        neorons = i;
        name = create_model_name(dont_use_free_parking_var, group_day_type, include_time, neorons);
        print("{0}{1}{2}: {3}".format(dont_use_free_parking_var,group_day_type, include_time,neorons));
        train_lstm(name, csv_file_name, dont_use_free_parking_var, group_day_type, include_time, neorons);
        evaluate_model(name, csv_file_name, dont_use_free_parking_var, group_day_type, include_time);
    
    return;

# = [15, 20, 24, 30, 38, 45, 52, 60, 68, 75];
def train_and_evaluate_all (array_neurons, csv_file_name):
    
    print ("----START----");
    
    #Config 1
    dont_use_free_parking_var = 0; 
    group_day_type = 0;
    include_time = 0;
    train_and_evaluate (array_neurons, csv_file_name, dont_use_free_parking_var, group_day_type, include_time);
   
    #Config 2
    dont_use_free_parking_var = 0; 
    group_day_type = 0;
    include_time = 1;
    train_and_evaluate (array_neurons, csv_file_name, dont_use_free_parking_var, group_day_type, include_time);

    # Config 3
    dont_use_free_parking_var = 0; 
    group_day_type = 1;
    include_time = 0;
    train_and_evaluate (array_neurons, csv_file_name, dont_use_free_parking_var, group_day_type, include_time);
  
    #Config 4
    dont_use_free_parking_var = 0; 
    group_day_type = 1;
    include_time = 1;
    train_and_evaluate (array_neurons, csv_file_name, dont_use_free_parking_var, group_day_type, include_time);
    
    #Config 5 
    dont_use_free_parking_var = 1; 
    group_day_type = 0;
    include_time = 0;
    train_and_evaluate (array_neurons, csv_file_name, dont_use_free_parking_var, group_day_type, include_time);

    #Config 6
    dont_use_free_parking_var = 1; 
    group_day_type = 0;
    include_time = 1;
    train_and_evaluate (array_neurons, csv_file_name, dont_use_free_parking_var, group_day_type, include_time);
   
    #Config 7
    dont_use_free_parking_var = 1; 
    group_day_type = 1;
    include_time = 0;
    train_and_evaluate (array_neurons, csv_file_name, dont_use_free_parking_var, group_day_type, include_time);
     
    #Config 8
    dont_use_free_parking_var = 1; 
    group_day_type = 1;
    include_time = 1;
    train_and_evaluate (array_neurons, csv_file_name, dont_use_free_parking_var, group_day_type, include_time);    

    print ("----END----");


def get_all_garages():
    onlyfiles = [];
    for root, dirs, files in os.walk("Garages/"):  
        for filename in files:
            onlyfiles.append(root+"/"+filename); 
    return onlyfiles;

def evaluate_all ():
    
    garages = get_all_garages();
    
    for garage in garages:
        
         #Config 1
         dont_use_free_parking_var = 0; 
         group_day_type = 0;
         include_time = 0;
         evaluate_model(create_model_name(dont_use_free_parking_var, group_day_type, include_time, 0), garage, dont_use_free_parking_var, group_day_type, include_time);
   
         #Config 2
         dont_use_free_parking_var = 0; 
         group_day_type = 0;
         include_time = 1;
         evaluate_model(create_model_name(dont_use_free_parking_var, group_day_type, include_time, 0), garage, dont_use_free_parking_var, group_day_type, include_time);
    
        # Config 3
         dont_use_free_parking_var = 0; 
         group_day_type = 1;
         include_time = 0;
         evaluate_model(create_model_name(dont_use_free_parking_var, group_day_type, include_time, 0), garage, dont_use_free_parking_var, group_day_type, include_time);  
    
         #Config 4
         dont_use_free_parking_var = 0; 
         group_day_type = 1;
         include_time = 1;
         evaluate_model(create_model_name(dont_use_free_parking_var, group_day_type, include_time, 0), garage, dont_use_free_parking_var, group_day_type, include_time);
      
         #Config 5 
         dont_use_free_parking_var = 1; 
         group_day_type = 0;
         include_time = 0;
         evaluate_model(create_model_name(dont_use_free_parking_var, group_day_type, include_time, 0), garage, dont_use_free_parking_var, group_day_type, include_time);
  
         #Config 6
         dont_use_free_parking_var = 1; 
         group_day_type = 0;
         include_time = 1;
         evaluate_model(create_model_name(dont_use_free_parking_var, group_day_type, include_time, 0), garage, dont_use_free_parking_var, group_day_type, include_time);

         #Config 7
         dont_use_free_parking_var = 1; 
         group_day_type = 1;
         include_time = 0;
         evaluate_model(create_model_name(dont_use_free_parking_var, group_day_type, include_time, 0), garage, dont_use_free_parking_var, group_day_type, include_time);

         #Config 8
         dont_use_free_parking_var = 1; 
         group_day_type = 1;
         include_time = 1;
         evaluate_model(create_model_name(dont_use_free_parking_var, group_day_type, include_time, 0), garage, dont_use_free_parking_var, group_day_type, include_time);
    print ("----END----");
    
    return ;

#train_and_evaluate_all ([1, 2],"DataCSV/8_8353.csv");
#print("DataCSV/8_8353.csv".split("/")[-1].split(".")[0]);
#evaluate_all();
#evaluate_model(create_model_name(0, 0, 0, 0), "Garages/AnnArbor/4_FI10.csv", 0, 0, 0);  
    

