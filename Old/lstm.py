from datetime import datetime
from keras.layers import Dense, LSTM
from keras.models import Sequential, model_from_json
from math import sqrt
from matplotlib import pyplot
from pandas import concat
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

 
ALL_COLUMNS = ['Date','Time','TimeID','DayType','LotID','CityID','LotName','FreeParking','TotalJamINRadius_1','TotalJamOUTRadius_1','TotalJamINRadius_2','TotalJamOUTRadius_2','TotalJamINRadius_3','TotalJamOUTRadius_3','AvgINForRadius_1','AvgOUTForRadius_1','AvgINForRadius_2','AvgOUTForRadius_2','AvgINForRadius_3','AvgOUTForRadius_3'];
PROCESS_COLUMNS = ['Date', 'TimeID','DayType', 'FreeParking','TotalJamINRadius_1','TotalJamOUTRadius_1','TotalJamINRadius_2','TotalJamOUTRadius_2','TotalJamINRadius_3','TotalJamOUTRadius_3'];
REINDEX_COLUMNS = ['FreeParking', 'DayType', 'TotalJamINRadius_1','TotalJamOUTRadius_1','TotalJamINRadius_2','TotalJamOUTRadius_2','TotalJamINRadius_3','TotalJamOUTRadius_3'];
 


#number of previous time ID's. 15 Time ID's is equal to 1 hour 
nr_time_ids = 15;
n_features = 8;
# fix random seed for reproducibility
np.random.seed(7)

#config
csv_file_name = "DataCSV/5_G22-Seattle.csv";
model_name = "Seattle_LSTM";
optimizer = "adam";
loss = "mse";
metrics = ['mae'];
epochs = 50;
batch_size = 5;
percentage_training = 0.8;
model_root_folder = "NeuralNetworks/";

 
def parse(x):
    return datetime.strptime(x, '%d/%m/%Y %H:%M')
 
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
 
    return dataset;
 
#Prepare the data for LSTM networks
def get_data (name):
     
    #Get the CSV file
    dataset = read_and_process_csv(name);
    values = dataset.values
     
    # ensure all data is float
    values = values.astype('float32')
     
    # normalize features
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #scaled = scaler.fit_transform(values)
     
    reframed = series_to_supervised(values, nr_time_ids, 1)
    values = reframed.values
    
    # split into train and test sets
    #n_train_hours
    n_train_hours = int(values.shape[0]*percentage_training);
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
     
    # split into input and outputs
    n_obs = nr_time_ids * n_features
    train_X, train_y = train[:, :n_obs], train[:, -n_features]
    test_X, test_y = test[:, :n_obs], test[:, -n_features]
     
 
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], nr_time_ids, n_features))
    test_X = test_X.reshape((test_X.shape[0], nr_time_ids, n_features))

    return train_X, train_y, test_X, test_y;
 

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

def evaluate_model():
    
    _, _, test_X, test_y = get_data(csv_file_name);
    model = load_model(model_name);
    
    #Evaluate the model
    score = model.predict(test_X)
    
    # calculate RMSE
    rmse = sqrt(mean_squared_error(score, test_y))
    print('Test RMSE: %.3f' % rmse)
    
 
    pyplot.plot(test_y)
    pyplot.plot(score)
    pyplot.ylabel('Number of parking spaces')
    pyplot.xlabel('Numebr of records')
    pyplot.legend(['Test Data', 'Predicted Data'], loc='upper left')
    pyplot.show()

    print("Model was evaluated");
     
def train_lstm():
     
    train_X, train_y, test_X, test_y = get_data(csv_file_name);
   
    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
 

    model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_X, test_y), verbose=1, shuffle=False)

    save_model(model, model_name);
    print("Training the model has finished");
   
 
   
train_lstm();
evaluate_model();