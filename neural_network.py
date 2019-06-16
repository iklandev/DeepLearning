import logging
import pandas as pd
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from math import sqrt
from sklearn.metrics import mean_squared_error
from statistics import mean

#Global config
activation_function = 'relu';
optimizer = 'adam';
loss = 'mse';
metrics=['mae'];
model_root_folder = "NeuralNetworks/";
processed_root_folder = "Processed/";
save_actual_and_forecast_data = 1;
epochs = 100;
csv_file_name = "5_G17"; 


ALL_FILES = ['1_12', '1_14', '1_25', '1_26', '1_27', '1_28', '1_2', '1_30', '1_33', '1_35', '1_36', '1_37', '1_38', '1_39', '1_40', '1_41', '1_42', '1_4', '1_5', '2_1', '2_2', '2_3', '2_5', '2_6', '2_9', '3_BI', '3_CI', '3_RA', '3_WA', '4_AN7', '4_FI10', '4_FI2', '4_FO1', '4_FO4', '4_FO5', '4_LI6', '4_LI8', '4_MA3', '4_SO9', '5_G11', '5_G14', '5_G16', '5_G17', '5_G3', '5_G4', '5_G7', '5_G9', '6_0', '6_10', '6_11', '6_12', '6_13', '6_14', '6_1', '6_2', '6_3', '6_5', '6_7', '6_8', '6_9', '7_10', '7_12', '7_1', '7_2', '7_3', '7_7', '7_8', '7_9', '8_667276', '8_6767', '8_764978', '8_765178', '8_765283', '8_765383', '8_765678', '8_76', '8_8068', '8_8349', '8_8350', '8_8351', '8_8352', '8_8353', '8_8354', '8_8355', '8_8356', '8_8357'];


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, filename='neural_network.log');

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
def read_csv (name, use_free_parking, include_time, percentage_training):
    
    in_file = "{0}{1}_in_{2}{3}.csv".format(processed_root_folder, name, use_free_parking, include_time);
    out_file = "{0}{1}_out.csv".format(processed_root_folder, name);   
    
    input_data = pd.read_csv(in_file, sep=';', index_col=False, header=None); 
    output_data = pd.read_csv(out_file, sep=';', index_col=False, header=None); 
              
    train_X = input_data.values;
    train_Y = output_data.values;     
         
    n_train_data = int(train_X.shape[0]*percentage_training);
    
    test_X = train_X[n_train_data:, :];
    train_X = train_X[:n_train_data, :];
    
    test_Y = train_Y[n_train_data:, :];
    train_Y = train_Y[:n_train_data, :];  
                
    return train_X, train_Y, test_X, test_Y; 

def train (neurons, X_train, Y_train, output_dim):
    
    logging.info('train=>neurons: {0};'.format(neurons));
    input_dim = len(X_train[0]);
        
    #Initializing Neural Network
    model = Sequential();
     
    # Adding the input layer and the first hidden layer
    model.add(Dense(neurons, input_dim=input_dim, activation= activation_function))
    model.add(Dense(neurons//4, activation=activation_function))
    model.add(Dense(output_dim, activation='sigmoid'))

    # Compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(X_train, Y_train, epochs=epochs, batch_size=10, verbose=1, shuffle=False, validation_split = 0.2);
    
    logging.info('train=>finish');    
    return model;

def evaluate_model (model, X_test, Y_test, model_name, garage):
    
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
        np.savetxt("{0}_detail_{1}.csv".format(garage.replace('DataCSV/', '').replace('Garages//', '').replace('.csv', ''), model_name), np.array(arr2), delimiter=";", fmt='%s');  
    
    return mean(eval);
    
    

def train_and_evaluate(array_neurons, file_name, nr_time_ids, use_free_parking, include_time, percentage_training):
    
    logging.info('train_and_evaluate=>start');
    train_X, train_Y, test_X, test_Y = read_csv(csv_file_name, use_free_parking, include_time, percentage_training);
    
    np.savetxt("test_X.csv", test_X, delimiter=";");
    np.savetxt("test_Y.csv", test_Y, delimiter=";");
    
    for neurons in array_neurons:
        logging.info('train_and_evaluate=>neurons: {0};'.format(neurons));
        model_name = "{0}{1}_{2}".format(use_free_parking, include_time, neurons);
        model = train(neurons, train_X, train_Y, nr_time_ids);
        save_model(model, model_name);
        mean_error = evaluate_model(model, test_X, test_Y, model_name, csv_file_name);
        f = open('evaluates.csv','a')  # w : writing mode  /  r : reading mode  /  a  :  appending mode
        f.write('{0};{1}\n'.format(model_name, mean_error));
        f.close();
    
    logging.info('train_and_evaluate=>finish');
    return;


def train_networks():    

    percentage_training = 0.8;
    nr_time_ids = 15;
    array_neurons = [25];
    ''' 
    #Config_1
    use_free_parking = 0; 
    include_time = 0;
    train_and_evaluate(array_neurons, csv_file_name, nr_time_ids, use_free_parking, include_time, percentage_training);

    #Config_2
    use_free_parking = 0; 
    include_time = 1;
    train_and_evaluate(array_neurons, csv_file_name, nr_time_ids, use_free_parking, include_time, percentage_training);

    #Config_3
    use_free_parking = 1; 
    include_time = 0;
    train_and_evaluate(array_neurons, csv_file_name, nr_time_ids, use_free_parking, include_time, percentage_training);'''

    #Config_4
    use_free_parking = 1; 
    include_time = 1;
    train_and_evaluate(array_neurons, csv_file_name, nr_time_ids, use_free_parking, include_time, percentage_training);
    
    return;

def evaluate(garage, use_free_parking, include_time):
    
    logging.info("Evaluate garage: {0} for {1}{2}".format(garage, use_free_parking, include_time));
    
    percentage_training = 0.0;
    _, _, test_X, test_Y = read_csv(garage, use_free_parking, include_time, percentage_training);
    
    
    model_name = "{0}{1}".format(use_free_parking, include_time);
    model = load_model(model_name);
    
    mean_error = evaluate_model(model, test_X, test_Y, model_name, garage);
    f = open('evaluates_all.csv','a')  # w : writing mode  /  r : reading mode  /  a  :  appending mode
    f.write('{0}_{1};{2}\n'.format(garage, model_name, mean_error));
    f.close();

    return;

def evaluate_networks(): 
    for garage in ALL_FILES:
        
        print(garage);
        
        use_free_parking = 0; 
        include_time = 0;
        evaluate(garage, use_free_parking, include_time);
        
        use_free_parking = 0; 
        include_time = 1;
        evaluate(garage, use_free_parking, include_time);
        
        use_free_parking = 1; 
        include_time = 0;
        evaluate(garage, use_free_parking, include_time);
        
        use_free_parking = 1; 
        include_time = 1;
        evaluate(garage, use_free_parking, include_time);
        
       
    return;

train_networks();
#evaluate_networks();
