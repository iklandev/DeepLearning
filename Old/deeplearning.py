import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.models import model_from_json


#Define the netwok properties:
activation_function = 'relu';
optimizer = 'adam';
loss = 'mse';
metrics=['mae'];

#Define the data properties:
categorize_day_type = 0;

# fix random seed for reproducibility
np.random.seed(7)

ALL_COLUMNS = ['Date','Time','TimeID','DayType','LotID','CityID','LotName','FreeParking','TotalJamINRadius_1','TotalJamOUTRadius_1','TotalJamINRadius_2','TotalJamOUTRadius_2','TotalJamINRadius_3','TotalJamOUTRadius_3','AvgINForRadius_1','AvgOUTForRadius_1','AvgINForRadius_2','AvgOUTForRadius_2','AvgINForRadius_3','AvgOUTForRadius_3'];
PROCESS_COLUMNS = ['TimeID','DayType', 'FreeParking','TotalJamINRadius_1','TotalJamOUTRadius_1','TotalJamINRadius_2','TotalJamOUTRadius_2','TotalJamINRadius_3','TotalJamOUTRadius_3'];

def save_model (model, name):
    # serialize model to JSON
    model_json = model.to_json()
    with open("NeuralNetworks/"+name+".json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("NeuralNetworks/"+name+".h5")
    print("Saved model to disk")
    
def load_model (name):
    
    # load json and create model
    json_file = open("NeuralNetworks/"+name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("NeuralNetworks/"+name+".h5")
    print("Loaded model from disk")
    return loaded_model

def categorize_day (x):
    if x<6:
        return 0;
    else:
        return 1;

def init_data ():
    
    # Read data 
    dataset = pd.read_csv("DataCSV/5_G22-Seattle.csv", sep=';', usecols=PROCESS_COLUMNS);
    dataset_test = pd.read_csv("DataCSV/5_G22-Seattle_Test.csv", sep=';', usecols=PROCESS_COLUMNS);

    if categorize_day_type == 1:
        dataset['DayType'] = dataset['DayType'].apply(categorize_day);
        dataset_test['DayType'] = dataset_test['DayType'].apply(categorize_day);
        print("day type is categorized");
        
    #Create matrix of features X and matrix of target variable Y
    X_train = dataset[['TimeID','DayType', 'TotalJamINRadius_1','TotalJamOUTRadius_1','TotalJamINRadius_2','TotalJamOUTRadius_2','TotalJamINRadius_3','TotalJamOUTRadius_3']].values
    Y_train = dataset[['FreeParking']].values

    X_test = dataset_test[['TimeID','DayType', 'TotalJamINRadius_1','TotalJamOUTRadius_1','TotalJamINRadius_2','TotalJamOUTRadius_2','TotalJamINRadius_3','TotalJamOUTRadius_3']].values
    Y_test = dataset_test[['FreeParking']].values
    
    #Encoding string variables - TimeID slot
    labelencoder_X_1 = LabelEncoder();
    X_train[:, 0] = labelencoder_X_1.fit_transform(X_train[:, 0]);
    X_test[:, 0] = labelencoder_X_1.fit_transform(X_test[:, 0]);

    #Create dummy variable for encoded string variables for TimeID and DayType
    onehotencoder = OneHotEncoder(categorical_features = [0, 1], dtype = np.int64)
    X_train = onehotencoder.fit_transform(X_train).toarray()
    X_test = onehotencoder.fit_transform(X_test).toarray()

    # Feature Scaling
    sc = StandardScaler()   
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    return X_train, Y_train, X_test, Y_test

def init_model(input_dim):
    
    #Initializing Neural Network
    model = Sequential();
 
    # Adding the input layer and the first hidden layer
    model.add(Dense(input_dim, input_dim=input_dim, activation=activation_function))
    model.add(Dense(10, activation=activation_function))
    model.add(Dense(1))

    # Compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model;
    

def train (name, epochs, batch_size):

    X_train, Y_train, X_test, Y_test = init_data();
    model = init_model(len(X_train[0]))
    
    # Fitting our model 
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    save_model(model, name);
    
    plt.plot(history.history['mean_absolute_error'])
    plt.title('Mean Absolute Error trought epochs')
    plt.ylabel('mean_absolute_error')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
    
    
def evaluate_model (name, nubmer_of_records):
    
    X_train, Y_train, X_test, Y_test = init_data();
    model = load_model(name);
    model.compile(optimizer='adam', loss='mse', metrics=['mae']);
    
    scores = model.evaluate(X_test, Y_test);
    print(scores);
    
    #Evaluate the model
    score = model.predict(X_test)

    plt.plot(Y_test[:nubmer_of_records])
    plt.plot(score[:nubmer_of_records])
    plt.ylabel('Number of parking spaces')
    plt.xlabel('Numebr of records')
    plt.legend(['Test Data', 'Predicted Data'], loc='upper left')
    plt.show()

def train_and_evaluate ():
    
    nr_layers = 1;
    nr_neurons_1_layer = 2;
    nr_neurons_2_layer = 2;
    X_train, Y_train, X_test, Y_test = init_data();
    
    while (nr_layers <3):
             
              
        if (nr_layers == 2):
            nr_neurons_2_layer = nr_neurons_2_layer+1 

        model = Sequential();
        model.add(Dense(nr_neurons_1_layer, input_dim=len(X_train[0]), activation=activation_function)) 
        if (nr_layers == 2):
            model.add(Dense(nr_neurons_2_layer, activation=activation_function));
        model.add(Dense(1))
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics);
        
        
        history = model.fit(X_train, Y_train, epochs=500, batch_size=5, verbose=1)
        scores = model.evaluate(X_test, Y_test);
                
        f = open('networks.csv','a')
        f.write(str(nr_layers)+";"
                +str(nr_neurons_1_layer)+";"
                +str(nr_neurons_2_layer)+";"
                +"{0:.2f}".format(round(history.history['mean_absolute_error'][-1],2))+";"
                +"{0:.2f}".format(round(scores[-1],2))
                +"\n");
        f.close();
                    
        if(nr_neurons_1_layer == 42):
            nr_neurons_1_layer = 2;
            nr_layers = nr_layers +1;
            if (nr_layers == 2):
                nr_neurons_1_layer = 4;
                continue;
            

        if(nr_layers==1 or (nr_neurons_2_layer+1==nr_neurons_1_layer)):
            nr_neurons_1_layer = nr_neurons_1_layer+2; 
            nr_neurons_2_layer = 1; 



#train ("5_G22-Seattle", 1, 5);
#evaluate_model ("5_G22-Seattle", 1000);
train_and_evaluate();