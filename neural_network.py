import logging
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

#Global config
csv_file_name = "DataCSV/5_G17.csv";

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


#Read the CSV file and prepare data
#name - name of the CSV file
#nr_time_ids - number of time slots - 15 is 1h
#day_types - 1=>week day; 2=> weekend;
#use_free_parking - use free parking spaces as entry in network
#include_time - use time as entry in network
#include_day_type - use day type as entry in network
def read_and_process_csv (name, nr_time_ids, day_types, use_free_parking, include_time, include_day_type=0):
    
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
        print(index);
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
        
            
    
    return np.asarray(input_data), np.asarray(output_data);



#Config_1
use_free_parking_var = 0; 
include_time = 0;
inp, oyt =  read_and_process_csv(csv_file_name, 15, 1, use_free_parking_var, include_time);
#np.savetxt("in_{0}{1}.csv".format(use_free_parking_var, include_time), inp, delimiter=";");


#Config_2
use_free_parking_var = 0; 
include_time = 1;
inp, oyt =  read_and_process_csv(csv_file_name, 15, 1, use_free_parking_var, include_time);

#Config_3
use_free_parking_var = 1; 
include_time = 0;
inp, oyt =  read_and_process_csv(csv_file_name, 15, 1, use_free_parking_var, include_time);

#Config_4
use_free_parking_var = 1; 
include_time = 1;
inp, oyt =  read_and_process_csv(csv_file_name, 15, 1, use_free_parking_var, include_time);
print("Don4e");