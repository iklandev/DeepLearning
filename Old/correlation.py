import os;
import pandas as pn
import numpy as np


CSV_FLIE_PATH = 'CSV';
PROCESSED_CSV_FLIE_PATH = 'ProcessedCSV';
ALL_COLUMNS = ['Date','Time','TimeID','DayType','LotID','CityID','LotName','FreeParking','TotalJamINRadius_1','TotalJamOUTRadius_1','TotalJamINRadius_2','TotalJamOUTRadius_2','TotalJamINRadius_3','TotalJamOUTRadius_3','AvgINForRadius_1','AvgOUTForRadius_1','AvgINForRadius_2','AvgOUTForRadius_2','AvgINForRadius_3','AvgOUTForRadius_3'];
PROCESS_COLUMNS = ['TimeID','DayType','LotID', 'FreeParking','TotalJamINRadius_1','TotalJamOUTRadius_1','TotalJamINRadius_2','TotalJamOUTRadius_2','TotalJamINRadius_3','TotalJamOUTRadius_3','AvgINForRadius_1','AvgOUTForRadius_1','AvgINForRadius_2','AvgOUTForRadius_2','AvgINForRadius_3','AvgOUTForRadius_3'];

LOT_ID ='LotID'; 
DAY_TYPE ='DayType';
CORR_TOTAL_IN1 = 'CTIN1';
CORR_TOTAL_OUT1 = 'CTOUT1';
CORR_TOTAL_IN2 = 'CTIN2';
CORR_TOTAL_OUT2 = 'CTOUT2';
CORR_TOTAL_IN3 = 'CTIN3';
CORR_TOTAL_OUT3 = 'CTOUT3';

STD_TOTAL_IN1 = 'STDCTIN1';
STD_TOTAL_OUT1 = 'STDCTOUT1';
STD_TOTAL_IN2 = 'STDCTIN2';
STD_TOTAL_OUT2 = 'STDCTOUT2';
STD_TOTAL_IN3 = 'STDCTIN3';
STD_TOTAL_OUT3 = 'STDCTOUT3'; 

'''
Return the file path for the CSV file
@name: string
     name of the CSV file 
'''
def get_CSV_file_path (name):
    module_dir = os.path.dirname(__file__);  # get current directory
    
    return os.path.join(module_dir, name);

'''
Create data frame for the CSV file
@name: string
     name of the CSV file 
@columns: array
     subset of the columns to be read from the CSV file
'''
def create_data_frame (name, columns):
    
    return pn.read_csv(get_CSV_file_path(name), sep=';', index_col=False, usecols=columns, warn_bad_lines=False );



'''
Main program
'''
def generateAllCorrelation ():
    
    allCsv = os.listdir(CSV_FLIE_PATH);
    
    for x in allCsv:
        temp = create_data_frame(CSV_FLIE_PATH+"\\"+x, PROCESS_COLUMNS);
        t = temp.groupby(['LotID', 'DayType', 'TimeID'], as_index=False).mean(); 
        
        p = t.groupby('LotID');
        result = pn.DataFrame(columns=[LOT_ID, DAY_TYPE,
                                       CORR_TOTAL_IN1, CORR_TOTAL_OUT1,
                                       CORR_TOTAL_IN2,CORR_TOTAL_OUT2,
                                       CORR_TOTAL_IN3, CORR_TOTAL_OUT3]);
        i=0;
        for name, group in p:
            print name;
            f = group.groupby('DayType');
            for na, gr in f:
                print na;
                TotalJamINRadius_1 = gr['FreeParking'].corr(gr['TotalJamINRadius_1'])
                #print "TotalJamINRadius_1: "+str(TotalJamINRadius_1);
                TotalJamOUTRadius_1 = gr['FreeParking'].corr(gr['TotalJamOUTRadius_1'])
                #print "TotalJamOUTRadius_1: "+str(TotalJamOUTRadius_1);
                TotalJamINRadius_2 = gr['FreeParking'].corr(gr['TotalJamINRadius_2'])
                #print "TotalJamINRadius_2: "+str(TotalJamINRadius_2);
                TotalJamOUTRadius_2 = gr['FreeParking'].corr(gr['TotalJamOUTRadius_2'])
                #print "TotalJamOUTRadius_2: "+str(TotalJamOUTRadius_2);
                TotalJamINRadius_3 = gr['FreeParking'].corr(gr['TotalJamINRadius_3'])
                #print "TotalJamINRadius_3: "+str(TotalJamINRadius_3);
                TotalJamOUTRadius_3 = gr['FreeParking'].corr(gr['TotalJamOUTRadius_3'])
                #print "TotalJamOUTRadius_3: "+str(TotalJamOUTRadius_3);

                result.loc[i] = [name, na, 
                                 TotalJamINRadius_1, TotalJamOUTRadius_1,
                                 TotalJamINRadius_2, TotalJamOUTRadius_2,
                                 TotalJamINRadius_3, TotalJamOUTRadius_3];

                i=i+1;
        result.to_csv(path_or_buf=PROCESSED_CSV_FLIE_PATH+"\\"+x, sep=';', index=False, float_format='%.2f'); 

        

               
    return 0;
            
#t = create_data_frame(CSV_FLIE_PATH+"\Seattle2.csv", PROCESS_COLUMNS);
##tt = t.groupby(['LotID', 'DayType', 'TimeID'], as_index=False).mean(); 
#tt.to_csv(path_or_buf=PROCESSED_CSV_FLIE_PATH+"\Seattle2.csv", sep=';', index=False, float_format='%.2f');
print "Start";
generateAllCorrelation();

print "End";



        #Calculate standard deviation and standard error deviation
        #standardDeviationTotalJamINRadius_1=np.std(result[CORR_TOTAL_IN1]); 
        #standardDeviationTotalJamOUTRadius_1=np.std(result[CORR_TOTAL_OUT1]);
        #standardDeviationTotalJamINRadius_2=np.std(result[CORR_TOTAL_IN2]);
        #standardDeviationTotalJamOUTRadius_2=np.std(result[CORR_TOTAL_OUT2]);
        #standardDeviationTotalJamINRadius_3=np.std(result[CORR_TOTAL_IN3]);
        #standardDeviationTotalJamOUTRadius_3=np.std(result[CORR_TOTAL_OUT3]);
''' meanDeviationTotalJamINRadius_1=np.mean(result[CORR_TOTAL_IN1]); 
        meanDeviationTotalJamOUTRadius_1=np.mean(result[CORR_TOTAL_OUT1]);
        meanDeviationTotalJamINRadius_2=np.mean(result[CORR_TOTAL_IN2]);
        meanDeviationTotalJamOUTRadius_2=np.mean(result[CORR_TOTAL_OUT2]);
        meanDeviationTotalJamINRadius_3=np.mean(result[CORR_TOTAL_IN3]);
        meanDeviationTotalJamOUTRadius_3=np.mean(result[CORR_TOTAL_OUT3]);
        
        
        result2 = pn.DataFrame(columns=[LOT_ID, DAY_TYPE,
                                       CORR_TOTAL_IN1, STD_TOTAL_IN1,
                                       CORR_TOTAL_OUT1, STD_TOTAL_OUT1,
                                       CORR_TOTAL_IN2, STD_TOTAL_IN2,
                                       CORR_TOTAL_OUT2, STD_TOTAL_OUT2,
                                       CORR_TOTAL_IN3, STD_TOTAL_IN3,
                                       CORR_TOTAL_OUT3, STD_TOTAL_OUT3]);
        j = 0;
        for index, row in result.iterrows():
            result2.loc[j] = [row[LOT_ID], row[DAY_TYPE],
                             row[CORR_TOTAL_IN1], abs(row[CORR_TOTAL_IN1] - meanDeviationTotalJamINRadius_1),
                             row[CORR_TOTAL_OUT1], abs(row[CORR_TOTAL_OUT1] - meanDeviationTotalJamOUTRadius_1),
                             row[CORR_TOTAL_IN2], abs(row[CORR_TOTAL_IN2] - meanDeviationTotalJamINRadius_2),
                             row[CORR_TOTAL_OUT2], abs(row[CORR_TOTAL_OUT2] - meanDeviationTotalJamOUTRadius_2),
                             row[CORR_TOTAL_IN3], abs(row[CORR_TOTAL_IN3] - meanDeviationTotalJamINRadius_3),
                             row[CORR_TOTAL_OUT3], abs(row[CORR_TOTAL_OUT3] - meanDeviationTotalJamOUTRadius_3)
                             ];
            j=j+1;     
        
        '''