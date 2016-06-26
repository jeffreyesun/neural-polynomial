import csv
import time
from time import strptime
import calendar
import re
from numpy import array as npa


def training_data():
    training_data = []
    pointfulfieldsin = [0,2,5,7,8]
    pffo = [10,11]    
    with open("../../data/train.csv", "rU") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            inputs = [row[i] for i in pointfulfieldsin]
            outputs = [row[i] for i in pffo]
            
            inputs = sanitiz(inputs)
            outputs = santrain(outputs)
            
            training_data.append((inputs, outputs))

    return training_data

#def prediction_data():
#    testing_Data = []
#    pointfulfieldsin = [1,3,6,8,9]




    #Get all values between about -1 and 1. TODO: this programmatically
def sanitiz(row):
    timenum =calendar.timegm(strptime(row[0], "%m/%d/%y"))/300000000.0 - 4
        
    regex = re.compile(r'[^\d.]+')
        
    row[0] = npa([timenum])
    row[1] = npa([float(row[1])])
    row[2] = npa([float(regex.sub('',row[2][1:]))])
    row[3] = npa([float(row[3])*10-419])
    row[4] = npa([float(row[4])*10+877])
    
    return npa(row)

def santrain(row):
    row[0] = npa([float(row[0])/50.0])
    row[1] = npa([float(row[1])])
    return npa(row)
