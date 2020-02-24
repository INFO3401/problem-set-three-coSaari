
import pandas as pd
import numpy as py

def loadAndCleanData(filename):
    data = pd.read_csv(filename)
    columns = list(data)
    for i in columns:
        data[i].fillna(0,inplace=True)
    return data

print()
print("This is the new creditData.csv after it was passed through load and clean data.")
print()
print(loadAndCleanData("creditData.csv"))

def computePDF(feature,filename):
    data1 = pd.read_csv(filename)
    ax1 = data1[feature].plot.kde()
    ax1.set_xlabel(feature)
    
import matplotlib.pyplot as plt

def printPDF():
    df = pd.read_csv("creditData.csv")
    columns = list(df)
    for i in columns:
        computePDF(i, "creditData.csv")
        plt.show()

print("These are the PDF graphs")
print()
print(printPDF())

def viewLogDistribution(column,filename):
    data1 = pd.read_csv(filename)
    ax1 = data1[column].plot(kind = 'hist',bins = 20, log = True)
    ax1.set_xlabel(column)
    
import matplotlib.pyplot as plt

def printLog():
    df = pd.read_csv("creditData.csv")
    columns = list(df)
    for i in columns:
        x = viewLogDistribution(i, "creditData.csv")
        plt.show()

print("These are the log graphs")
print()
print(printLog())

import pandas as pd
import numpy as np

def computeDefaultRisk(columnName, binSize, targetFeature, dataframe):
    columns = list(dataframe)
    
    if (targetFeature in columns) and (columnName in columns):
        dataframe[columnName] = np.where((dataframe[columnName] >= binSize[0]) & (dataframe[columnName] <= binSize[-1]), 1, 0)
        dataframe["count"] = 1
        
        newDF = dataframe[[columnName,targetFeature,"count"]]
         
        pivotT = pd.pivot_table(newDF, values="count",index=[columnName],columns=[targetFeature],aggfunc=np.size, fill_value=0)
        
        
        one = pivotT.loc[0,0]
        two = pivotT.loc[0,1]
        three = pivotT.loc[1,0]
        four = pivotT.loc[1,1]
        
        probDlqin = ((three + four) / (one + two + three + four))
        probTarg = ((two + four) / (one + two + three + four))
        
        targAndDlqin = (four / (one + two + three + four))

        prob = targAndDlqin/probTarg
        
        #print(one,two,three,four)
        return(prob)

def makeDF():
    df = pd.read_csv("creditData.csv")
    return df
    
def findRisks():
    riskDictionary = {}

    x = makeDF()
    featureDict = {"RevolvingUtilizationOfUnsecuredLines" : [[0, 0.0544],[0.0544, 0.38],[0.38, 50708.0]],"age" : [[0,40],[40,60],[60,115]],
               "NumberOfTime30-59DaysPastDueNotWorse" : [[0,0.5],[0.5,1],[1,98]],"DebtRatio":[[0, 0.238],[0.238, 0.57],[0.57, 329664.0]],
               "MonthlyIncome" : [[0,4000],[4000,7080],[7080,3009000]], "NumberOfOpenCreditLinesAndLoans" : [[-0.001, 15],[15, 30],[30, 58.0]],
               "NumberOfTimes90DaysLate" : [[0,5],[5,20],[20,100]], "NumberRealEstateLoansOrLines": [[0,5],[5,15],[15,60]],
               "NumberOfTime60-89DaysPastDueNotWorse" : [[0,3],[3,20],[20,100]], "NumberOfDependents" : [[0,2.5],[2.5,7],[7,20]]}
    print("These are each of the bins with their corresponding risk")
    print()
    for a in featureDict:
        riskDictionary[a] = {}
        for i in featureDict[a]:
            y = x.copy()
            risk = computeDefaultRisk(a,i,"SeriousDlqin2yrs",y)
            riskDictionary[a][str(i)] = risk
    
            print("Feature : ")
            print("\t" + a)
            
            print("Bin : ")
            print("\t" + str(i))
            
            print("Risk : ")
            print("\t" + str(risk))
            
            print()
            print()
            
        
    print("This is the nested risk dictionary, with the features, their corresponding bins and the corresponding risk for that bin.")
    print()
    return riskDictionary
        
print(findRisks())


import pandas as pd
def binSplit(filename):
    dataframe = pd.read_csv(filename)
    columns = list(dataframe)
    for i in columns:
        x = pd.qcut(dataframe[i],q=3,duplicates='drop')
        print(i)
        print(x.unique())
        print()
    
print("These are the features split into bins")
print()
binSplit("creditData.csv")

print("This is the new dataset for newLoans.csv after it was passed through loadAndCleanData()")
print()
print(loadAndCleanData("newLoans.csv"))


def predictDefaultRisk():
    x = loadAndCleanData("newLoans.csv")
    y = findRisks()
    print()

    columns = list(pd.read_csv("creditData.csv"))
    z = 45

    for i in columns:
        if i != "SeriousDlqin2yrs":
            print(i)
            print(y[i])

    #for index, row in x.iterrows():
        #for 
        #print(row["age"])
