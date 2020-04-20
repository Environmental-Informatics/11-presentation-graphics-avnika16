#!/bin/env python
# Program 11- Solution file for Assignment 11 of ABE651. Uses input dataframes of river discharge statistics, processes them and plots figures of daily flow, coefficient of variation, TQmean, R-B index, mean annual monthly flow and return period of peak flow events.
# Author: Avnika Manaktala

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs

def ReadData( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    the metrics from the assignment on descriptive statistics and 
    environmental metrics.  Works for both annual and monthly metrics. 
    Date column should be used as the index for the new dataframe.  Function 
    returns the completed DataFrame."""
    
   # define column names
    colNames = ['agency_cd', 'site_no', 'Date', 'Discharge', 'Quality']

    # open and read the file
    DataDF = pd.read_csv(fileName, header=1, names=colNames,  
                         delimiter=r"\s+",parse_dates=[2], comment='#',
                         na_values=['Eqp'])
    DataDF = DataDF.set_index('Date')
    
    #checking for negative streamflow
    DataDF.loc[~(DataDF['Discharge'] > 0), 'Discharge'] = np.nan
    
    # quantify the number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()
    
    #removing NAs
    DataDF= DataDF.dropna() 
    
    return( DataDF, MissingValues )
    

def ClipData( DataDF, startDate, endDate ):
    """This function clips the given time series dataframe to a given range 
    of dates. Function returns the clipped dataframe and and the number of 
    missing values."""
    
    #Subsetting for dates required
    DataDF=DataDF.loc[startDate:endDate]
    
    #Removing NAs
    MissingValues = DataDF["Discharge"].isna().sum()
    
    return( DataDF, MissingValues )


def ReadMetrics( fileName ):
    """This function reads the contents of both the Annual and Monthly metric CSV files into separate dataframes."""
    
    # open and read the file
    DataDF = pd.read_csv(fileName, header=0, delimiter=',', parse_dates=['Date'])
    DataDF = DataDF.set_index('Date') # Set Date as index 
    
    return( DataDF )

# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

if __name__ == '__main__':

    # define full river names as a dictionary so that abbreviations are not used in figures
    riverName = { "Wildcat": "Wildcat Creek",
                  "Tippe": "Tippecanoe River" }
    
    # define filenames as a dictionary
    fileName = { "Wildcat": "WildcatCreek_Discharge_03335000_19540601-20200315.txt",
                 "Tippe": "TippecanoeRiver_Discharge_03331500_19431001-20200315.txt" }
    
    # define filenames for annual and monthly metrics
    metsf = { "Annual": "Annual_Metrics.csv",
                 "Monthly": "Monthly_Metrics.csv" }
    
    # define colors for plots
    color = {"Wildcat":'black',
             "Tippe":'r'}
    
    # define blank dictionaries (these will use the same keys as fileName)
    DataDF = {}
    MissingValues = {}
    
    # Read and plot daily flow
    for file in fileName.keys():
        # Read data
        DataDF[file], MissingValues[file] = ReadData(fileName[file])
        # clip to consistent period (last 5 years)
        DataDF[file], MissingValues[file] = ClipData( DataDF[file], '2014-10-01', '2019-09-30' )
        # plot data    
        plt.plot(DataDF[file]['Discharge'], label=riverName[file], color=color[file])
    plt.legend(loc='upper right')
    plt.title('Daily Flow')
    plt.ylabel('Discharge (cfs)')
    plt.xlabel('Time')
    plt.rcParams.update({'font.size':12})
    plt.grid(which='both') 
    plt.savefig('Daily_flow.png', dpi=96, bbox_inches='tight') 
    plt.show()
       
    # Create annual dataframe    
    DataDF['Annual'] = ReadMetrics(metsf['Annual'])    
    #Groupby data by the station name
    Annual_Wildcat = DataDF['Annual'].groupby('Station')
    
        #Plot annual coefficient variation
    for name, data in Annual_Wildcat:
        plt.scatter(data.index.values, data['Coeff Var'].values, label=riverName[name], color=color[name], marker= 'x')
    plt.legend()
    plt.title('Coefficient of Variation')
    plt.ylabel('Coefficient of Vaviation')
    plt.xlabel('Time')
    ax=plt.gca()
    ax.set_xticklabels(np.arange(1969, 2019, 10))
    plt.rcParams.update({'font.size':12})
    plt.grid(which='both') 
    plt.savefig('CoeffVar.png', dpi=96)

    
    #Plot annual TQmean
    for name, data in Annual_Wildcat:
        plt.plot(data.index.values, data['Tqmean'].values, label=riverName[name], color=color[name])
    plt.legend()
    plt.title('TQmean')
    plt.ylabel('TQmean')
    plt.xlabel('Time')
    ax=plt.gca()
    ax.set_xticklabels(np.arange(1969, 2019, 10))
    plt.rcParams.update({'font.size':12})
    plt.grid(which='both') 
    plt.savefig('TQmean.png', dpi=96)

      
    #Plot annual R-B Index
    for name, data in Annual_Wildcat:
        plt.plot(data.index.values, data['R-B Index'].values, label=riverName[name], color=color[name])
    plt.legend()
    plt.title('R-B Index')
    plt.ylabel('R-B Index')
    plt.xlabel('Time')
    ax=plt.gca()
    ax.set_xticklabels(np.arange(1969, 2019, 10))
    plt.grid(which='both') 
    plt.rcParams.update({'font.size':12})
    plt.savefig('RB_Index.png', dpi=96)

   
    # Reading monthly dataframe    
    DataDF['Monthly'] = ReadMetrics(metsf['Monthly'])
    # Groupby the monthly data based on different station
    MoDataDF = DataDF['Monthly'].groupby('Station')
    # Calculate annual monthly average values
    for name, data in MoDataDF:
        cols=['Mean Flow']
        m=[3,4,5,6,7,8,9,10,11,0,1,2]
        index=0
        # Create a new dataframe
        MonthlyAverages=pd.DataFrame(0,index=range(1,13),columns=cols)
        # Create the output table
        for i in range(12):
            MonthlyAverages.iloc[index,0]=data['Mean Flow'][m[index]::12].mean()
            index+=1
        plt.plot(MonthlyAverages.index.values, MonthlyAverages['Mean Flow'].values, label=riverName[name], color=color[name])    
    plt.legend()
    plt.title('Average Annual Monthly Flow')
    plt.ylabel('Discharge (cfs)')
    plt.xlabel('Month')
    plt.grid(which='both') 
    plt.rcParams.update({'font.size':12})
    plt.savefig('Annual_mo.png', dpi=96)

    
    #Return Period of annual peak flow
    DataDF['Annual'] = DataDF['Annual'].drop(columns=['site_no','Mean Flow','Tqmean','Median Flow','Coeff Var','Skew','R-B Index','7Q','3xMedian'])
    Annual_peak = DataDF['Annual'].groupby('Station')
    # Plot return period 
    for name, data in Annual_peak:
        # Sort the data in descnding order
        flow = data.sort_values('Peak Flow', ascending=False)
        # Check the data
        print(flow.head())
        # Calculate the rank and reversing the rank to give the largest discharge rank of 1
        ranks1 = scs.rankdata(flow['Peak Flow'], method='average')
        ranks2 = ranks1[::-1]
        
        # Calculate the exceedance probability
        exceed_prob = [100*(ranks2[i]/(len(flow)+1)) for i in range(len(flow))]
        # Plot the exceedance probability
        plt.scatter(exceed_prob, flow['Peak Flow'],label=riverName[name], color=color[name], marker='.')
    plt.grid(which='both') 
    plt.legend()
    plt.title('Return Period of Annual Peak Flow Events')
    plt.ylabel('Peak Discharge (cfs)')
    plt.xlabel('Exceedance Probability (%)')
    plt.xticks(range(0,100,5))
    plt.rcParams.update({'font.size':12}) 
    plt.savefig('Exceed_prob.png', dpi=96, bbox_inches='tight')
    
 