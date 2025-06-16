# Final Project
import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import FunctionTransformer





import os
import re
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# https://pypi.org/project/wbgapi/
import wbgapi as wb

# Exploring the data:

def getdata(searchterms):
    terms = searchterms
    termlist = [] 
    for i in terms:
        rows = pd.DataFrame(wb.series.Series(q=f'{i}')).reset_index()
        rows.columns = ['ID','Desc']
        id_list = rows['ID'].to_list()
        termlist = termlist + id_list
    df = wb.data.DataFrame(termlist, 'USA', mrv=20)
    df = pd.DataFrame(df).reset_index()
    dfmelted = pd.melt(df, id_vars=['series'], var_name='Date', value_name='Value')
    dfpivoted = dfmelted.pivot(index='Date', columns='series', values='Value').reset_index()
    dfpivoted['Date'] = dfpivoted['Date'].str.strip('YR')
    dfpivoted['Year'] = pd.to_datetime(dfpivoted['Date']).dt.year
    dfpivoted = dfpivoted.dropna(thresh=3)
    return dfpivoted

def columnlookup(searchterms, dir):
    terms = searchterms
    df = None
    for i in terms:
        rows = pd.DataFrame(wb.series.Series(q=f'{i}')).reset_index()
        rows.columns = ['ID', 'Desc']
        if df is None:
            df = rows
        else:
            df = pd.merge(df, rows, how='outer')
    dir = dir
    file = os.path.join(dir, 'description.csv')
    if os.path.exists(file):
        os.remove(file)
        df.to_csv(file, index=False)
        return df 
    else:
        df.to_csv(file, index=False)
        return df

# Graphing
def labels(str, searchterms, dir):
    lookup = columnlookup(searchterms, dir)
    label = lookup.loc[lookup['ID'] == str, 'Desc'].iloc[0]
    return label
    

def varovertime(str, searchterms, dir, saveterm):
    df = getdata(searchterms)
    sns.set_theme('paper')
    sns.barplot(data=df, x='Year', y=str)
    # xlabel = lookup.query(f'ID == {str}')
    # ylabel = pd.DataFrame(lookup[lookup['ID']==str]['Desc']).iloc[0]
    ylabel = labels(str, searchterms, dir)
    # ylabel = ylabel.to_string()
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.savefig(saveterm, bbox_inches ='tight')
    plt.show()

# Fossil fuels, inflation, GDP per capita, energy imports 
def lines(strs, searchterms, dir):
    df = getdata(searchterms)
    sns.set_theme('paper')
    fig, axes = plt.subplots(2, 2, figsize=(12,8))
    axes = axes.flatten()
    colors = ['steelblue', 'darkorange', 'seagreen', 'crimson']

    for i in range(4):
        sns.lineplot(data=df, x='Year', y=strs[i], ax=axes[i], color = colors[i])
        # titlematch = re.sub(r"^[^(]+", '', strs[i])
        # axes[i].set_title(f"{titlematch} Over Time")
        ylabel = labels(strs[i], searchterms, dir)
        plt.ylabel(ylabel)
        axes[i].set_ylabel(ylabel)
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('fourplots.png', bbox_inches = 'tight')
    plt.show()

# Emissions plots: 
def emissions(searchterms):
    df = getdata(searchterms)
# No2 total: 'EN.GHG.N2O.MT.CE.AR5'
# CH4 total: 'EN.GHG.CH4.MT.CE.AR5'
# CO2 total: 'EN.GHG.CO2.MT.CE.AR5'
    sns.set_theme('paper')
    plt.figure()
    sns.lineplot(data = df, x='Year',y='EN.GHG.N2O.MT.CE.AR5', label = 'NO2', color='seagreen')
    sns.lineplot(data = df, x='Year', y= 'EN.GHG.CH4.MT.CE.AR5', label= 'CH4', color='darkorange')
    sns.lineplot(data=df, x='Year', y='EN.GHG.CO2.MT.CE.AR5', label='CO2', color='steelblue')
    plt.title('Total Emissions by Type')
    plt.ylabel('Million Tonnes of CO2 Equivalent')
    plt.savefig('emissiontypes.png',bbox_inches='tight')
    plt.show()


# Model 1:
def randomforest(searchterms):
    df = getdata(searchterms)
    df = df.apply(pd.to_numeric, errors='coerce')
    X = df.drop(columns='EG.FEC.RNEW.ZS')
    y = df['EG.FEC.RNEW.ZS']
    
    y_filled = y.fillna(y.mean())

    X_train, X_test, Y_train, Y_test =  train_test_split(X, y_filled, test_size=0.3, random_state=42)
    pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    # ('scaler', StandardScaler()),  # Optional
    ('model', RandomForestRegressor(n_estimators=50, random_state=42))])
    pipeline.fit(X_train, Y_train)
    ypred = pipeline.predict(X_test)
    trainpred = pipeline.predict(X_train)
    testacc = r2_score(Y_test, ypred)
    trainacc = r2_score(Y_train, trainpred)
    
    # graph:
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=Y_test, y=ypred, alpha=0.6)
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], '--r')  # identity line
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs. Predicted')
    plt.tight_layout()
    plt.savefig('model1.png', bbox_inches='tight')
    plt.show()


    return f"Training accuracy: {trainacc}. \nTesting accuracy: {testacc}."

# Model 2:
def append_clusters(X):
    clusters = KMeans(n_clusters=6, n_init=10, random_state=42).fit_predict(X)
    return np.column_stack((X, clusters))

    
def secondmodel(searchterms, dir):
    df = getdata(searchterms)
    df = df.apply(pd.to_numeric, errors='coerce')
    lookup = columnlookup(searchterms, dir)
    contains_investment = lookup['Desc'].str.contains('investment', case=False)
    investmentcols = df[contains_investment]

    df[[f"{col}_Lag" for col in investmentcols]] = df[investmentcols].shift(1)

    droplist = ['NY.ADJ.DNGY.CD',
    'NY.ADJ.DNGY.GN.ZS',
    'EN.ATM.PM25.MC.T1.ZS',
    'EN.ATM.PM25.MC.T2.ZS',
    'EN.ATM.PM25.MC.T3.ZS',
    'EN.ATM.PM25.MC.ZS',
    'EN.POP.EL5M.RU.ZS',
    'EN.POP.EL5M.UR.ZS',
    'EN.POP.EL5M.ZS',
    'EN.URB.MCTY',
    'EN.URB.MCTY.TL.ZS',
    'SH.CON.1524.FE.ZS',
    'SH.CON.1524.MA.ZS',
    'SH.DYN.AIDS.FE.ZS',
    'SH.UHC.FBP1.ZS',
    'SH.UHC.FBP2.ZS',
    'SH.UHC.FBPR.ZS',
    'SH.UHC.NOP1.ZS',
    'SH.UHC.NOP2.ZS',
    'SH.UHC.NOPR.ZS',
    'SH.UHC.OOPC.10.ZS',
    'SH.UHC.OOPC.25.ZS',
    'SH.UHC.TOT1.ZS',
    'SH.UHC.TOT2.ZS',
    'SH.UHC.TOTR.ZS',
    'SI.SPR.PC40',
    'SI.SPR.PC40.ZG',
    'SI.SPR.PCAP',
    'SL.EMP.1524.SP.FE.NE.ZS',
    'SL.EMP.1524.SP.FE.ZS',
    'SL.EMP.1524.SP.MA.NE.ZS',
    'SL.EMP.1524.SP.MA.ZS',
    'SL.EMP.1524.SP.NE.ZS',
    'SL.EMP.1524.SP.ZS',
    'SL.EMP.TOTL.SP.FE.ZS',
    'SL.EMP.TOTL.SP.MA.ZS',
    'SL.EMP.TOTL.SP.NE.ZS',
    'SL.EMP.TOTL.SP.ZS',
    'SL.TLF.ACTI.FE.ZS',
    'SL.TLF.ACTI.MA.ZS',
    'SL.TLF.ACTI.ZS',
    'SL.TLF.CACT.MA.ZS',
    'SL.TLF.CACT.NE.ZS',
    'SL.TLF.CACT.ZS',
    'IC.BRE.BI.P2',
    'IC.BRE.FS.P2',
    'IQ.SPI.PIL5',
    ]

    dftrim = df.drop(columns=droplist)

    dftrim = df

    numeric = dftrim.columns
    X = dftrim.drop(columns='EG.FEC.RNEW.ZS')
    y = dftrim['EG.FEC.RNEW.ZS']
    y_filled = y.fillna(y.mean())

    X_train, X_test, Y_train, Y_test =  train_test_split(X, y_filled, test_size=0.4, random_state=42)
    os.environ["OMP_NUM_THREADS"] = "1"

    pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='mean')),
    ('cluster_feature', FunctionTransformer(append_clusters, validate=False)),
    ('model', RandomForestRegressor(n_estimators=70, random_state=42))])
    pipeline.fit(X_train, Y_train)
    ypred = pipeline.predict(X_test)
    trainpred = pipeline.predict(X_train)

    testacc = r2_score(Y_test, ypred)
    trainacc = r2_score(Y_train, trainpred)

    # Graph:
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=Y_test, y=ypred, alpha=0.6)
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], '--r')  # identity line
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs. Predicted')
    plt.tight_layout()
    plt.savefig('kmeans.png', bbox_inches='tight')
    plt.show()

    return f"Training accuracy: {trainacc}. \nTesting accuracy: {testacc}."






if __name__ == '__main__':
    searchterms = ['energy',
                   'population',
                   'Education',
                   'emissions',
                   'interest',
                   'Technology',
                   'infrastructure',
                   'transport',
                   'internet',
                   'income share',
                   'investment in',
                   'Control of Corruption',
                   'Regulatory Quality:',
                   'FP.CPI.TOTL.ZG',
                   'GDP'
                   ]
    
    directory = r"C:\Users\griff\OneDrive\718\Project04\Project04"

    # print(getdata(searchterms))

    # print(columnlookup(searchterms, directory))

    # Renewable energy consumption:
    # print(varovertime('EG.FEC.RNEW.ZS', searchterms, directory, 'renewableconsumption.png'))
    # Energy use per capita:
    # print(varovertime('EG.USE.PCAP.KG.OE', searchterms, directory, 'totalconsumption.png'))
    # fossil fuels, inflation, GDP per capita, and energy imports:
    linelist = ['EG.USE.COMM.FO.ZS','NY.GDP.DEFL.KD.ZG','NY.GDP.PCAP.CD', 'EG.IMP.CONS.ZS']
    # print(lines(linelist, searchterms, directory))
    # print((emissions(searchterms)))
    # print(randomforest(searchterms))
    print(secondmodel(searchterms, directory))

    

