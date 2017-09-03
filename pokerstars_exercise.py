# -*- coding: utf-8 -*-
"""
Created on Tue May 24 18:26:03 2017

@author: jozh
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import pylab 
import scipy.stats as stats
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std

#import the raw data
Dailyplay = pd.read_csv('Dailyplay.csv')
Promotion = pd.read_csv('Promotions.csv')

df_dailyplay = pd.DataFrame(Dailyplay)
df_promotion = pd.DataFrame(Promotion)

result = pd.merge(df_dailyplay, df_promotion, how='left', on='Date')
result.head(20) #check the merged result

#to aggregate the data on daily basis
df0 = result.replace(np.nan, '', regex=True)  #need to replace nan with a string, otherwise nan is excluded in the groupby function
df_revenue = df0.groupby(['Date', 'Promo'])['Revenue'].sum().reset_index()
df_players = df0.groupby(['Date', 'Promo'])['Playerid'].count().reset_index()
df = pd.merge(df_revenue, df_players, how='left', on='Date')

#rename the column
df.columns = ([u'Date', u'Promo', u'Revenue', u'Promo_y', u'Players'])
df['my_dates'] = pd.to_datetime(df['Date'])
df['Day'] = df['my_dates'].dt.dayofweek
#map dayofweek to Strings  
days = {0:'Mon',1:'Tues',2:'Weds',3:'Thurs',4:'Fri',5:'Sat',6:'Sun'}
df['day_of_week'] = df['Day'].apply(lambda x: days[x])
df1 = df.drop(['my_dates', 'Day', 'Date', 'Promo_y'], axis=1) #delete interim variables

df2 = pd.get_dummies(df1)
df2.columns
df_final = df2.drop(['Promo_', 'day_of_week_Mon'], axis=1) #avoid the dummy trap!

#df_final contains Index([          u'Revenue',           u'Players',           u'Promo_A',
#                 u'Promo_B',   u'day_of_week_Fri',   u'day_of_week_Sat',
#         u'day_of_week_Sun', u'day_of_week_Thurs',  u'day_of_week_Tues',
#        u'day_of_week_Weds'],
#      dtype='object') - so we are ready for the modelling stage
print(df_final.shape) #(182, 10) - it only has 182 obs so split by training/ testing set may not be a good idea

X = df_final.drop(['Revenue'], axis = 1)     
y = df_final['Revenue'] 
    
##Modellng
#To check the corr matrix for the predictors
corrmat = X.corr()
sns.heatmap(corrmat, vmax=.8, square=True)

regr = LinearRegression() # use simple linear regression
regr.fit(X, y)
print(regr.coef_)
r_squared = regr.score(X,y)  #0.864
residues = y - regr.predict(X)  #get residuals
sns.distplot(residues)
#QQ plot for the normal distribution

stats.probplot(residues, dist="norm", plot=pylab)
pylab.show()


#Generate feature importance                      
clf = RandomForestClassifier(n_estimators = 10)
clf.fit(X, y)
clf.score(X, y)
feat_labels = X.columns
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
    feat_labels[indices[f]],
    importances[indices[f]])) #as we need all these predictors, we're not going to conduct feature selection

#prepare out-of-sample data for scoring
topredict = pd.DataFrame({'Players':[3000,4000,4000,5000,6000,6000,7000], 'Promo_B':[0,0,0,1,0,1,0],'Promo_A':[0,1,0,0,1,0,0], \
'day_of_week_Sun':[0,0,0,0,0,0,1], 'day_of_week_Thurs':[0,0,0,1,0,0,0], 'day_of_week_Sat':[0,0,0,0,0,1,0], \
'day_of_week_Fri':[0,0,0,0,1,0,0], 'day_of_week_Weds':[0,0,1,0,0,0,0], 'day_of_week_Tues':[0,1,0,0,0,0,0]}) 
    #probably i should create the dayofweek and then use get_dummy
Rev_predicted = regr.predict(topredict)
                   
#it seems sklearn doesn't report 95% confidence interval for prediction...
#switching to stat.model

data = pd.concat([X, y], axis=1) #prepare data for statsmodels

lm = smf.ols(formula='Revenue ~ Players+ Promo_A+ Promo_B+ day_of_week_Fri + day_of_week_Sat+ day_of_week_Sun +\
       day_of_week_Thurs+ day_of_week_Tues+ day_of_week_Weds', data=data).fit()
print(lm.summary())

#seems there is a bug regarding out-of-sample prediction in statsmodels
#Found the below function on stackoverflow, not varified due to time constraint
def transform_exog_to_model(fit, exog):
    transform=True
    self=fit

    # The following is lifted straight from statsmodels.base.model.Results.predict()
    if transform and hasattr(self.model, 'formula') and exog is not None:
        from patsy import dmatrix
        exog = dmatrix(self.model.data.orig_exog.design_info.builder,
                       exog)

    if exog is not None:
        exog = np.asarray(exog)
        if exog.ndim == 1 and (self.model.exog.ndim == 1 or
                               self.model.exog.shape[1] == 1):
            exog = exog[:, None]
        exog = np.atleast_2d(exog)  # needed in count model shape[1]

    # end lifted code
    return exog

transformed_exog = transform_exog_to_model(lm, topredict)
print(transformed_exog)
sdev, lower, upper = wls_prediction_std(lm, exog=transformed_exog, alpha=0.05, weights=[1])
print(sdev, lower, upper)