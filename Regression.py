import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression



df = quandl.get("CHRIS/MGEX_IH1", authtoken="Ns1FcMzkyM-kGRMbixt2")

#print(df.head)

df = df[['Open', 'High', 'Low', 'Last', 'Volume']]
df['Volatility_PCT'] = (df['High']-df['Low'])/df['Low']*100.0
df['Inc_Dec_PCT'] = (df['Last']-df['Open'])/df['Open']*100.0

df = df[['Last', 'Volatility_PCT', 'Inc_Dec_PCT', 'Volume']]

forecast_col = 'Last'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df))) # go back 10% of total number of days for the prediction
'''
Here we use the pandas shift method to shift the forecast_out label that is the label will have 'Last' values 10 days 
into the future.
'''

df['Labels'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['Labels'], 1))
y = np.array(df['Labels'])
X = preprocessing.scale(X)      #normalising x
y = preprocessing.scale(y)      #normalising y

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)
clf1 = LinearRegression()
clf1.fit(X_train, y_train)
accuracy1 = clf1.score(X_test, y_test)

clf2 = svm.SVR(kernel="linear")
clf2.fit(X_train, y_train)
accuracy2 = clf2.score(X_test, y_test)


print(accuracy1, accuracy2)