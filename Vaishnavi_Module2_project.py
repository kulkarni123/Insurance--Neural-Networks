# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 11:41:23 2022

@author: Vaishu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.offline import plot
#%%
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler,LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
#%%
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn  import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError
#%%
# Reading the data and exploring the columns
df=pd.read_csv(r"C:\Users\Vaishu\Desktop\Work\Neural_Network\US_Insurance.csv")
print(df.head())
df.info()
df.describe()
df.age.describe()#distribution of age 
df.columns
#%% Comparing charges with age


#sns.histplot(data=df, x="age",  binwidth=1)
plt.figure(figsize=(13,7))
sns.barplot( x = 'age', y = 'charges', data = df)
plt.show()
'''
There is a clear indication that with increase in  age the charges  increase

'''
#%%Comparision of number of customers with age 
fig = px.histogram(df, x='age', marginal='box', nbins=47, title='Distribution of Age')
fig.update_layout(bargap=0.1)
plot(fig)
'''
The distribution of ages in the dataset is almost uniform, with 20-30 customers 
at every age, but for the ages 18 and 19 there are twice number of customers
Why there is increase in the number of customers in the age group 18 to 19??

'''
#%% Comparision of number of customers with bmi 
fig = px.histogram(df, x='bmi', marginal='box', nbins=47,color_discrete_sequence=['green'] ,title='Distribution of BMI')
fig.update_layout(bargap=0.1)
plot(fig)
'''
The distribution of BMIs forms a gaussian distribution unlike the distribution of age 
It means most of the customers are having normal weightBMI or slightly overweight 
(i.e, range from 18 to 30 )but there few outliers as well.

'''
#%% Comparing charges according to gender
ax = sns.barplot(x="sex", y="charges", hue="sex", data=df)
plt.legend(loc="center")
'''
The graph clearly indicates that the gender of the customer affects slightly in the charges

'''
#%% Comparision of charges with respect to smoker and non smoker with gender
g = sns.catplot(x="sex", y="charges", hue="smoker", data=df, kind="bar",height=4, aspect=.7)
'''
There is a significant difference in medical expenses between smokers and non-smokers. 
Though the female smokers expenses is less compared to male smokers.
This inturn indicates the strong correlation between the charges incured
by smokers and non smoker.Note that the charges for most customers are below 10,000$

'''
#%%
sns.scatterplot(data=df, x="age", y="charges", hue="smoker")

'''
There is significant variation at every age, and it's clear that age alone cannot 
be used to accurately determine medical charges.
We also observe three clusters  the first one shows non-smokers who have lower medical charges.
The second shows mix of both smokers and non smokers which have a bit of high medical charges 
The third one show completly smokers who have higher medical charges . 
So the assumption would be that people who are non smokers and have less health isssues 
have lesser charges than the smokers and non smoker with healthe issues. 
'''
#%% viaualization of  charges and BMI
sns.scatterplot(data=df, x="bmi", y="charges",hue="smoker" )
'''
It seems that the smokers with lower BMI have less medical expenses compared to 
smokers with higher BMI
'''
#%%Comparing the charges with number of children
ax = sns.barplot(x="children", y="charges", data=df)
#%%
#Comparing the charges with region
ax = sns.barplot(x="region", y="charges", data=df)
'''
There isn't much difference in the charges with respect to number of children  unless
the number of childeren is greater than 4
Also region does not have much difference on the charges
The southeast region have slightly higher charges compared to other regions
'''
#%%
'''
With the above Anslysis we can say that the values in some columns are more closely 
related to charges  compared to others .Lets move on to  creating a Regression model. 
For this we need to convert all columns to numericall
'''


#%% Converting categorical column 'sex' and 'smoker' to numerical using Label Encoder

cols = ['sex', 'smoker']
df[cols] = df[cols].apply(LabelEncoder().fit_transform)
df.head()
df[cols]
df.columns

#%%
#Correlation Matrix
df.corr()
sns.heatmap(df.corr(), cmap='Reds', annot=True)
plt.title('Correlation Matrix');
#%% converting 'region' categorical column to numeric using One Hot Encoder

enc = preprocessing.OneHotEncoder()
enc.fit(df[['region']])
one_hot = enc.transform(df[['region']]).toarray()
one_hot
df[['northeast', 'northwest', 'southeast', 'southwest']] = one_hot
df.head()
df.shape
df.columns
num_df=df.drop(labels=['region'],axis=1)
num_df.columns

#%%
#Splitting the data into train and  test then scaling the train data.

inputs = ['age', 'bmi', 'children', 'smoker', 'sex', 'northeast', 'northwest', 'southeast', 'southwest']
target =['charges']

X,y=num_df[inputs],num_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state=1)

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
print(X_train_scaled)
X_test_scaled=scaler.transform(X_test)


#%%
#creating keras squential model
model = Sequential()
model.add(Dense(16, input_dim=9,  activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

#Compliling the model for learning process
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
callback=EarlyStopping(monitor="mse",patience=10)
history = model.fit(X_train_scaled, y_train, epochs=2000, batch_size=100,callbacks=[callback],verbose=1)

#%%Plotting the loss
plt.plot(history.history["mse"])
plt.show()
#%%
#Calculating the Error
y_pred=model.predict(X_test_scaled)
y_pred
error=np.sqrt((y_pred-y_test)**2)
error.mean()
#%%








