from numpy import NaN
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

import numpy as np
import pandas as pd

train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

print(train_data.head()) 
#Cleaning the data
unused_atributes = ['Name']

train_data = train_data.drop("Name",axis=1)
test_data = test_data.drop('Name',axis=1)

#Function for splithing the tiket data
def my_custom_split(ticket):
    separator_index = ticket.rfind(' ')

    if(separator_index == -1):
        if(type(ticket) == str):
            ticket = int(0)
        return ticket,NaN

    return ticket[separator_index+1:],ticket[:separator_index].replace('.','').replace(' ','').upper()

def separateTicket(row):
    row.Ticket_number, row.Ticket_string = my_custom_split(row.Ticket)
    return row

#if we have the ticker 443243 AB-por2
# we are going to separate 443242 in ticket number
# and AB-por2 in the ticket string
#INITIALIZE TICKET SERIES
train_data['Ticket_number'] = 0
train_data['Ticket_string'] = ''
train_data = train_data.apply(separateTicket,axis='columns')
# train_data.loc[train_data.Ticket_number == 'LINE','Ticket_number'] = 0
#OPTIONAL CHANGE THE DATATYPE
# train_data.SibSp = train_data.SibSp.astype(np.uint8)
train_data.Ticket_number = train_data.Ticket_number.astype(int)

cat_pipeline = Pipeline([
        ('impute',SimpleImputer(strategy='constant',fill_value='')),
        ('encode',OneHotEncoder(handle_unknown='ignore'))])

num_pipeline = Pipeline([
        ('impute',SimpleImputer(strategy='mean')),
        ('standardize',StandardScaler())
        ])


num_attribs = ['PassengerId','Pclass','Age','SibSp','Parch','Fare','Ticket_number']
cat_atribs = ['Sex','Embarked','Cabin','Ticket_string']

preprocessing_pipeline  = ColumnTransformer([
    ('num',num_pipeline,num_attribs),
    ('cat',cat_pipeline,cat_atribs)
    ])

processed_train_data = preprocessing_pipeline.fit_transform(train_data)
print(processed_train_data[:5])
#just another name
label = 'Survived'
X = processed_train_data
y = train_data[label]

train_X,val_X, train_y,val_y = train_test_split(X,y,test_size=0.2)
model = RandomForestRegressor(random_state=42)
model.fit(train_X,train_y)

predictions = model.predict(val_X)

total_asserts = 0
for i in range(len(predictions)):
    print('predigo: ',predictions[i],'redondeo y da',round(predictions[i]), 'pero el resultado',val_y.iloc[i])
    predictions[i]  = round(predictions[i])
    if(predictions[i] == val_y.iloc[i]):
        total_asserts+=1
print('hola mis predicciones fueron',total_asserts/len(predictions))
# print(mean_squared_error(predictions,val_y,squared=True))


#using cross validation
from sklearn.model_selection import cross_val_score

random_forest_cross_val_scores = -cross_val_score(model, processed_train_data, train_data['Survived'],scoring='neg_root_mean_squared_error',cv=12)
print(random_forest_cross_val_scores)
print(pd.Series(random_forest_cross_val_scores).describe())

#TEST SET
test_data['Ticket_number'] = 0
test_data['Ticket_string'] = ''
test_data = test_data.apply(separateTicket,axis='columns')

test_data_processed = preprocessing_pipeline.fit_transform(test_data)


print(test_data_processed)
