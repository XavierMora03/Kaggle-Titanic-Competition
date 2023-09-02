from numpy import NaN
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

import numpy as np
import pandas as pd

train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

#Cleaning the data
unused_atributes = ['Name']

train_data = train_data.drop("Name",axis=1)
test_data = test_data.drop('Name',axis=1)

#Function for splithing the tiket data
def split_ticket_string(ticket):
    separator_index = ticket.rfind(' ')

    if(separator_index == -1):
        if(type(ticket) == str):
            ticket = int(0)
        return ticket,NaN

    return ticket[separator_index+1:],ticket[:separator_index].replace('.','').replace(' ','').upper()

def assing_ticket_string(row):
    row.Ticket_number, row.Ticket_string = split_ticket_string(row.Ticket)
    return row

#if we have the ticker 443243 AB-por2
# we are going to separate 443242 in ticket number
# and AB-por2 in the ticket string

def normalizeTicket(df):
    df['Ticket_number'] = 0
    df['Ticket_string'] = ''
    
    df = df.apply(assing_ticket_string, axis='columns')

    df.Ticket_number = df.Ticket_number.astype(int)
    df = df.drop('Ticket',axis=1)
    
    df.SibSp = df.SibSp.astype(np.uint8)
    return df

def normalizeTicketNamesOut(function_transformer, feature_names_in):
    return ['PassengerId', 'Survived' ,'Pclass' ,'Sex' ,'Age,' 'SibSp,' 'Parch' ,'Fare', 'Cabin', 'Embarked','Ticket_number','Ticket_string']

ticket_transformer = FunctionTransformer(normalizeTicket,feature_names_out=normalizeTicketNamesOut)

train_data = ticket_transformer.fit_transform(train_data)
print(train_data)
print(ticket_transformer.get_feature_names_out())
cat_pipeline = Pipeline([
        ('impute',SimpleImputer(strategy='constant',fill_value='')),
        ('encode',OneHotEncoder(handle_unknown='ignore'))])

num_pipeline = Pipeline([
        ('impute',SimpleImputer(strategy='mean')),
        ('standardize',StandardScaler())
        ])

num_attribs = ['PassengerId','Pclass','Age','SibSp','Parch','Fare','Ticket_number']
print(num_attribs)
cat_atribs = ['Sex','Embarked','Cabin','Ticket_string']

preprocessing_pipeline  = ColumnTransformer([
    ('num',num_pipeline,num_attribs),
    ('cat',cat_pipeline,cat_atribs)
    ])



processed_train_data = preprocessing_pipeline.fit_transform(train_data)
#just another name
label = 'Survived'
X = processed_train_data
y = train_data[label]

train_X,val_X, train_y,val_y = train_test_split(X,y,test_size=0.2)
model = RandomForestRegressor(random_state=42)
model.fit(train_X,train_y)

predictions = model.predict(val_X)

#using cross validation
from sklearn.model_selection import cross_val_score

random_forest_cross_val_scores = -cross_val_score(model, processed_train_data, train_data['Survived'],scoring='neg_root_mean_squared_error',cv=12)
print(pd.Series(random_forest_cross_val_scores).describe())

#TEST SET
test_data['Ticket_number'] = 0
test_data['Ticket_string'] = ''

test_data_processed = preprocessing_pipeline.fit_transform(test_data)


