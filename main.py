from numpy import NaN
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import numpy as np
import pandas as pd

 
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

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


train_data.info()
from sklearn.compose import ColumnTransformer


num_attribs = ['PassengerId','Survived','Pclass','Age','SibSp','Parch','Fare','Ticket_number']
cat_atribs = ['Sex','Embarked','Cabin','Ticket_string']

'''
#funcion para ver cada categoria

print(train_data.head())
print(train_data['Ticket_string'])
output_cat_Ticket_string = cat_pipeline.fit_transform(train_data[['Ticket_string']])

for i in range(len(cat_pipeline['encode'].categories_[0])):
    print('I ES IGUAL A',i)
    print(output_cat_Ticket_string[:,i])
    print(cat_pipeline['encode'].categories_[0][i])
quit()
'''

print(train_data.head())
preprocessing_pipeline  = ColumnTransformer([
    ('num',num_pipeline,num_attribs),
    ('cat',cat_pipeline,cat_atribs)
    ])

processed_train_data = preprocessing_pipeline.fit_transform(train_data)
print(processed_train_data)
# print(preprocessing_pipeline['cat'])
# print(processed_train_data[0,range(0,10)])
quit()
proceced_train_data_Data_Frame = pd.DataFrame(proceced_train_data,columns=preprocessing_pipeline.get_feature_names_out())#,index=train_data.index)
print(proceced_train_data)
quit()
#just another name
label = 'Survived'
X = train_data.loc[:,train_data.columns != label]
y = train_data[label]



train_X,val_X, train_y,val_y = train_test_split(X,y,test_size=0.2)
model = RandomForestRegressor(random_state=42)
model.fit(train_X,train_y)


predictions = model.predict(val_X)

print(mean_squared_error(predictions,val_y,squared=False))

passengers  = train_data

