from numpy import NaN
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV, RandomizedSearchCV
import numpy as np
import pandas as pd
from scipy.stats import randint
print(randint(low=1,high=3))
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')
# quit()
#Cleaning the data
unused_atributes = ['Name']

# train_data = train_data.drop("Name",axis=1)
# test_data = test_data.drop('Name',axis=1)

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

cat_pipeline = Pipeline([
        ('impute',SimpleImputer(strategy='constant',fill_value='')),
        ('encode',OneHotEncoder(handle_unknown='ignore'))])

num_pipeline = Pipeline([
        ('impute',SimpleImputer(strategy='mean')),
        ('standardize',StandardScaler())
        ])

num_attribs = ['PassengerId','Pclass','Age','SibSp','Parch','Fare','Ticket_number']
cat_atribs = ['Sex','Embarked','Cabin','Ticket_string']

preprocessing_pipeline_partial  = ColumnTransformer([
    ('num',num_pipeline,num_attribs),
    ('cat',cat_pipeline,cat_atribs)
    ])

preprocessing_pipeline = Pipeline([
    ('ticket_separator',ticket_transformer),
    ('general_pipeline',preprocessing_pipeline_partial)
    ])

label = 'Survived'
'''

train_data = ticket_transformer.fit_transform(train_data)
processed_train_data = preprocessing_pipeline.fit_transform(train_data)
#just another name
X = processed_train_data
y = train_data[label]

train_X,val_X, train_y,val_y = train_test_split(X,y,test_size=0.2)
model = RandomForestRegressor(random_state=42)
model.fit(train_X,train_y)

predictions = model.predict(val_X)

#using cross validation
from sklearn.model_selection import cross_val_score

# random_forest_cross_val_scores = -cross_val_score(model, X, y,scoring='neg_root_mean_squared_error',cv=12)
# print(pd.Series(random_forest_cross_val_scores).describe())

'''
#GRID SEARCH

pipeline_with_forest_model = Pipeline([
    ('pipeline',preprocessing_pipeline),
    ('random_forest',RandomForestRegressor(random_state=42))
    ])

parameters_grid = [
        {'random_forest__n_estimators':[50,100,200,300],
        'random_forest__max_features':[5,8,20,25,40]}
        ]

n__estimator = range(20,250)
parameters_grid = {'random_forest__n_estimators':randint(low=20,high=250),
                    'random_forest__max_features':randint(low=2, high=30)}
# parameters_grid = {'random_forest__n_estimators':range(20,250),
#                     'random_forest__max_features':range(2, 30)}
#

# print(train_data,train_data[label])
random_search = RandomizedSearchCV(pipeline_with_forest_model, param_distributions=parameters_grid,n_iter=300,cv=3,scoring='neg_root_mean_squared_error',random_state=42)
random_search.fit(train_data,train_data[label])
with open('resultsRandomizedSearch.txt','w') as file:
    file.write('\nBEST_PARAMS\n')
    file.write(str(random_search.best_params_))
    file.write('\nBEST_SCORE\n')
    file.write(str(random_search.best_score_))
    file.write('\nBEST_INDEX\n')
    file.write(str(random_search.best_index_))

# print(tnd)
        
        
quit()
grid_search = GridSearchCV(pipeline_with_forest_model,parameters_grid,cv=3, scoring = 'neg_root_mean_squared_error')

grid_search.fit(train_data,train_data[label])
print(grid_search.best_params_)
print(grid_search.best_estimator_)

cv_results  = pd.DataFrame(grid_search.cv_results_)
cv_results.sort_values(by='mean_test_score',ascending=False,inplace=True)
print(cv_results.head())
#TEST SET
test_data_processed = preprocessing_pipeline.fit_transform(test_data)
