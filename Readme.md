```python
# Understanding Workplace Turnover with HR Analytics

This ReadMe file seeks to uncover potential links to turnover.

## Installation

The following imports all packages for this project.

```python
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats import pearsonr
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from statsmodels.stats.mediation import Mediation
from statsmodels.stats.outliers_influence import variance_inflation_factor

logisticRegression = LogisticRegression()
```

## Data Preparation

The following is used to import and transform the dataset so it can be used for analysis.

```python
hr = pd.read_csv('turnover.csv') #Importing the csv
hr['department'] = hr['sales'] #Renaming the column to avoid confusion.
del hr['sales']
#Turning the salary column into categorical integers.
hr.loc[(hr.salary == 'low'),'salary']='1'
hr.loc[(hr.salary == 'medium'),'salary']='2'
hr.loc[(hr.salary == 'high'),'salary']='3'

sales = pd.get_dummies(hr['department'])
#Getting get_dummies of the sales column to create new columns, one for each department.

hr_data = pd.concat([hr, sales], axis=1) #Concatenating the sales and the hr dfs into a new df, hr_data.
del hr_data['department'] #deleting unneeded column
hr_data['salary'] = hr_data['salary'].astype('int64')
hr_data.groupby(['left']).mean()
#Using the groupby to see the difference in all columns by those who have stayed and left their company.

predictors = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company',
              'Work_accident', 'promotion_last_5years', 'sales', 'salary', 'IT', 'RandD', 'accounting', 'hr', 'management',
              'marketing', 'product_mng', 'support', 'technical'] #Creating a list of predictors.
for predictor in predictors:
    if predictor == 'sales':
            corr, _ = pearsonr(hr_data[predictor].astype('float'), hr_data['left'])
    print(predictor + ": ")
    corr, _ = pearsonr(hr_data[predictor], hr_data['left'])
    print(round(corr, 3))
#Correlation tests

#Creating two different dfs based on left status.
left0 = hr_data[hr_data['left']==0]
left1 = hr_data[hr_data['left']==1]
```

## Correlation Test

The following generated the correlation test.

```python
corr = hr_data.corr() #Seeing how all variables correlate with one another.

corr.style.background_gradient(cmap='coolwarm')
```

## Multicollinearity Test

The following checks the VIF levels for all variables.

```python
vif_data = hr_data.copy() #Creating a new df to look for multcollinearity.

del_column = []

for column in list(vif_data.columns[9:]):
    del_column.append(column)

for column in del_column:
    del vif_data[column]
    
vif_data.columns

del vif_data['left'] #Deleting the target column.

X = hr_data[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 'salary']] #Showing predictors
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
vif_data
```

## Plotting Variables Based on Target Variable Classification
```python
#Visualizing seeing all variables based on left status.

sns.set(rc = {'figure.figsize':(17,8)})

for predictor in predictors:
    print(predictor + ": ")
    sns.boxplot(data=hr_data, x='left', y=predictor)
    plt.show()
```

## Logistic - Feature Selection

```python
hr_tuning = hr_data.copy() #creating a new dataframe without messing with the original dataframe.
del hr_tuning['left'] #Removing the target variable from features.
for column in hr_tuning.columns[:-10]: #Removing dummy features unsuitable for feature selection
    hr_tuning[column] = (hr_tuning[column]-hr_tuning[column].mean())/hr_tuning[column].std()
del hr_tuning['management'] #For normalization.

#Creating a list and looping through to format.
hr_list = list(hr_tuning.columns)

for item in hr_list:
    item = 'hr_list["'+item+'"]'

#Creating a logistic regression and conerting the coefficients to a list.
regr = linear_model.LogisticRegression()
regr.fit(hr_tuning[hr_list], hr["left"])
coef = regr.coef_[0].tolist()

#Displaying the results and formatting it to display next to the name of each categorical value.
for x,y in zip(hr_list,coef):
    y=str(y)
    print(x+": "+y)

hr_tuning[column].mean()/hr_tuning[column].std()

log_important_features = []
log_coef_list = []

for coe,f_name in zip(coef,hr_list):
    if abs(coe) > std_val/10.5:
        log_important_features.append(f_name)
        log_coef_list.append(abs(coe))
        
print(log_important_features) #Put this into a df and sort it that way. #pd.DataFrame(dictionary) #Find a significance factor, special t test to prevent too many t tests.
print(log_coef_list)

len(log_important_features) == len(log_coef_list)
```

## Logistic Regression

```python
#Logistic regression without interaction.
import statsmodels.api as sm
sm_model = sm.Logit(hr_data['left'], sm.add_constant(hr_tuning)).fit(disp=0)
print(sm_model.pvalues)
sm_model.summary()

hr_tuning['Interaction (Time/Accident)'] = hr_tuning['Work_accident']*hr_tuning['time_spend_company']
hr_tuning.head()

#WITH interaction

sm_model = sm.Logit(hr_data['left'], sm.add_constant(hr_tuning)).fit(disp=0)
print(sm_model.pvalues)
sm_model.summary()

#Creating a list and looping through to format.
hr_list = list(hr_tuning.columns)

for item in hr_list:
    item = 'hr_list["'+item+'"]'

#Creating a logistic regression and conerting the coefficients to a list.
regr = linear_model.LogisticRegression()
regr.fit(hr_tuning[hr_list], hr["left"])
coef = regr.coef_[0].tolist()

#Displaying the results and formatting it to display next to the name of each categorical value.
for x,y in zip(hr_list,coef):
    y=str(y)
    print(x+": "+y)
```

## Ridge - Feature Selection

```python
#Repeating the same thing for Ridge.
hr_tuning = hr_data.copy() #creating a new dataframe without messing with the original dataframe.
del hr_tuning['left']

for column in hr_tuning.columns[:-10]:
    hr_tuning[column] = (hr_tuning[column]-hr_tuning[column].mean())/hr_tuning[column].std()
    
hr_tuning['Interaction (Time/Accident)'] = hr_tuning['Work_accident']*hr_tuning['time_spend_company']
del hr_tuning['management']

hr_list = list(hr_tuning.columns)

for item in hr_list:
    item = 'hr_list["'+item+'"]'
    
#Creating a ridge regression and conerting the coefficients to a list.
rig = Ridge(alpha=1.0)
rig.fit(hr_tuning[hr_list], hr["left"])
rig.score(hr_tuning[hr_list], hr["left"])
#coef = regr.coef_[0].tolist()

#Displaying the results and formatting it to display next to the name of each categorical value.
for x,y in zip(hr_list,rig.coef_):
    y=str(y)
    print(x + ": " + y)

#Ridge regression
rig_important_features = []
rig_coef_list = []

for coe,f_name in zip(rig.coef_,hr_list):
    if abs(coe) > std_val/80:
        rig_important_features.append(f_name)
        rig_coef_list.append(abs(coe))
        
print(rig_important_features) #Put this into a df and sort it that way. #pd.DataFrame(dictionary) #Find a significance factor, special t test to prevent too many t tests.
print(rig_coef_list)

len(rig_important_features) == len(rig_coef_list)
```

## Mediation

``` python
#Mediation

sm_model = sm.Logit(hr_data['left'], sm.add_constant(hr_tuning['Work_accident'])).fit(disp=0)
print(sm_model.pvalues)
sm_model.summary()

#Mediation

sm_model = sm.OLS(hr_data['time_spend_company'], sm.add_constant(hr_tuning['Work_accident'])).fit(disp=0)
print(sm_model.pvalues)
sm_model.summary()

#We did not see a mediating effect.
#Third part is logistic.


```
