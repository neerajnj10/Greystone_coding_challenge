
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import zipcode
import uszipcode
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi
from uszipcode import ZipcodeSearchEngine
### We will also need to suppress the scientific notation display to generate/restore floating point values
pd.options.display.float_format = '{:.2f}'.format
get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.float_format = '{:.2f}'.format
from math import log, exp
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn import grid_search
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta


# In[2]:


df = pd.read_csv('C:/Users/Nj_neeraj/Documents/re_data.csv')


# In[3]:


df.dtypes
# to change zip to object/character 
df['Zip'] = df.Zip.astype(object)
###first we remove empty cells
df['Year Built'].replace('', np.nan, inplace=True)
df.dropna(subset=['Year Built'], inplace=True)


# In[4]:


## we notice the year column also has multiple entries in the same cell, we will remove them as well.
df[df['Year Built'].str.contains(",")]
df = df[df['Year Built'].str.contains(",") == False]


# In[5]:


#check for nan values.
zip=df['Zip']
pd.isnull(zip).sum() > 0
##subset data with non-nan values in zip column
df_zipsub = df.dropna()
#df_zipsub = df[pd.notnull(df['Zip'])]
##check again for nan values.
zips=df_zipsub['Zip']
pd.isnull(zips).sum() > 0


# In[6]:


##now create states from zip
zips=zips.astype(int)
zips=zips.astype(str)
states = pd.DataFrame()
zipcode = pd.DataFrame()
for i in zips:
    search = ZipcodeSearchEngine()
    myzip = search.by_zipcode(i)
    if myzip:
        state = (myzip.State).split()
        states = states.append(state)
        zipp = (myzip.Zipcode).split()
        zipcode = zipcode.append(zipp)
        combine =  pd.concat([states, zipcode], ignore_index=True, axis=1)


# In[7]:


combine.columns = ['State','Zip']
combine = combine.reset_index(drop=True)


# In[8]:


df_zipsub['Zip'] = df_zipsub['Zip'].astype(int)
df_zipsub['Zip'] = df_zipsub['Zip'].astype(str)
df_new = pd.merge(df_zipsub,combine, on = 'Zip', how= 'left')
df_new=df_new.drop_duplicates() 
#df_new.sort_values(by=['First Payment Date'], ascending=[True])
#df_new = df_new.dropna()


# In[9]:


### We find if there are NA rows.
null_data = df_new[df_new.isnull().any(axis=1)]
### We are able to see that there are rows for which 'State' columns returns NaN values because of 'invalid type' zipcodes.
### so we will remove them from our dataset now.
df_new = df_new.dropna()
### Now we have clean dataset and we will reste the index one more time for clarity and readability purpose.
df_new = df_new.reset_index(drop=True)


# In[10]:


### Lets first find out which state has highest average loan amount

average = df_new.groupby(['State']).mean()
average = average[['Loan Amount']]
##Sort it descending order to see which state has highest value.
average.sort_values(by=['Loan Amount'], ascending=[False])


# In[11]:


## We can clearly see Ohio has the highest average loan amount.
## We can visualize it now to make the comparison clear.
sns.set(style="whitegrid", color_codes=True)


# In[12]:


average.reset_index().plot(x='State', y='Loan Amount')


# In[13]:


### taxes as a % of property value
## df_new.dtypes shows that 'Property Value' column intead of Floating point, which gives the indication the column has erroneous 
## values stored along as well, we can confirm our assumption using the following

#df_new['Property Value'] = df_new['Property Value'].astype(float)
## Above line of code will throw error suggesting there are values such as 'Error' stored in the column.
##Following line of code confirms the same.

df_new[df_new['Property Value'].str.contains("Error")]
### Now we will remove thr errorneous rows from our dataset.
df_new = df_new[df_new['Property Value'].str.contains("Error") == False]

### Now we can convert object type to floating point for calculation.
df_new['Property Value'] = df_new['Property Value'].astype(float)
df_new['Tax_Percent'] = (df_new['Taxes Expense']/df_new['Property Value'])*100


# In[14]:


state_tax = df_new[['State','Tax_Percent']]
state_tax.sort_values(by=['Tax_Percent'], ascending=[False])
### At this point our assumption that data is cleaned is challenged as we find an 'Inf' value for State Florida because it has
### the value '0' stored in 'Property Value' column, therefore we will delete this row from the original dataframe to clean it further.
df_new = df_new[df_new['Property Value'] != 0]
##calling above line of codes again to see the data without Inf values now.
state_tax = df_new[['State','Tax_Percent', 'Property Value', 'Taxes Expense', 'Zip', 'Loan Amount']]
state_tax.sort_values(by=['Tax_Percent'], ascending=[False])

### We can clearly see Illinois has highest tax percent per property value. We can also see that many states are repeated in
### the sequence which is because of different zip codes that these property belong to, and price of each property can genuinely differ
### as some location might more expensive than others within the same state.

#df_new[df_new['Taxes Expense']==0]
#df_new[df_new['Effective Gross Income']==0]

### We also notice there are values in 'Taxes Expense' that are 0, now we may argue, can tax expenses be 0? For the sake of 
### 'real world' scenario we know it can not be, and we would want to delete these rows from original dataset again. However
### another argument we can made is, when we are trying to find the amount to lend a candidate profile, do we want to also,
### find out when we can not lend any money at all? such as the case with loan amount as 0, or do we only want to focus on 
### actual non-zero amount to lend someone. If we want to do the former, then we might want to keep 0 values in Taxes expense and 
### other columns/variables as well. This is subjective and for now, we will keep it simple with our given understanding.


# In[15]:


### Maintenance as a percentage of property value.
df_new['Maintenance_Percent'] = (df_new['Maintenance Expense']/df_new['Property Value'])*100


# In[16]:


### Easiest ad simplest way to find the strong predictors, is corrlation matrix or scatter plot matrix that describes the 
### dependencies and relation between two variables to each other.

df_maintenance = df_new[df_new.columns.difference(['Maintenance_Percent','Tax_Percent'])]
#print(df_maintenance.corr())
### Now if we just look at the 'Maintenance Expense' then we can see 'Total Operating Expenses' is strongest predictor with 91% 
### positive correlation.
### However in order to find the strong predictor(s) for 'Maintenance_Percent', we will remove Maintenance Expenses and 
### Property Value column, since it is directly derived from them. Additionally we will remove 'Tax_percent' column as well,
### since it is a derived variable as well (not i original dataset) and was obtained from Property Value varibale as well.
df_maintenance = df_new[df_new.columns.difference(['Maintenance Expense','Property Value','Tax_Percent'])]
print(df_maintenance.corr())

### From this we can see and interpret there are no 'strongest' predictor for Maintenance_Percent, the relation with other variables is fairly low
### and this indicates, the Maintenance_Percent is not a very useful variable for us to keep as well.
### However to answer the original question, the strongest predictor from the list is 'Insurance Expensse' with 29%, cloely followed by
### 'Total Operating Expenses' with 27% of relation. Note, 'Total Operating Expenses' was strongest predictor for 'Maintenance Expense'
### variable.


# In[17]:


corr = df_maintenance.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[18]:


df_maintenance.head()


# In[19]:


### Now we can also confirm that Maintenance_Percent doe snot have strong relation with any predictor, we can use Anova test for
### the same and notice R-squared value to be very low for any consideration.
### In addition notice F-statistice value is not high at all, hence we can conclude our hypothesis in the favor of null relation between
### Maintenance_Percent and Insurance Expenses, similarly for Total Operating Expenses

#ANOVA F Test

model = smf.ols(formula='Maintenance_Percent ~ df_maintenance["Insurance Expense"]', data=df_maintenance)
results = model.fit()
print (results.summary())
model = smf.ols(formula='Maintenance_Percent ~ df_maintenance["Total Operating Expenses"]', data=df_maintenance)
results = model.fit()
print (results.summary())


# In[20]:


median= df_new['Loan Amount'].median()
range_ = (df_new['Loan Amount'].max() - df_new['Loan Amount'].min())
var = df_new['Loan Amount'].var()
### We can obtain the same with describe(summary statistic)
df_new['Loan Amount'].describe()
### Calculation of loan amount from the above statistics look highly un-deterministic, rather probabilistic.
df_new.columns


# In[21]:


df1 = df_new[['Property Value','Total Operating Expenses',
       'Maintenance Expense', 'Parking Expense', 'Taxes Expense',
       'Insurance Expense', 'Utilities Expense', 'Payroll Expense']]


# In[22]:


g = sns.PairGrid(df1)
g.map(plt.scatter);


# In[23]:


df1.corr()


# In[24]:


### From the scatter plot (first column) and the correlation matrix (first column), we are able to find certain pattern.
### We see that property value has very high linear trending relation with 'Total Operating Expenses', 'Maintenance Expenses',
### 'Taxes Expenses', somewhat strong linear upward trending relation with 'Payroll expense', 'Insurance Expense',, and very less,
### and more scatterted (less linear) trending relation with 'Parking Expense' and 'Utilities Expense'.

### What we can say from this, is that with the Expenses, Property Value shows strong upward trending relation, are able to exhibit 
### economic relation, where Higher Property value generally would mean Higher Expenses in terms of Total Operation, Maintenance
### and higher taxes, somewhat Higher Payroll as well to compensate for location and property value, and somehwat higher insurance expenses.
### To Summarize, this basically confirms our assumptions, that the locations (zipcodes), with higher property value are by default
### more expensive overall than locations with lower property values.

### We can also try to add all the expenses together as one and see their combined relation with property value.
df1['Total_Expenses'] =df1['Total Operating Expenses'].values + df1['Maintenance Expense'].values + df1['Parking Expense'].values +df1['Taxes Expense'].values + df1['Insurance Expense'].values + df1['Utilities Expense'].values + df1['Payroll Expense'].values


# In[25]:


df1[['Property Value', 'Total_Expenses']].corr()
g = sns.PairGrid(df1[['Property Value', 'Total_Expenses']])
g.map(plt.scatter);

### We are able to confirm, higher property value means higher expenses. Property values are usually higher in mmore expensive locations/areas,
### hence higher cost of overall living.


# In[26]:


df_new['Expense_ratio'] = df_new['Total Operating Expenses']/df_new['Effective Gross Income']


# In[27]:


fig, ax = plt.subplots(figsize=(10,10)) 
sns.stripplot(x="State", y="Expense_ratio", data=df_new, jitter=True, ax= ax);


# In[28]:


fig, ax = plt.subplots(figsize=(10,10)) 
sns.boxplot(x="State", y="Expense_ratio", data=df_new, ax=ax);


# In[29]:


fig, ax = plt.subplots(figsize=(10,10)) 
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.barplot(x="State", y="Expense_ratio", data=df_new,ci=None, ax = ax)


# In[30]:


### We can see from all 3 plots that states on the right side have higher expense ratio concentration, higher than mean value 
df_new['Expense_ratio'].mean()
### A stand out observation from this is of NY state, which has lowest Expense ration compared to other states.


# In[31]:


df_new.head()
df2 = df_new[['Year Built','Total Operating Expenses',
       'Maintenance Expense', 'Parking Expense', 'Taxes Expense',
       'Insurance Expense', 'Utilities Expense', 'Payroll Expense']]
df2['Total_Expenses'] =df2['Total Operating Expenses'].values + df2['Maintenance Expense'].values + df2['Parking Expense'].values +df2['Taxes Expense'].values + df2['Insurance Expense'].values + df2['Utilities Expense'].values + df2['Payroll Expense'].values


# In[32]:


#df_new.to_csv('C:/Users/Nj_neeraj/Documents/clean_data.csv', sep=',')
df2.to_csv('C:/Users/Nj_neeraj/Documents/expense_year.csv', sep=',')
###link to plot for Expenses vs Year Built. 
### We ae able to clearly identify the pattern where we see that as years roll by,the overall expenses are increasing.
### From less than a million in year 1940 and below, to close to 5 million and above, for succesive years.
### correlation between expenses and year for property comes out ot be .42
#https://plot.ly/~neeraj10/13/


# In[33]:


###Create fiscal quarter
df_new['First Payment Date'] = pd.to_datetime(df_new['First Payment Date'])
df_new['Maturity Date'] = pd.to_datetime(df_new['Maturity Date'])
df_new['Quarter'] = df_new['First Payment Date'].dt.quarter
df_new['Year']  = df_new['First Payment Date'].dt.year
df_new['Year'] = df_new['Year'].astype(object)
df_new['Quarter'] = df_new['Quarter'].astype(object)
df_new['Quarter'] = df_new["Year"].map(str) + "_" +df_new["Quarter"].map(str)
df3 = df_new[['Loan Amount', 'Quarter']]


# In[34]:


quarter_cumulate = df3.groupby(by=['Quarter']).sum().groupby(level=[0]).cumsum()
quarter_cumulate.reset_index().plot(x='Quarter', y='Loan Amount')
### As we can clearly see, the exponential rise in loan amount from 4th quarter of 2015, peaking the highest in Quarter 4th of 2016
### and again, trending down from there till quarter 4th of 2017. It is almost a bell shaped curve that defines almost the normal
### distribution nature and relation between loan amount and quarterly distribution.


# In[52]:


### Loan Prediction
df_new['Loan_Days'] = df_new['Maturity Date'] - df_new['First Payment Date']
##create new feature of loan days and then convert timedelta to float/int type to use in model building.
df_new['Loan_Days'] = df_new.Loan_Days/timedelta (days=1)

##We remove rows with 0 values and then scale the data with logarithmic operation to make sure the range of data is within 
## limits, a more extended range usually performs bad, hence scaling  is a good idea. We will scale almost all the columns
## that have huge range.

### Removing loan amount of 0 value is again subjective, we can use different scaling (instead of log) method if we wish to retain 0 values.
df_new = df_new[df_new['Loan Amount'] != 0]
#df_new = df_new[df_new['Property Value'] != 0]
#df_new = df_new[df_new['Net Operating Income'] != 0]
#df_new = df_new[df_new['Total Operating Expenses'] != 0]
#df_new = df_new[df_new['Effective Gross Income'] != 0]
#df_new = df_new[df_new['Taxes Expense'] != 0]
#df_new = df_new[df_new['Insurance Expense'] != 0]
#df_new = df_new[df_new['Utilities Expense'] != 0]
#df_new = df_new[df_new['Payroll Expense'] != 0]
df_new['Loan Amount'] = np.log(df_new['Loan Amount'])
df_new['Property Value'] = np.log(df_new['Property Value'])
df_new['Net Operating Income'] = df_new['Net Operating Income'].replace(0,df_new['Net Operating Income'].mean())
df_new['Net Operating Income'] = np.log(df_new['Net Operating Income'])
df_new['Effective Gross Income'] = df_new['Effective Gross Income'].replace(0,df_new['Effective Gross Income'].mean())
df_new['Effective Gross Income'] = np.log(df_new['Effective Gross Income'])
df_new['Total Operating Expenses'] = df_new['Total Operating Expenses'].replace(0,df_new['Total Operating Expenses'].mean())
df_new['Total Operating Expenses'] = np.log(df_new['Total Operating Expenses'])
df_new['Parking Expense'] = df_new['Parking Expense'].replace(0,df_new['Parking Expense'].mean())
df_new['Parking Expense'] = np.log(df_new['Parking Expense'])
df_new['Taxes Expense'] = df_new['Taxes Expense'].replace(0,df_new['Taxes Expense'].mean())
df_new['Taxes Expense'] = np.log(df_new['Taxes Expense'])
df_new['Insurance Expense'] = df_new['Insurance Expense'].replace(0,df_new['Insurance Expense'].mean())
df_new['Insurance Expense'] = np.log(df_new['Insurance Expense'])
df_new['Utilities Expense'] = df_new['Utilities Expense'].replace(0,df_new['Utilities Expense'].mean())
df_new['Utilities Expense'] = np.log(df_new['Utilities Expense'])
df_new['Payroll Expense'] = df_new['Payroll Expense'].replace(0,df_new['Payroll Expense'].mean())
df_new['Payroll Expense'] = np.log(df_new['Payroll Expense'])


# In[53]:


##Now we split the dataset to train and test set.
train, test = train_test_split(df_new, test_size=0.2)


# In[55]:


pd.options.display.float_format = '{:.2f}'.format
num_vbl = {'Property Value','Net Operating Income','Effective Gross Income','Total Operating Expenses','Maintenance Expense',
          'Parking Expense','Taxes Expense','Insurance Expense','Utilities Expense','Payroll Expense','Expense_ratio','Loan_Days',
          'Tax_Percent',"Maintenance_Percent"}
cat_vbl = {'Quarter','Year','State','Year Built','Zip'}
for var in num_vbl:
    train[var] = train[var].fillna(value = train[var].mean())
    test[var] = test[var].fillna(value = test[var].mean())
train = train.fillna(value = -999)
test = test.fillna(value = -999)
print ("Filled Missing Values")
print ("Starting Label Encode")
for var in cat_vbl:
    lb = preprocessing.LabelEncoder()
    full_data = pd.concat((train[var],test[var]),axis=0).astype('str')
    lb.fit( full_data )
    train[var] = lb.transform(train[var].astype('str'))
    test[var] = lb.transform(test[var].astype('str'))
features = {'Property Value','Net Operating Income','Effective Gross Income','Total Operating Expenses','Maintenance Expense',
          'Parking Expense','Taxes Expense','Insurance Expense','Utilities Expense','Payroll Expense','Loan_Days','Quarter','Year','State','Year Built','Zip'}
 
x_train = train[list(features)].values
y_train = train['Loan Amount'].values
x_test = test[list(features)].values


# In[56]:


param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["mse"],
              "n_estimators": [80, 160, 320, 400]}
gs = grid_search.GridSearchCV(RandomForestRegressor(), param_grid=param_grid)
gs.fit(x_train, y_train)


# In[57]:


print ("Starting to predict on the dataset")
rec= gs.predict(x_test)
print ("Prediction Completed")
test['LoanAmount_Predicted'] = rec
print("r2 / variance : ", gs.best_score_)


# In[66]:


test['LoanAmount_Predicted'] = np.exp(test['LoanAmount_Predicted'])
test['Loan Amount'] = np.exp(test['Loan Amount'])
test[['LoanAmount_Predicted','Loan Amount']]

### Our current model explains 80% of variance in output variable, that is, loan amount.
### There is of course scope of improvement, depending on 'domain knowledge', whether we want to replace 0 values with mean or not
### which columns can have 0 values and which ones can not. We can also create more advanced feature combinations and 
### and even tune m=ourhyper parameter more to further make the predictions better.

### In addition, we can try different type of models and compare them for the accuracy, and select the best one, this is an
### iterative process of course and would require more work and more time. 
### However this should provide good idea on how we can create model and make predictions to achieve a respectable accuracy to begin with.

