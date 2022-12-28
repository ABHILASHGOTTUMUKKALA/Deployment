#!/usr/bin/env python
# coding: utf-8

# # Business Understanding
# ### Customer churn is a big problem for telecommunications companies. Indeed, their annual churn rates are usually higher than 10%. For that reason, they develop strategies to keep as many clients as possible. This is a classification project since the variable to be predicted is binary (churn or loyal customer). The goal here is to model churn probability, conditioned on the customer features.

# # ![image.png](attachment:image.png)

# ## Data Acquisition
# 

# In[1]:


# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# DataFrame Preparation
df=pd.read_csv(r"C:\Users\abhil\Downloads\Model_deployment\telecommunications_churn (1).csv")


# ## EDA

# In[3]:


# Checking first 10 Rows
df.head(10)


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


# Checking for Duplicates
df.duplicated().sum()


# In[8]:


# Checking Data Size
df.size


# In[9]:


# Checking for Null Values
df.isnull().sum()


# In[10]:


# Plotting heatmap for Null Values
sns.heatmap(df.isnull())


# In[11]:


col=df.columns


# In[12]:


col


# In[13]:


# Plotting Distplot and Histplot for checking data Distribution.
#columns=['account_length', 'voice_mail_plan', 'voice_mail_messages', 'day_mins',
#       'evening_mins', 'night_mins', 'international_mins',
#       'customer_service_calls', 'international_plan', 'day_calls',
#       'day_charge', 'evening_calls', 'evening_charge', 'night_calls',
#       'night_charge', 'international_calls', 'international_charge',
#       'total_charge', 'churn']
#for col in columns:
#    sns.displot(x=df[col],kde=True)


# In[14]:


# Checking for Skewness in Dataset
df.skew().sort_values(ascending=True)


# ### International_charge  and international_mins has Negative Skewness 
# ### Voice_mail_plan, customer_service_calls, voice_mail_messages, international_calls, churn and international_plan has Positive Skewness

# In[15]:


# Checking for Kurtosis in Dataset
df.kurtosis().sort_values(ascending=True)


# ### Voice_mail_plan has Negative Kurtosis
# ### Day_calls, international_mins, international_charge, customer_service_calls, churn, international_calls, international_plan has Positive Kurtosis

# In[16]:


## Checking for Correlation in Dataset
df.corr()


# In[17]:


# Box Plot for Outlier Detection.
#plt.figure(figsize=(17,17))
#labels=df.columns
#plt.boxplot(df,labels=labels,vert=False)
#plt.show()


# #### Account_length has Outliers in Lower Quartile
# #### Voice_mail_messages has Outliers in Upper Quartile
# #### Day_mins has Outliers in Both Upper and Lower Quartile
# #### Evening_mins has Outliers in Both Upper and Lower Quartile
# #### Night_mins has Outliers in Both Upper and Lower Quartile
# #### International_mins has Outliers in Both Upper and Lower Quartile
# #### Customer_service_calls has Outliers in Upper Quartile
# #### International_plan has Outliers in Upper Quartile
# #### Day_calls has Outliers in Both Upper and Lower Quartile
# #### Day_charge has Outliers in Both Upper and Lower Quartile
# #### Evening_calls has Outliers in Both Upper and Lower Quartile
# #### Evening_charge has Outliers in Both Upper and Lower Quartile
# #### Customer_service_calls has Outliers in Upper Quartile
# #### Night_calls has Outliers in Both Upper and Lower Quartile
# #### Night_charge has Outliers in Both Upper and Lower Quartile
# #### International_calls has Outliers in Upper Quartile
# #### International_charge has Outliers in Both Upper and Lower Quartile
# #### Total_charge has Outliers in Both Upper and Lower Quartile

# In[18]:


#Checking churn count
df.churn.value_counts()


# In[19]:


# plotting Chrun and Non Churn
#plt.pie(df['churn'].value_counts(), autopct='%1.1f%%')
#plt.show()


# In[20]:


df['account_length'].value_counts()


# In[21]:


# plotting pairplot
#sns.pairplot(df, hue='churn')


# In[22]:


# Plotting a heatmap for Correlation
plt.figure(figsize=(17,17))
sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)


# ### Voice_mail_plan & voice_mail_messages has Strong Correlation
# ### Total Charge & Day_min, Day_Charges has a Strong Correlation
# ### Total Charge & Evening_min, Evening_Charges has a  Weak Correlation

# In[23]:


#sns.histplot(x='account_length', data=df, hue='churn')


# #### Customers with account_length account length around 50 to 150 has higher Churn

# In[24]:


sns.countplot(x=df["voice_mail_plan"], hue=df['churn'])


# #### Customers with No Voice_mail Plan has higher Churn 

# In[25]:


#sns.histplot(x=df["voice_mail_messages"], hue=df['churn'])


# #### Customers with No Voice_mail Messages has higher Churn 

# In[26]:


#sns.histplot(x=df["day_mins"], hue=df['churn'])


# #### Higher Churn is in found in customers with day_mins around 100 to 280

# In[27]:


#sns.histplot(x=df["evening_mins"], hue=df['churn'])


# #### Higher Churn is in found in customers with day_mins around 130 to 300

# In[28]:


#sns.histplot(x=df["night_mins"], hue=df['churn'])


# #### Higher Churn is in found in customers with Night_mins around 120 to 280

# In[29]:


#sns.countplot(x=df["customer_service_calls"], hue=df['churn'])


# In[30]:


sns.countplot(x=df["international_plan"], hue=df['churn'])


# #### Higher Churn is in found in customers with No International Plan

# In[31]:


sns.histplot(x=df["day_calls"], hue=df['churn'])


# #### Higher Churn is in found in customers with day_calls around 70 to 130

# In[32]:


#sns.histplot(x=df["day_charge"], hue=df['churn'])


# #### Higher Churn is in found in customers with day_charges around 15 to 50

# In[33]:


#sns.histplot(x=df["evening_calls"], hue=df['churn'])


# #### Higher Churn is in found in customers with Evening_Calls around 70 to 130

# In[34]:


#sns.histplot(x=df["evening_charge"], hue=df['churn'])


# #### Higher Churn is in found in customers with Evening_Charge around 14 to 25

# In[35]:


#sns.histplot(x=df["night_calls"], hue=df['churn'])


# #### Higher Churn is in found in customers with day_mins around 70 to 130

# In[36]:


#sns.histplot(x=df["night_charge"], hue=df['churn'])


# #### Higher Churn is in found in customers with Night_Charges around 6 to 12.5

# In[37]:


#sns.histplot(x=df["international_calls"], hue=df['churn'])


# #### Higher Churn is in found in customers with International_Calls around 2.5 to 6.0

# In[38]:


#sns.histplot(x=df["total_charge"], hue=df['churn'])


# #### Higher Churn is in found in customers with Total_Charges around 45 to 80

# ## Outlier Detection

# ### Outlier Detection can be done using following Methods
# #### 1. Numeric Outlier
# #### 2. Z-Score 
# #### 3. DBSCAN
# #### 4. Isolation Forest

# ### Numeric Outlier Method

# #### Steps Involved
# ##### Calculating the IQR Range
# ##### Finding the Upper and lower limit

# In[39]:


# Finding the IQR
percentile25 = df.quantile(0.25)
percentile75 = df.quantile(0.75)


# In[40]:


# Checking Values for 75 Percentile
percentile75


# In[41]:


# Checking Values for 25 Percentile
percentile25


# In[42]:


# Calculating the IQR
iqr = percentile75 - percentile25


# In[43]:


iqr


# In[44]:


# Calculating the Upper and lower limit for each columns
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr


# In[45]:


# Printing Upper and Lower Limits
print("Upper limit",upper_limit)
print("Lower limit",lower_limit)


# ### Finding the Outliers

# #### Highest number of Outliers are found in Columns customer_service_calls, international_plan

# In[46]:


# Finding the number of outliers
df[(df < lower_limit) | (df > upper_limit)].count()


# In[47]:


# Finding the percentage of Outliers
(df[(df < lower_limit) | (df > upper_limit)].count())/len(df)*100


# #### Highest Percentage of Outliers are found in Columns customer_service_calls, international_plan
# #### customer_service_calls, international_plan & international_mins has Contributed to around 20% of outliers in the Dataframe

# ### Capping the Outliers

# #### Using Capping method for outlier treatment as the number of datapoints are less.
# #### Deleting the Data would result in less datapoints which can result in underfitting and may produce incorrect prediction on the new datapoints

# In[48]:


# Creaing a New dataframe for capping the Outliers.
new_df_cap = df.copy()

new_df_cap['account_length'] = np.where(
    new_df_cap['account_length'] > upper_limit['account_length'],
    upper_limit['account_length'],
    np.where(
        new_df_cap['account_length'] < lower_limit['account_length'],
        lower_limit['account_length'],
        new_df_cap['account_length']
    )
)


# In[49]:


new_df_cap['voice_mail_messages'] = np.where(
    new_df_cap['voice_mail_messages'] > upper_limit['voice_mail_messages'],
    upper_limit['voice_mail_messages'],
    np.where(
        new_df_cap['voice_mail_messages'] < lower_limit['voice_mail_messages'],
        lower_limit['voice_mail_messages'],
        new_df_cap['voice_mail_messages']
    )
)
new_df_cap['day_mins'] = np.where(
    new_df_cap['day_mins'] > upper_limit['day_mins'],
    upper_limit['day_mins'],
    np.where(
        new_df_cap['day_mins'] < lower_limit['day_mins'],
        lower_limit['day_mins'],
        new_df_cap['day_mins']
    )
)
new_df_cap['evening_mins'] = np.where(
    new_df_cap['evening_mins'] > upper_limit['evening_mins'],
    upper_limit['evening_mins'],
    np.where(
        new_df_cap['evening_mins'] < lower_limit['evening_mins'],
        lower_limit['evening_mins'],
        new_df_cap['evening_mins']
    )
)
new_df_cap['night_mins'] = np.where(
    new_df_cap['night_mins'] > upper_limit['night_mins'],
    upper_limit['night_mins'],
    np.where(
        new_df_cap['night_mins'] < lower_limit['night_mins'],
        lower_limit['night_mins'],
        new_df_cap['night_mins']
    )
)
new_df_cap['international_mins'] = np.where(
    new_df_cap['international_mins'] > upper_limit['international_mins'],
    upper_limit['international_mins'],
    np.where(
        new_df_cap['international_mins'] < lower_limit['international_mins'],
        lower_limit['international_mins'],
        new_df_cap['international_mins']
    )
)
new_df_cap['customer_service_calls'] = np.where(
    new_df_cap['customer_service_calls'] > upper_limit['customer_service_calls'],
    upper_limit['customer_service_calls'],
    np.where(
        new_df_cap['customer_service_calls'] < lower_limit['customer_service_calls'],
        lower_limit['customer_service_calls'],
        new_df_cap['customer_service_calls']
    )
)
new_df_cap['day_calls'] = np.where(
    new_df_cap['day_calls'] > upper_limit['day_calls'],
    upper_limit['day_calls'],
    np.where(
        new_df_cap['day_calls'] < lower_limit['day_calls'],
        lower_limit['day_calls'],
        new_df_cap['day_calls']
    )
)
new_df_cap['day_charge'] = np.where(
    new_df_cap['day_charge'] > upper_limit['day_charge'],
    upper_limit['day_charge'],
    np.where(
        new_df_cap['day_charge'] < lower_limit['day_charge'],
        lower_limit['day_charge'],
        new_df_cap['day_charge']
    )
)
new_df_cap['evening_calls'] = np.where(
    new_df_cap['evening_calls'] > upper_limit['evening_calls'],
    upper_limit['evening_calls'],
    np.where(
        new_df_cap['evening_calls'] < lower_limit['evening_calls'],
        lower_limit['evening_calls'],
        new_df_cap['evening_calls']
    )
)
new_df_cap['evening_charge'] = np.where(
    new_df_cap['evening_charge'] > upper_limit['evening_charge'],
    upper_limit['evening_charge'],
    np.where(
        new_df_cap['evening_charge'] < lower_limit['evening_charge'],
        lower_limit['evening_charge'],
        new_df_cap['evening_charge']
    )
)
new_df_cap['night_calls'] = np.where(
    new_df_cap['night_calls'] > upper_limit['night_calls'],
    upper_limit['night_calls'],
    np.where(
        new_df_cap['night_calls'] < lower_limit['night_calls'],
        lower_limit['night_calls'],
        new_df_cap['night_calls']
    )
)
new_df_cap['night_charge'] = np.where(
    new_df_cap['night_charge'] > upper_limit['night_charge'],
    upper_limit['night_charge'],
    np.where(
        new_df_cap['night_charge'] < lower_limit['night_charge'],
        lower_limit['night_charge'],
        new_df_cap['night_charge']
    )
)
new_df_cap['international_calls'] = np.where(
    new_df_cap['international_calls'] > upper_limit['international_calls'],
    upper_limit['international_calls'],
    np.where(
        new_df_cap['international_calls'] < lower_limit['international_calls'],
        lower_limit['international_calls'],
        new_df_cap['international_calls']
    )
)
new_df_cap['international_charge'] = np.where(
    new_df_cap['international_charge'] > upper_limit['international_charge'],
    upper_limit['international_charge'],
    np.where(
        new_df_cap['international_charge'] < lower_limit['international_charge'],
        lower_limit['international_charge'],
        new_df_cap['international_charge']
    )
)
new_df_cap['total_charge'] = np.where(
    new_df_cap['total_charge'] > upper_limit['total_charge'],
    upper_limit['total_charge'],
    np.where(
        new_df_cap['total_charge'] < lower_limit['total_charge'],
        lower_limit['total_charge'],
        new_df_cap['total_charge']
    )
)


# In[50]:


# Outlier count after capping the Obutliers.
new_df_cap[(new_df_cap>upper_limit)|(new_df_cap<lower_limit)].count()


# In[51]:


new_df_cap.describe()


# In[52]:


# Shape of New Df
new_df_cap.shape


# In[53]:


# Box Plot for Outlier Detection after treatment.
#plt.figure(figsize=(17,17))
#labels=new_df_cap.columns
#plt.boxplot(new_df_cap,labels=labels,vert=False)
#plt.show()


# #### After Capping and Plotting the Boxplot we see not outliers

# ## Trimming

# In[54]:


# Dropping the Outliers
new_df_trim = df.copy()
new_df_trim1 = new_df_trim[(new_df_trim['account_length'] < 206.500) & (new_df_trim['account_length'] >  -5.500)]
new_df_trim2= new_df_trim1[(new_df_trim1['voice_mail_plan'] < 2.500) & (new_df_trim1['voice_mail_plan'] >  -1.500)]
new_df_trim3= new_df_trim2[(new_df_trim2['voice_mail_messages'] < 50.000) & (new_df_trim2['voice_mail_messages'] >  -30.000)]
new_df_trim4= new_df_trim3[(new_df_trim3['day_mins'] < 325.450) & (new_df_trim3['day_mins'] >  34.650)]
new_df_trim5= new_df_trim4[(new_df_trim4['evening_mins'] < 338.350) & (new_df_trim4['evening_mins'] >  63.550)]
new_df_trim6= new_df_trim5[(new_df_trim5['night_mins'] < 337.750) & (new_df_trim5['night_mins'] >  64.550)]
new_df_trim7= new_df_trim6[(new_df_trim6['international_mins'] < 17.500) & (new_df_trim6['international_mins'] >  3.100)]
new_df_trim8= new_df_trim7[(new_df_trim7['customer_service_calls'] < 3.500) & (new_df_trim7['customer_service_calls'] > -0.500)]
new_df_trim_10 = new_df_trim8[(new_df_trim8['day_calls'] < 154.500) & (new_df_trim8['day_calls'] >  46.500)]
new_df_trim_11 = new_df_trim_10[(new_df_trim_10['day_charge'] < 55.330) & (new_df_trim_10['day_charge'] >  5.890)]
new_df_trim_12 = new_df_trim_11[(new_df_trim_11['evening_calls'] < 154.500) & (new_df_trim_11['evening_calls'] >  46.500)]
new_df_trim_13 = new_df_trim_12[(new_df_trim_12['evening_charge'] < 28.760) & (new_df_trim_12['evening_charge'] >  5.400)]
new_df_trim_14 = new_df_trim_13[(new_df_trim_13['night_calls'] < 152.000) & (new_df_trim_13['night_calls'] >  48.000)]
new_df_trim_16 = new_df_trim_14[(new_df_trim_14['night_charge'] < 15.195) & (new_df_trim_14['night_charge'] >  2.915)]
new_df_trim_17 = new_df_trim_16[(new_df_trim_16['international_calls'] < 10.500) & (new_df_trim_16['international_calls'] > -1.500)]
new_df_trim_18 = new_df_trim_17[(new_df_trim_17['international_charge'] < 4.725) & (new_df_trim_17['international_charge'] > 0.845)]
new_df_trim_final = new_df_trim_18[(new_df_trim_18['total_charge'] < 87.630) & (new_df_trim_18['total_charge'] >  31.230)] 


# In[55]:


new_df_trim_final.shape


# ### Dropping the Outliers has resulted in loss of 543 datapoints

# In[56]:


new_df_trim_final


# In[57]:


new_df_trim_final['churn'].value_counts()


# ### Deletion of outliers has resulted in removal of 361 datapoints for Customers Stayed back and 182 Datapoints from Churned Customers

# In[58]:


df.shape


# ## IsolationForest

# In[59]:


df_isf = df.copy()
from sklearn.ensemble import IsolationForest
clf = IsolationForest(n_estimators=20, warm_start=True)
clf.fit(df_isf)
df_isf['scores']=clf.predict(df_isf)


# In[60]:


# 1 is not an Outlier and -1 is an outlier
sns.countplot(x='scores', data = df_isf, hue='churn')


# In[61]:


# Deleting the outliers
df_new_final = df_isf[df_isf.scores != -1]


# In[62]:


df_new_final.shape


# In[63]:


df_new_final['churn'].value_counts()


# #### Isolation Forest has Detected around 603 Datapoints as outliers
# #### Highest Number of outliers are being detected in the Customers who have churned and deleting data using isolation forest is resulting in loss of 226 datapoints from the Churned customers.

# ## Z_Score

# In[64]:


print("Mean value of cgpa",df.mean())
print("Std value of cgpa",df.std())
print("Min value of cgpa",df.min())
print("Max value of cgpa",df.max())


# In[65]:


Highest_allowed = df.mean() + 3*df.std()
Lowest_allowed = df.mean() - 3*df.std()


# In[66]:


Highest_allowed


# In[67]:


Lowest_allowed


# In[68]:


# Finding Outliers
df[(df[col] > Highest_allowed[col]) | (df[col] < Lowest_allowed[col])].count()


# #### International_calls has Highest Number of Outliers

# In[69]:


# Finding Outliers Percentage
df[(df[col] > Highest_allowed[col]) | (df[col] < Lowest_allowed[col])].count()*100/len(df)


# #### International_calls had the Highest Percentage outliers

# In[70]:


df_Z_score_cap = df.copy()


# In[71]:


cols = ['account_length', 'voice_mail_plan', 'voice_mail_messages', 'day_mins',
       'evening_mins', 'night_mins', 'international_mins',
       'customer_service_calls', 'day_calls',
       'day_charge', 'evening_calls', 'evening_charge', 'night_calls',
       'night_charge', 'international_calls', 'international_charge',
       'total_charge']


# In[72]:


def Z_score_capping(df_Z_score_cap, cols):
    for col in cols:
        Highest_allowed = df.mean() + 3*df.std()
        Lowest_allowed = df.mean() - 3*df.std()
        df_Z_score_cap[col] = np.where(df_Z_score_cap[col] > Highest_allowed[col], Highest_allowed[col], np.where(df_Z_score_cap[col] < Lowest_allowed[col], Lowest_allowed[col], df_Z_score_cap[col]))


# In[73]:


Z_score_capping(df_Z_score_cap, cols)


# In[74]:


# Finding Outliers
df_Z_score_cap[(df_Z_score_cap[col] > Highest_allowed[col]) | (df_Z_score_cap[col] < Lowest_allowed[col])].count()


# ## Feature Engineering

# ### Feature engineering refers to manipulation — addition, deletion, combination, mutation — of your data set to improve machine learning model training, leading to better performance and greater accuracy.

# In[75]:


df_Z_score_cap


# In[76]:


# Creating a new column as Total_mins by merging day_mins, evening_mins, night_mins & international_mins
df_Z_score_cap['Total_mins'] = df_Z_score_cap['day_mins']+df_Z_score_cap['evening_mins']+df_Z_score_cap['night_mins']+df_Z_score_cap['international_mins']


# In[77]:


# Creating a new column as Total_calls by merging day_calls, evening_calls, night_calls & international_calls
df_Z_score_cap['Total_calls'] = df_Z_score_cap['day_calls']+df_Z_score_cap['evening_calls']+df_Z_score_cap['night_calls']+df_Z_score_cap['international_calls']


# In[78]:


df_Z_score_cap


# In[79]:


# Creating a new Dataframe
df_final = df_Z_score_cap.copy()


# In[80]:


# Dropping the old columns
df_final.drop(columns=['day_mins','evening_mins','night_mins','international_mins','day_calls','evening_calls','night_calls','international_calls'], inplace=True)


# In[81]:


# Dropping the old columns
df_final.drop(columns=['day_charge','evening_charge','night_charge','international_charge'], inplace=True)


# In[82]:


# Final Dataframe after dropping the unwanted columns
df_final


# In[83]:


# Checking correlation for new Dataframe.
df_final.corr()


# In[84]:


# Plotting a correlation plot
sns.heatmap(df_final.corr(), cmap="YlGnBu", annot=True)


# #### Voice_mail_Plan and Voice_mail_messages have a Positive Correlation
# #### Total_charges and Total_mins have a Positive Correlation

# In[85]:


# Dropping Highly Correlated Columns
df_final.drop(columns=['voice_mail_messages','Total_mins'], inplace=True)


# In[86]:


# Checking shape are dropping the correlated columns
df_final.shape


# In[87]:


# Checking correlation after dropping highly correlated Datapoints.
df_final.corr()


# In[88]:


# Plotting a correlation plot after dropping highly correlated columns
sns.heatmap(df_final.corr(), cmap="YlGnBu", annot=True)


# #### Upon checking the Data found that the datapoints are independent

# ### Model Building

# In[89]:


# Creating X and y Variables
X = df_final.drop(columns=['churn'])
y = df_final.churn


# In[90]:


from sklearn.preprocessing import StandardScaler


# In[91]:


# Scaling the Data using Standard Scaler
Scaled = StandardScaler()


# In[92]:


X_scaled = Scaled.fit_transform(X)


# In[93]:


from sklearn.model_selection import train_test_split


# In[94]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=.20, shuffle=True, random_state=40)


# In[95]:


y_train.value_counts()


# In[96]:


y_test.value_counts()


# In[97]:


# Model Selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


# In[98]:


models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('SVM', SVC()))
models.append(('XGB', xgb.XGBClassifier()))
models.append(('RFC', RandomForestClassifier()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10)
    cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[99]:


# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# #### Building different Machine Learning has resulted in 
# #### 85% accuracy for Logistic Regression
# #### 86% accuracy for KNN
# #### 90% accuracy for DTC
# #### 86% accuracy for SVM
# #### 94% accuracy for XGBoost
# #### 94% accuracy for RandomForest

# ## Hyperparameter Tunning

# In[100]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


# In[101]:


classifier = xgb.XGBClassifier()


# In[102]:


params = {
 "learning_rate" : [0.05,0.10,0.15,0.20,0.25,0.30],
 "max_depth" : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma": [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
}


# In[103]:


rs_model=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,cv=5,verbose=3)


# In[104]:


rs_model.fit(X,y)


# In[105]:


rs_model.best_estimator_


# In[106]:


rs_model.cv_results_


# In[107]:


model = rs_model.best_estimator_


# In[108]:


model.fit(X_train, y_train)


# In[109]:


pred = model.predict(X_test)


# In[110]:


from sklearn.metrics import confusion_matrix, plot_confusion_matrix


# In[111]:


confusion_matrix(y_test, pred)


# In[112]:


plot_confusion_matrix(model, X_test, y_test)


# In[113]:


from sklearn.metrics import classification_report


# In[114]:


print(classification_report(y_test, pred))


# #### Building a model using Xgboost has resulted in 96% accuracy

# In[115]:


classifier1 = RandomForestClassifier()


# In[116]:


params = {
 "n_estimators" : [50,100,150,200,250,300],
 "criterion" : [ 'gini', 'entropy'],
 "max_depth" : [ 3,4,5,6,7],
 "min_samples_split": [ 2,3,4,5,6,7,8]
}


# In[117]:


rsv_model1=RandomizedSearchCV(classifier1, param_distributions= params, cv=10, verbose=5, return_train_score=True)


# In[118]:


rsv_model1.fit(X, y)


# In[119]:


rsv_model1.best_score_


# In[120]:


rsv_model1.cv_results_


# In[121]:


rsv_model1.best_estimator_


# In[122]:


model_RFC = rsv_model1.best_estimator_


# In[123]:


model_RFC.fit(X_train, y_train)


# In[124]:


pred1 = model_RFC.predict(X_test)


# In[125]:


print(classification_report(y_test, pred))


# In[126]:


print(classification_report(y_test, pred1))


# In[127]:


import pickle


# In[128]:


with open('model_pickle', 'wb') as f:
    pickle.dump(model_RFC, f)


# In[129]:


with open('model_pickle', 'rb') as f:
    mp = pickle.load(f)

