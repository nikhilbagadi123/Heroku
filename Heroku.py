#!/usr/bin/env python
# coding: utf-8

# # Problem Statement
The Bank Indessa has not done well in last 3 quarters. Their NPAs (Non Performing Assets) have reached
all time high. It is starting to lose confidence of its investors. As a result, it’s stock has fallen by 20% in the
previous quarter alone.

After careful analysis, it was found that the majority of NPA was contributed by loan defaulters. With the
messy data collected over all the years, this bank has decided to use machine learning to figure out a way
to find these defaulters and devise a plan to reduce them.

This bank uses a pool of investors to sanction their loans. For example: If any customer has applied for a
loan of $20000, along with bank, the investors perform a due diligence on the requested loan application.
Keep this in mind while understanding data.
# # Description

# In[ ]:


Variable                       Description

member_id                  -unique ID assigned to each member
loan_amnt                  -loan amount (in dollar) applied by the member
funded_amnt                -loan amount (in dollar) sanctioned by the bank
funded_amnt_inv            -loan amount (in dollar) sanctioned by the investors
term	                   - term of loan (in months)
batch_enrolled             -batch numbers allotted to members
int_rate                   -interest rate (%) on loan
grade	                   - grade assigned by the bank
sub_grade                  -grade assigned by the bank
emp_title                  -job / Employer title of member
emp_length                 -employment length, where 0 means less than one year and 10 means ten or more years
home_ownership             -status of home ownership
annual_inc                 -annual income (in dollar) reported by the member
verification_status        -status of income verified by the bank
pymnt_plan                 -indicates if any payment plan has started against loan
desc                       -loan description provided by member
purpose                    -purpose of loan
title                      -loan title provided by member
zip_code                   -first three digits of area zipcode of member
addr_state                 -living state of member
dti                        -ratio of member's total monthly debt repayment excluding mortgage divided by self reported                                     monthly income                   
delinq_2yrs                -number of 30+ days delinquency in past 2 years
inq_last_6mths             -number of inquiries in last 6 months
mths_since_last_delinq     -number of months since last delinq
mths_since_last_record     -number of months since last public record
open_acc                   -number of open credit line in member's credit line
pub_rec                    -number of derogatory public records
revol_bal                  -total credit revolving balance
revol_util                 -amount of credit a member is using relative to revol_bal
total_acc                  -total number of credit lines available in members credit line
initial_list_status        -unique listing status of the loan - W(Waiting), F(Forwarded)
total_rec_int              -interest received till date
total_rec_late_fee         -Late fee received till date
recoveries                 -post charge off gross recovery
collection_recovery_fee    -post charge off collection fee
collections_12_mths_ex_med -number of collections in last 12 months excluding medical collections
mths_since_last_major_derog-months since most recent 90 day or worse rating
application_type           -indicates when the member is an individual or joint
verification_status_joint  -indicates if the joint members income was verified by the bank
last_week_pay              -indicates how long (in weeks) a member has paid EMI after batch enrolled
acc_now_delinq             -number of accounts on which the member is delinquent
tot_coll_amt               -total collection amount ever owed
tot_cur_bal                -total current balance of all accounts
total_rev_hi_lim           -total revolving credit limit
loan_status                -status of loan amount, 1 = Defaulter, 0 = Non Defaulters


# In[83]:


#importing files 
import warnings 
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from statsmodels import robust
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy import stats


# In[84]:


#importing of dataset 
df=pd.read_csv('C:\\Users\\NGS010\\Downloads\\sample_indessa.csv')


# In[85]:


#here head() method prints first five records of the dataset
df.head()


# In[86]:


#so as we are having some of the special charecters in this dataset so to remove those special charecters we use this below method.so that our dataset doesnot have any special charecters. 
spec_chars = ["!",'"',"#","%","&","'","(",")",
              "*","+",",","-",".","/",":",";","<",
              "=",">","?","@","[","\\","]","^","_",
              "`","{","|","}","~","–"]
for char in spec_chars:
    df['desc'] = df['desc'].str.replace(char, ' ')
    df['emp_title']=df['emp_title'].str.replace(char,' ')


# In[87]:


#so after removing the special charecters we have some white spaces in those columns so to join them we use this below method
df['desc'] = df['desc'].str.split().str.join(" ")
df['emp_title']=df['emp_title'].str.split().str.join(" ")


# In[4]:


df.columns


# In[5]:


df.head()


# In[6]:


#here we are taking one of the function to perform EDA for the dataset.
def data(df):
#So here we can take all the unique,missing,infinite,zeros,min,max,mean,std,varience,coefficient of deviation,mean absolute deviation,kurtosis,skewness,quartiles,monotocity of all the numeric columns
    numerical_df = pd.DataFrame(columns = ["col",'Distnict','Distnict%','Missing','Missing%','Infinite','Infinite%','Zeros','Zeros%','Maximum','Minimum','STD','Variance','COD','MAD','Skewness','Kurtosis','5th Percentile','Q1','Median','Q3','95th percentile','IQR','Monotonic'])
#So here we can take all the unique,missing,frequency,minstringlength,maxlengthstring of all the categoric columns
    categorical_df=pd.DataFrame(columns=["col",'Distnict','Distnict%','Missing','Missing%','Frequency','MinStringLength','MaxStringLength'])
    for column in df.columns:
        distnct = df[column].nunique()
        dflen = len(df[column])
        distnct_per = round(((df[column].nunique())/dflen) * 100, 2)

        missing = df[column].isnull().sum()
        missing_per =df[column].isnull().sum()*100

        if str(df[column].dtypes) in ['int64', 'float64']:

            infinite=((df[column]== np.inf) | (df[column] == -np.inf)).sum()
            infinite_per=round((infinite/dflen) * 100, 2)



            zeros = (df[column]==0).sum()
            dflen = len(df[column])
            mean = sum(df[column])/dflen
            zeros_per = round(((df[column] == 0).astype(int).sum(axis=0)/dflen)*100,2)
            maxValues = df[column].max()
            minValues = df[column].min()
            standardDevation = df[column].std()
            variance = df[column].var()
            cv = standardDevation/mean if mean else np.NaN
            mad = stats.median_absolute_deviation(df[column])
            skew = df[column].skew(axis = 0, skipna = True)
            kurto = df[column].kurtosis(axis = 0, skipna = True)
            fifth_th_per = df[column].quantile(0.05)
            Q1 = df[column].quantile(0.25)
            median = df[column].median(axis = 0)
            Q3 = df[column].quantile(0.25)
            iqr = Q3-Q1
            NintyFifth_th_per = df[column].quantile(0.95)
            if df[column].is_monotonic:
                monotonic = 'monotonic'
            else:
                monotonic = 'no monotonic'
            numerical_df = numerical_df.append({'col': column ,'Distnict':distnct,'Distnict%':distnct_per,'Missing':missing,'Missing%': missing_per,'Infinite':infinite,'Infinite%':infinite_per,'Zeros':zeros,'Zeros%':zeros_per,'Maximum':maxValues,'Minimum':minValues,'STD':standardDevation,'Variance':variance,'COD':cv,'MAD':mad,'Skewness':skew,'Kurtosis':kurto,'5th Percentile':fifth_th_per,'Q1':Q1,'Median':median,'Q3':Q3,'95th percentile':NintyFifth_th_per,'IQR':iqr,'Monotonic':monotonic},ignore_index=True)
        else:
            freq = df[column].mode()
            length = np.vectorize(len)
            MaxStrLen = df[column].astype(str).map(len).max()
            MinStrLen = df[column].astype(str).map(len).min()
            categorical_df = categorical_df.append({"col":column,'Distnict':distnct,'Distnict%':distnct_per,'Missing':missing,'Missing%':missing_per,'Frequency':freq,'MinStringLength':MaxStrLen,'MaxStringLength':MinStrLen},ignore_index=True)
    numerical_df= numerical_df.set_index('col')
    categorical_df= categorical_df.set_index('col')
    return numerical_df,categorical_df


# In[7]:


#So here we can take the numeric and categoric of all the eda into a dataframe
numerical_df,categorical_df=data(df)


# In[8]:


#So here we display all the numeric EDA operations in this one code
numerical_df


# In[9]:


#So here we can display all the categoric EDA operations in one code.
categorical_df


# In[10]:


#so here we can find the correlation of all the columns
df.corr()


# # Data Visualization

# In[12]:


plt.figure(figsize=(20,10))
sns.heatmap(round(df.corr(),2),annot=True,cmap='Blues')
plt.show()


# In[11]:


plt.hist(data = df, x = 'loan_amnt')


# In[12]:


df.drop(df.columns[[0, 9, 31,42]], axis = 1, inplace = True)


# In[13]:


df


# In[14]:


df.isnull().sum()


# In[15]:


df.drop(df.columns[[13, 21, 22,33,35]], axis = 1, inplace = True)


# In[16]:


df


# In[17]:


df.isnull().sum()


# In[18]:


cols = ["batch_enrolled", "emp_length","title"]
df[cols]=df[cols].fillna(df.mode().iloc[0])


# In[19]:


df.isnull().sum()


# In[20]:


cols = ["annual_inc", "delinq_2yrs","inq_last_6mths","open_acc","pub_rec","revol_util","total_acc","collections_12_mths_ex_med","acc_now_delinq","tot_coll_amt","total_rev_hi_lim"]
df[cols]=df[cols].fillna(df.mean().iloc[0])


# In[21]:


(df == 0).sum()


# In[22]:


df.drop(df.columns[[18,19, 21, 26,27,28,29,32,33]], axis = 1, inplace = True)


# In[23]:


df.shape


# In[24]:


(df == 0).sum()


# In[25]:


num=df.select_dtypes(include=[np.number])
cat=df.select_dtypes(exclude=[np.number])
num


# In[26]:


cat


# In[27]:


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
cat['term']    = label.fit_transform(cat['term'])
cat['batch_enrolled'] = label.fit_transform(cat['batch_enrolled'])
cat['grade'] = label.fit_transform(cat['grade'])
cat['sub_grade']    = label.fit_transform(cat['sub_grade'])
cat['emp_length'] = label.fit_transform(cat['emp_length'])
cat['home_ownership'] = label.fit_transform(cat['home_ownership'])
cat['verification_status']    = label.fit_transform(cat['verification_status'])
cat['pymnt_plan'] = label.fit_transform(cat['pymnt_plan'])
cat['purpose'] = label.fit_transform(cat['purpose'])
cat['title']    = label.fit_transform(cat['title'])
cat['zip_code'] = label.fit_transform(cat['zip_code'])
cat['addr_state'] = label.fit_transform(cat['addr_state'])
cat['initial_list_status']    = label.fit_transform(cat['initial_list_status'])
cat['application_type'] = label.fit_transform(cat['application_type'])
cat['last_week_pay'] = label.fit_transform(cat['last_week_pay'])


# In[28]:


cat.head()


# In[29]:


cat.isnull().sum()


# In[30]:


fig = plt.figure(figsize =(10, 20))
plt.boxplot(num)
plt.show()


# In[31]:


num.shape


# In[32]:


for columns in num.columns:
    Q1 = num[columns].quantile(0.25)
    Q3 = num[columns].quantile(0.75)
    IQR = Q3 - Q1
    print(columns, int(IQR))
    num[columns] = np.where(num[columns] >= (Q3 + 1.5 * IQR),num[columns].mean(),num[columns])
    num[columns] = np.where(num[columns] <= (Q1 - 1.5 * IQR),num[columns].mean(),num[columns])


# In[33]:


num.shape


# In[34]:


plt.boxplot(num)


# In[35]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(num)
st =scaler.transform(num)
print(st)


# In[36]:


sd=pd.DataFrame(data = st,columns =  ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate', 'annual_inc',
       'dti', 'open_acc', 'revol_bal', 'revol_util', 'total_acc',
       'total_rev_hi_lim', 'loan_status'])


# In[37]:


sd.isnull().sum()


# In[38]:


sd = sd.reset_index()
sd.head()


# In[55]:


cat = cat.reset_index()
cat.head()


# In[40]:


x = pd.merge(cat, sd, left_index=True, right_index=True)
x


# In[41]:


x.shape


# In[96]:


label=df['loan_status']


# In[97]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,label,test_size=0.4,random_state=42)


# In[98]:


from sklearn import linear_model


# In[99]:


lr = linear_model.LinearRegression()


# In[100]:


lr.fit(X_train,y_train)


# In[75]:


lr.coef_


# In[76]:


lr.intercept_


# In[89]:


lr.predict(X_test)


# In[90]:


import pickle


# In[91]:


with open('model_pkl', 'wb') as files:
    pickle.dump(lr, files)


# In[92]:


with open('model_pkl' , 'rb') as f:
    model = pickle.load(f)


# In[93]:


lr.predict(X_test)


# In[ ]:




