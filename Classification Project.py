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
Variable 	Description
member_id                  -	unique ID assigned to each member
loan_amnt                  -	loan amount (in dollar) applied by the member
funded_amnt                -	loan amount (in dollar) sanctioned by the bank
funded_amnt_inv            -	loan amount (in dollar) sanctioned by the investors
term	                   -  term of loan (in months)
batch_enrolled             -	batch numbers allotted to members
int_rate                   -	interest rate (%) on loan
grade	                   -  grade assigned by the bank
sub_grade                  -	grade assigned by the bank
emp_title                  -	job / Employer title of member
emp_length                 -	employment length, where 0 means less than one year and 10 means ten or more years
home_ownership             -	status of home ownership
annual_inc                 -	annual income (in dollar) reported by the member
verification_status        -	status of income verified by the bank
pymnt_plan                 -	indicates if any payment plan has started against loan
desc                       -	loan description provided by member
purpose                    -	purpose of loan
title                      -	loan title provided by member
zip_code                   -	first three digits of area zipcode of member
addr_state                 -	living state of member
dti                        -	ratio of member's total monthly debt repayment excluding mortgage divided by self reported                                     monthly income                   
delinq_2yrs                -	number of 30+ days delinquency in past 2 years
inq_last_6mths             -	number of inquiries in last 6 months
mths_since_last_delinq     -	number of months since last delinq
mths_since_last_record     -	number of months since last public record
open_acc                   -	number of open credit line in member's credit line
pub_rec                    -	number of derogatory public records
revol_bal                  -	total credit revolving balance
revol_util                 -	amount of credit a member is using relative to revol_bal
total_acc                  -	total number of credit lines available in members credit line
initial_list_status        -	unique listing status of the loan - W(Waiting), F(Forwarded)
total_rec_int              -	interest received till date
total_rec_late_fee         -	Late fee received till date
recoveries                 -	post charge off gross recovery
collection_recovery_fee    -	post charge off collection fee
collections_12_mths_ex_med -	number of collections in last 12 months excluding medical collections
mths_since_last_major_derog-	months since most recent 90 day or worse rating
application_type           -	indicates when the member is an individual or joint
verification_status_joint  -	indicates if the joint members income was verified by the bank
last_week_pay              -	indicates how long (in weeks) a member has paid EMI after batch enrolled
acc_now_delinq             -	number of accounts on which the member is delinquent
tot_coll_amt               -	total collection amount ever owed
tot_cur_bal                -	total current balance of all accounts
total_rev_hi_lim           -	total revolving credit limit
loan_status                -	status of loan amount, 1 = Defaulter, 0 = Non Defaulters

# # Importing libraries

# In[1]:


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


# # Importing Dataset

# In[2]:


#importing of dataset 
df=pd.read_csv('C:\\Users\\NGS010\\Downloads\\sample_indessa.csv')


# In[3]:


#here head() method prints first five records of the dataset
df.head()


# In[4]:


#this shape method describes the rows and columns of the dataset
df.shape


# In[5]:


df.shape


# In[6]:


#so as we are having some of the special charecters in this dataset so to remove those special charecters we use this below method.so that our dataset doesnot have any special charecters. 
spec_chars = ["!",'"',"#","%","&","'","(",")",
              "*","+",",","-",".","/",":",";","<",
              "=",">","?","@","[","\\","]","^","_",
              "`","{","|","}","~","–"]
for char in spec_chars:
    df['desc'] = df['desc'].str.replace(char, ' ')
    df['emp_title']=df['emp_title'].str.replace(char,' ')


# In[7]:


#so after removing the special charecters we have some white spaces in those columns so to join them we use this below method
df['desc'] = df['desc'].str.split().str.join(" ")
df['emp_title']=df['emp_title'].str.split().str.join(" ")


# In[8]:


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
        missing_per=df[column].isnull().sum() * 100 / dflen

        if str(df[column].dtypes) in ['int64', 'float64']:

            infinite=((df[column]== np.inf) | (df[column] == -np.inf)).sum()
            infinite_per=round((infinite/dflen) * 100%2)



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


# In[9]:


#So here we can take the numeric and categoric of all the eda into a dataframe
numerical_df,categorical_df=data(df)


# In[10]:


#So here we display all the numeric EDA operations in this one code
numerical_df


# In[11]:


#So here we can display all the categoric EDA operations in one code.
categorical_df


# In[12]:


#so here we can find the correlation of all the columns
df.corr()


# # Data Visualization

# In[13]:


#is negative correlation means if one variables decreases the other also decreases.zero means we have no correla#So by drawing the correlation matrix we can see the relation between two variables.so in this if we are seeing positive correlation means two variables increases together.
plt.figure(figsize=(20,10))
sns.heatmap(round(df.corr(),2),annot=True,cmap='Blues')
plt.show()


# In[93]:


#By drawing the histogram ,'loan_amnt' count is firstly low and then go up and again goes up ,after that it decreases again.
plt.hist(data = df, x = 'loan_amnt')


# # Data Pre-Processing

# In[13]:


df.columns


# In[14]:


df.drop(df.columns[[0, 9, 31,42]], axis = 1, inplace = True)


# In[15]:


df


# In[16]:


#here we can know all the missing values in the dataset.
df.isnull().sum()


# In[17]:


df.drop(df.columns[[13, 21, 22,33,35]], axis = 1, inplace = True)


# In[18]:


df


# In[19]:


#Here we can see the all the datatypes of the columns
df.isnull().sum()


# In[20]:


df.dtypes


# In[21]:


#Here we can remove all the null values in the categoric columns by replacing mode.
cols = ["batch_enrolled", "emp_length","title"]
df[cols]=df[cols].fillna(df.mode().iloc[0])


# In[22]:


#so after doing that we can see all the null values have been removed by the "emp_title", "desc","verification_status_joint"
df.isnull().sum()


# In[23]:


cols = ["annual_inc", "delinq_2yrs","inq_last_6mths","open_acc","pub_rec","revol_util","total_acc","collections_12_mths_ex_med","acc_now_delinq","tot_coll_amt","total_rev_hi_lim"]
df[cols]=df[cols].fillna(df.mean().iloc[0])


# In[202]:


#So here we see all the 'Zeros' in the columns.
(df == 0).sum()


# In[203]:


df.drop(df.columns[[18,19, 21, 26,27,28,29,32,33]], axis = 1, inplace = True)


# In[204]:


df.shape


# In[205]:


#after that we can see we dont have any zero values in our dataset.
(df == 0).sum()


# In[206]:


#So here we are separating all the numeric and categoric from the dataframe.
num=df.select_dtypes(include=[np.number])
cat=df.select_dtypes(exclude=[np.number])
num


# In[207]:


cat


# In[208]:


cat.columns


# # Encoding

# In[209]:


#So here we are performing Encoding to convert categoric to numeric.And we are using Label Encoding.
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


# In[210]:


cat.head()


# In[211]:


cat.isnull().sum()


# # Outlier Analysis

# In[212]:


#So here we are doing outlier analysis to know if we are having any outliers in the numeric column.
fig = plt.figure(figsize =(10, 20))
plt.boxplot(num)
plt.show()


# In[213]:


num.shape


# In[214]:


#So here we are doing IQR to remove outliers
for columns in num.columns:
    Q1 = num[columns].quantile(0.25)
    Q3 = num[columns].quantile(0.75)
    IQR = Q3 - Q1
    print(columns, int(IQR))
    num[columns] = np.where(num[columns] >= (Q3 + 1.5 * IQR),num[columns].mean(),num[columns])
    num[columns] = np.where(num[columns] <= (Q1 - 1.5 * IQR),num[columns].mean(),num[columns])


# In[215]:


num.shape


# In[216]:


plt.boxplot(num)


# # Feature Selection

# In[217]:


#So here we are doing feature selection to the numeric columns.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(num)
st =scaler.transform(num)
print(st)


# In[218]:


sd=pd.DataFrame(data = st,columns =  ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate', 'annual_inc',
       'dti', 'open_acc', 'revol_bal', 'revol_util', 'total_acc',
       'total_rev_hi_lim', 'loan_status'])


# In[219]:


num.columns


# In[220]:


sd.isnull().sum()


# In[221]:


cat.head()


# In[222]:


sd = sd.reset_index()
sd.head()


# In[223]:


cat = cat.reset_index()
sd.head()


# In[224]:


#So after performing all the pre-processing techniques here we are merging both numeric and categoric to perform Train-Test split and classification.
x = pd.merge(cat, sd, left_index=True, right_index=True)


# In[225]:


x


# In[226]:


x.shape


# # Model Building

# In[227]:


label=df['loan_status']


# # Train-Test Split

# In[228]:


#Here we are doing train test split.
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,label,test_size=0.4,random_state=42)


# # Logistic Regression

# In[229]:


#Here we are performing Logistic regression.
from sklearn.linear_model import LogisticRegression


# In[230]:


lr = LogisticRegression()


# In[231]:


lr.fit(X_train,y_train)


# In[232]:


prediction = lr.predict(X_test)


# In[233]:


#Here we are performing classification report.
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(classification_report(y_test,prediction))


# In[234]:


#So here we are constructing confusion matrix.
cmat = confusion_matrix(y_test,prediction)
print('TN - True Negative {}'.format(cmat[0,0]))
print('FP - False Positive {}'.format(cmat[0,1]))
print('FN - False Negative {}'.format(cmat[1,0]))
print('TP - True Positive {}'.format(cmat[1,1]))
print('Accuracy Rate: {}'.format(np.divide(np.sum([cmat[0,0],cmat[1,1]]),np.sum(cmat))))
print('Misclassification Rate: {}'.format(np.divide(np.sum([cmat[0,1],cmat[1,0]]),np.sum(cmat))))
LR = cmat[1,1]/(cmat[1,1]+cmat[0,1])
LR_recall = cmat[1,1] / (cmat[1,1]+cmat[1,0])
LR_acc = np.divide(np.sum([cmat[0,0],cmat[1,1]]),np.sum(cmat))
LR_f1 = 2*((LR*LR_recall)/(LR+LR_recall))


# In[235]:


print(cmat)


# # Decision Tree Classifier

# In[236]:


#So here we are performing decision tree.
from sklearn.tree import DecisionTreeClassifier


# In[237]:


dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)


# In[238]:


y_predict = dtree.predict(X_test)


# In[239]:


from sklearn.metrics import confusion_matrix, classification_report


# In[240]:


print(classification_report(y_test, y_predict))


# In[241]:


cmat = confusion_matrix(y_test,y_predict)
print('TN - True Negative {}'.format(cmat[0,0]))
print('FP - False Positive {}'.format(cmat[0,1]))
print('FN - False Negative {}'.format(cmat[1,0]))
print('TP - True Positive {}'.format(cmat[1,1]))
print('Accuracy Rate: {}'.format(np.divide(np.sum([cmat[0,0],cmat[1,1]]),np.sum(cmat))))
print('Misclassification Rate: {}'.format(np.divide(np.sum([cmat[0,1],cmat[1,0]]),np.sum(cmat))))
Decision = cmat[1,1]/(cmat[1,1]+cmat[0,1])
Decision_recall = cmat[1,1] / (cmat[1,1]+cmat[1,0])
Decision_acc = np.divide(np.sum([cmat[0,0],cmat[1,1]]),np.sum(cmat))
Decision_f1 = 2*((Decision*Decision_recall)/(Decision+Decision_recall))


# In[242]:


print(confusion_matrix(y_test,y_predict))


# # Random Forest Classifier

# In[247]:


#here we are performing random forest classifier
from sklearn.ensemble import RandomForestClassifier


# In[248]:


randomforest = RandomForestClassifier(n_estimators = 200)
randomforest.fit(X_train, y_train)


# In[251]:


predictor = randomforest.predict(X_test)


# In[253]:


print(classification_report(y_test,predictor))


# In[256]:


cmat = confusion_matrix(y_test,predictor)
print('TN - True Negative {}'.format(cmat[0,0]))
print('FP - False Positive {}'.format(cmat[0,1]))
print('FN - False Negative {}'.format(cmat[1,0]))
print('TP - True Positive {}'.format(cmat[1,1]))
print('Accuracy Rate: {}'.format(np.divide(np.sum([cmat[0,0],cmat[1,1]]),np.sum(cmat))))
print('Misclassification Rate: {}'.format(np.divide(np.sum([cmat[0,1],cmat[1,0]]),np.sum(cmat))))
Random = cmat[1,1]/(cmat[1,1]+cmat[0,1])
Random_recall = cmat[1,1] / (cmat[1,1]+cmat[1,0])
Random_acc = np.divide(np.sum([cmat[0,0],cmat[1,1]]),np.sum(cmat))
Random_f1 = 2*((Random*Random_recall)/(Random+Random_recall))


# In[257]:


print(confusion_matrix(y_test, predictor))


# # Naive Bayes

# In[258]:


#So here we are performing naive bayes.
from sklearn.naive_bayes import GaussianNB


# In[259]:


gaussian = GaussianNB()
y_pred = gaussian.fit(X_train, y_train).predict(X_test)


# In[260]:


y_pred = gaussian.predict(X_test)


# In[261]:


from sklearn.metrics import classification_report, confusion_matrix


# In[262]:


print(classification_report(y_test, y_pred))


# In[263]:


cmat = confusion_matrix(y_test,y_pred)
print('TN - True Negative {}'.format(cmat[0,0]))
print('FP - False Positive {}'.format(cmat[0,1]))
print('FN - False Negative {}'.format(cmat[1,0]))
print('TP - True Positive {}'.format(cmat[1,1]))
print('Accuracy Rate: {}'.format(np.divide(np.sum([cmat[0,0],cmat[1,1]]),np.sum(cmat))))
print('Misclassification Rate: {}'.format(np.divide(np.sum([cmat[0,1],cmat[1,0]]),np.sum(cmat))))
NB = cmat[1,1]/(cmat[1,1]+cmat[0,1])
NB_recall = cmat[1,1] / (cmat[1,1]+cmat[1,0])
NB_acc = np.divide(np.sum([cmat[0,0],cmat[1,1]]),np.sum(cmat))
NB_f1 = 2*((NB*NB_recall)/(NB+NB_recall))


# In[264]:


print(confusion_matrix(y_test, y_pred))


# In[265]:


all_models = pd.DataFrame({'Model_Name':['LogisticRegression','DecisionTreeClassifier','RandomForestModel','Naive Byes'],'Accuracy':[LR_acc,Decision_acc,Random_acc,NB_acc],'Precision':[LR,Decision,Random,NB],'ReCall':[LR_recall,Decision_recall,Random_recall,NB_recall],'F1_Score':[LR_f1,Decision_f1,Random_f1,NB_f1]})


# In[266]:


all_models


# In[ ]:





# # Conclusion
So after comparing all the classification techniques ranndom forest shows good accuracy rate.So we can take random forest as the
best classification for this dataset.
# In[ ]:




