# Loan-Prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


df = pd.read_csv("C:\\Users\\91986\\Desktop\\projects\\loan Eligibility Prediction ,python project for data science\\loan predictive csv\\Loan.csv")

df.head()

df.shape

df.info()

df.describe()

pd.crosstab(df['Credit_History'], df['Loan_Status'], margins=True)

df.boxplot(column= 'ApplicantIncome')

df['ApplicantIncome'].hist(bins=20)

df['CoapplicantIncome'].hist(bins=20)

df.boxplot(column='ApplicantIncome', by= 'Education')

df.boxplot(column= 'LoanAmount')

df['LoanAmount'].hist(bins=20)

df['LoanAmount_log']=np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)

df.isnull().sum()


df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Married'].fillna(df['Married'].mode()[0],inplace=True)

df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)

df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)

df.LoanAmount = df.LoanAmount.fillna(df.LoanAmount.mean())
df.LoanAmount_log = df.LoanAmount.fillna(df.LoanAmount_log.mean())

df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace=True)

df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)

df.isnull().sum()


df['TotalIncome']=df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log']=np.log(df['TotalIncome'])

df['TotalIncome_log'].hist(bins=20)

df.head()

x= df.iloc[:,np.r_[1:5,9:11,13:15]].values
y= df.iloc[:,12].values

x
y

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print(x_train)

from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()

for i in range(0,5): 
    x_train[:,i]=labelencoder_x.fit_transform(x_train[:,i])


x_train[:,7]= labelencoder_x.fit_transform(x_train[:,7])

x_train

labelencoder_y=LabelEncoder()
y_train= labelencoder_y.fit_transform(y_train)

y_train

for i in range(0,5): 
    x_test[:,i]=labelencoder_x.fit_transform(x_test[:,i])


x_test[:,7]= labelencoder_x.fit_transform(x_test[:,7])

labelencoder_y=LabelEncoder()
y_train= labelencoder_y.fit_transform(y_train)

x_test

y_test

from sklearn.preprocessing import StandardScaler 
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.fit_transform(x_test)




Applying algorithm -> 

#Decision tree 

from sklearn.tree import DecisionTreeClassifier
DTClassifier= DecisionTreeClassifier(criterion='entropy',random_state=0)
DTClassifier.fit(x_train,y_train)

y_pred= DTClassifier.predict(x_test)
y_pred


from sklearn import metrics
print('The accuracy of decision tree is:', metrics.accuracy_score(y_pred,y_test)) 

#Navie_bayes 

from sklearn.naive_bayes import GaussianNB
NBClassifier = GaussianNB()
NBClassifier.fit(x_train,y_train)


y_pred= NBClassifier.predict(x_test)

y_pred



^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

project 2 -> loan prediction 

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm


df = pd.read_csv("C:\\Users\\91986\\Desktop\\projects\\loan Eligibility Prediction ,python project for data science\\loan predictive csv\\Loan.csv")

df.head()

df.info()

df.isnull().sum()

df['loanAmount_log']= np.log(df['LoanAmount'])
df['loanAmount_log'].hist(bins=20)    

df.isnull().sum()

df['TotalIncome']= df['ApplicantIncome']+ df['CoapplicantIncome']
df['TotalIncome_log']= np.log(df['TotalIncome'])
df['TotalIncome_log'].hist(bins=20)

df['Gender'].fillna(df['Gender'].mode()[0], inplace = True)
df['Married'].fillna(df['Married'].mode()[0], inplace = True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace = True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace = True)

df.LoanAmount = df.LoanAmount.fillna(df.LoanAmount).mean()
df.loanAmount_log = df.loanAmount_log.fillna(df.loanAmount_log.mean()) 

df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace = True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace = True)

df.isnull().sum()

x= df.iloc[:,np.r_[1:5,9:11,13:15]].values
y= df.iloc[:,12].values

x

y

print("per of missing gender is %2f%%" %((df['Gender'].isnull().sum()/df.shape[0])*100))

print("number of people who take loan as group by gender:")
print(df['Gender'].value_counts())
sns.countplot(x='Gender', data=df,palette= 'Set1')

print("number of people who take loan as group by marital status:")
print(df['Married'].value_counts())
sns.countplot(x='Married', data=df,palette= 'Set1')

print("number of people who take loan as group by dependents:")
print(df['Dependents'].value_counts())
sns.countplot(x='Dependents', data=df,palette= 'Set1')

print("number of people who take loan as group by self employed:")
print(df['Self_Employed'].value_counts())
sns.countplot(x='Self_Employed', data=df,palette= 'Set1')

print("number of people who take loan as group by self Loanamount:")
print(df['LoanAmount'].value_counts())
sns.countplot(x='LoanAmount', data=df, palette= 'Set1')

print("number of people who take loan as group by self Credit history:")
print(df['Credit_History'].value_counts())
sns.countplot(x='Credit_History', data=df, palette= 'Set1')

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2,random_state=0)

from sklearn.preprocessing import LabelEncoder
Labelencoder_x = LabelEncoder()


for i  in range(0, 5):
    
    X_train[:,i]= Labelencoder_x.fit_transform(X_train[:,i])
    X_train[:,7]= Labelencoder_x.fit_transform(X_train[:,7])
    
X_train

for i  in range(0, 5):
    
    X_train[:,i]= Labelencoder_x.fit_transform(X_train[:,i])
    X_train[:,7]= Labelencoder_x.fit_transform(X_train[:,7])
    
X_train


Labelencoder_y = LabelEncoder()
y_train = Labelencoder_y.fit_transform(y_train)

y_train

for i in range(0,5):
    X_test[:,i]= Labelencoder_x.fit_transform(X_test[:,i]) 
    X_test[:,7] = Labelencoder_x.fit_transform(X_test[:,7])
    
X_test    

labelencoder_y = LabelEncoder() 

y_test= Labelencoder_y.fit_transform(y_test)

y_test 



from sklearn.preprocessing import StandardScaler

ss = StandardScaler() 
X_train = ss.fit_transform(X_train)
x_test = ss.fit_transform(X_test)


used algorithm for prediction 

from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier()  
rf_clf.fit(X_train, y_train)



from sklearn import metrics 
y_pred = rf_clf.predict(x_test)

print("acc of random forest clf is", metrics.accuracy_score(y_pred, y_test)) 

y_pred


from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)


y_pred = nb_clf.predict(X_test)
print("acc of navie bayes is % ", metrics.accuracy_score(y_pred, y_test) )

y_pred


from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)

y_pred

from sklearn.neighbors import KNeighborsClassifier
kn_clf = KNeighborsClassifier()
kn_clf.fit(X_train, y_train)


y_pred = kn_clf.predict(X_test)
print("acc of kn is % ", metrics.accuracy_score(y_pred, y_test) )







