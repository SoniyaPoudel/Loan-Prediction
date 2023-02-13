import numpy as np # importing necessary libaries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('Training Dataset.csv')
df.head()
df.info()
df.describe()
df.isna().sum() # Lets check if there is any null value in the data
# Remove the null value 
df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Married'].fillna(df['Married'].mode()[0],inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].mode()[0],inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace=True)

df.isna().sum()
df=df.drop(['Loan_ID'],axis=1)

plt.figure(figsize=(20,10))
plt.figure(1)
plt.subplot(221)
plt.title('Gender')
sns.countplot(x='Gender',data=df,hue='Gender')
plt.subplot(222)
plt.title('Married')
sns.countplot(x='Married',data=df,hue='Married')
plt.subplot(223)
plt.title('Dependents')
sns.countplot(x='Dependents',data=df,hue='Dependents')
plt.subplot(224)
plt.title('Self_Employed')
sns.countplot(x='Self_Employed',data=df,hue='Self_Employed')
testdata=pd.read_csv('test.csv')

plt.figure(1)
plt.subplot(121)
sns.distplot(df["ApplicantIncome"]);

plt.subplot(122)
df["ApplicantIncome"].plot.box(figsize=(16,5))
plt.show()
df.boxplot(column='ApplicantIncome',by="Education" )
plt.suptitle(" ")
plt.show()

#Let us see the relationship between the gender and Loan status
print(pd.crosstab(df["Gender"],df["Loan_Status"]))
Gender = pd.crosstab(df["Gender"],df["Loan_Status"])
Gender.div(Gender.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("Gender")
plt.ylabel("Percentage")
plt.show()

df["TotalIncome"]=df["ApplicantIncome"]+df["CoapplicantIncome"]
testdata["TotalIncome"]=testdata["ApplicantIncome"]+testdata["CoapplicantIncome"]

df=df.drop(["ApplicantIncome","CoapplicantIncome"],axis=1)
X=df.drop("Loan_Status",1)
y=df[["Loan_Status"]]
testdata=testdata.drop(["ApplicantIncome","CoapplicantIncome","Loan_ID"],axis=1)
X = pd.get_dummies(X)
X.head()
df=pd.get_dummies(df)
testdata=pd.get_dummies(testdata)
from sklearn.model_selection import train_test_split
x_train,x_cv,y_train,y_cv=train_test_split(X,y,test_size=0.3,random_state=1)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logistic_model = LogisticRegression(random_state=1)
logistic_model.fit(x_train,y_train)
pred_cv_logistic=logistic_model.predict(x_cv)
score_logistic =accuracy_score(pred_cv_logistic,y_cv)*100 
score_logistic
