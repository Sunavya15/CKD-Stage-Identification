# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
#Reading the dataset
kidney = pd.read_csv("kidney_disease.csv")
kidney.head()
# Information about the dataset
kidney.info()
# Description of the dataset
kidney.describe()
# To see what are the column names in our dataset
print(kidney.columns)
# Mapping the text to 1/0 and cleaning the dataset
kidney[['htn','dm','cad','pe','ane']] = kidney[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0})
kidney[['rbc','pc']] = kidney[['rbc','pc']].replace(to_replace={'abnormal':1,'normal':0})
kidney[['pcc','ba']] = kidney[['pcc','ba']].replace(to_replace={'present':1,'notpresent':0})
kidney[['appet']] = kidney[['appet']].replace(to_replace={'good':1,'poor':0,'no':np.nan})
kidney['classification'] = kidney['classification'].replace(to_replace={'ckd':1.0,'ckd\t':1.0,'notckd':0.0,'no':0.0})
kidney.rename(columns={'classification':'class'},inplace=True)

kidney['pe'] = kidney['pe'].replace(to_replace='good',value=0) # Not having pedal edema is good
kidney['appet'] = kidney['appet'].replace(to_replace='no',value=0)
kidney['cad'] = kidney['cad'].replace(to_replace='\tno',value=0)
kidney['dm'] = kidney['dm'].replace(to_replace={'\tno':0,'\tyes':1,' yes':1, '':np.nan})
kidney.drop('id',axis=1,inplace=True)
kidney.head()
# This helps us to count how many NaN are there in each column
len(kidney)-kidney.count()
# This shows number of rows with missing data
kidney.isnull().sum(axis = 1)
#This is a visualization of missing data in the dataset
sns.heatmap(kidney.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# This shows number of complete cases and also removes all the rows with NaN
kidney2 = kidney.dropna()
print(kidney2.shape)
# Now our dataset is clean
sns.heatmap(kidney2.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.heatmap(kidney2.corr())
# Counting number of normal vs. abnormal red blood cells of people having chronic kidney disease
print(kidney2.groupby('rbc').rbc.count().plot(kind="bar"))
#This plot shows the patient's sugar level compared to their ages
kidney2.plot(kind='scatter', x='age',y='su');
#plt.show()
# Shows the maximum blood pressure having chronic kidney disease
print(kidney2.groupby('class').bp.max())
print(kidney2['dm'].value_counts(dropna=False))
X_train, X_test, y_train, y_test = train_test_split(kidney2.iloc[:,:-1], kidney2['class'], test_size=0.33, random_state=44, stratify= kidney2['class'])
print(X_train.shape)
y_train.value_counts()

print("RANDOM FOREST CLASSFIER")

rfc = RandomForestClassifier(random_state = 22)
rfc_fit = rfc.fit(X_train,y_train)
rfc_pred = rfc_fit.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))
accuracy_score( y_test, rfc_pred)

print("ACCURACY")
print(accuracy_score( y_test, rfc_pred)*100)


print("SVM CLASSFIER")

svm = SVC()
svm_fit = svm.fit(X_train,y_train)
svm_pred = svm_fit.predict(X_test)
print(confusion_matrix(y_test,svm_pred))
print(classification_report(y_test,svm_pred))
accuracy_score( y_test, svm_pred)

print("ACCURACY")
print(accuracy_score( y_test, svm_pred)*100)



print("KNeighborsClassifier")

knn = KNeighborsClassifier(n_neighbors=1)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',metric_params=None, n_jobs=None, n_neighbors=1, p=2,weights='uniform')
knn.fit(X_train,y_train)

pred = knn.predict(X_test)

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
accuracy_score( y_test,pred)

print("ACCURACY")
print(accuracy_score( y_test,pred)*100)

print("LogisticRegression")

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
accuracy_score( y_test, predictions)


print("ACCURACY")
print(accuracy_score( y_test, predictions)*100)

