import pandas as pd  

import numpy as np 

import matplotlib.pyplot as plt 

from sklearn.feature_extraction.text import TfidfVectorizer 

from sklearn.ensemble import RandomForestClassifier 

from sklearn import metrics 

import itertools 

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score 

from sklearn.naive_bayes import MultinomialNB 

from sklearn.feature_extraction.text import CountVectorizer 

df=pd.read_csv('.../Stock_Dataa.csv', encoding="ISO-8859-1") 

train=df[df['Date']<'20150101'] 

test=df[df['Date']>'20141231'] 

train.shape 

data=train.iloc[:,2:27] 

data.replace("[^a-zA-Z]", " ",regex=True, inplace=True) 

data.columns 

for col in data.columns: 

     data[col]=data[col].str.lower() 

data.head(1) 

headlines = [] 

for row in range(0,len(data.index)):      

headlines.append(' '.join(str(x) for x in data.iloc[row,0:25])) 

tfvector=TfidfVectorizer(ngram_range=(2,3)) 

train_df=tfvector.fit_transform(headlines) 

randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')randomclassifier.fit(train_df,train['Label']) 

def plot_confusion_matrix(cm, classes,                          normalize=False,                          title='Confusion matrix',                          cmap=plt.cm.Blues):  

       plt.imshow(cm, interpolation='nearest', cmap=cmap)     

       plt.title(title) 

       Plt.colorbar() 

       tick_marks = np.arange(len(classes)) 

       plt.xtickstick_marks, classes, rotation=45) 

       pltyticks(tick_marks, classes)     

       if normalize: 

             cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 

             print("Normalized confusion matrix") 

      else: 

             print('Confusion matrix, without normalization') 

      thresh = cm.max()  

       for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): 

              plt.textj, i, cm[i, j], 

                    horizontalalignment"center", 

                    color="white" if cm[i, j] > thresh else "black") 

         plttight_layout() 

         plt.ylabel('True label') 

         plt.xlabel('Predicted label') 

test_transform= [] 

for row in range(0,len(test.index)): 

       test_transform.append' '.join(str(x) for x in test.iloc[row,2:27])) 

test_dataset = tfvector.transform(test_transform) 

predictions = randomclassifier.predict(test_dataset) 

matrix=confusion_matrix(test['Label'],predictions) 

print(matrix) 

score=accuracy_score(test['Label'],predictions) 

print(score) 

report=classification_report(test['Label'],predictions) 

print(report) 

plot_confusion_matrix(matrix, classes=['Down', 'Up']) 

nb=MultinomialNB() 

nb.fit(train_df,train['Label']) 

predictions = nb.predict(test_dataset)matrix=confusion_matrix(test['Label'],predictions) 

print(matrix) 

score=accuracy_score(test['Label'],predictions) 

print(score) 

report=classification_report(test['Label'],predictions) 

print(report) 

plot_confusion_matrix(matrix, classes=['Down', 'Up']) 

bow=CountVectorizer(ngram_range=(2,3)) 

train_df=bow.fit_transform(headlines) 

randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy') 

randomclassifier.fit(train_df,train['Label'])   

predictions = randomclassifier.predict(test_dataset)matrix=confusion_matrix(test['Label'],predictions) 

print(matrix) 

score=accuracy_score(test['Label'],predictions) 

print(score) 

report=classification_report(test['Label'],predictions) 

print(report) 

nb=MultinomialNB() 

nb.fit(train_df,train['Label']) 

predictions = nb.predict(test_dataset)matrix=confusion_matrix(test['Label'],predictions) 

print(matrix) 

score=accuracy_score(test['Label'],predictions) 

print(score) 

report=classification_report(test['Label'],predictions) 

print(report) 

plot_confusion_matrix(matrix, classes=['Down', 'Up']) 

