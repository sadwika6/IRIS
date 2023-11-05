import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

data=pd.read_csv(r'iris.csv')

print(data.shape)
#print(data)
print(data.head())

#print(data.isna())
print(data.isna().sum())
#print(data.describe())

# data['pl'].fillna((data['pl']).mode()[0],inplace=True)

count =  data.Class.value_counts()
print(count)

lab = data.Class.unique().tolist()
print(lab)

# plt.pie(count,labels=lab)
# plt.title("Count of Species",fontsize=20)
# plt.show()

x=np.array(data.iloc[:,[0,1,2,3]])
y=data.iloc[:,-1].values 
print(x,y)
from sklearn.model_selection import train_test_split

model = KNeighborsClassifier(n_neighbors=3)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)
model.fit(x_train,y_train)

import pickle
model_filename = "iris_model.pkl"
with open (model_filename, 'wb') as model_file:
    pickle.dump(model,model_file)

with open(model_filename,'rb') as model_file:
    loaded_model  = pickle.load(model_file)

res = loaded_model.predict(x_test)
print(res)

predictions = model.predict(x_test)
print(classification_report(y_test,predictions))
print("Confusion Matrix\n",confusion_matrix(y_test,predictions))

print(accuracy_score(y_test,predictions))
