import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df=pd.read_csv("creditcard.csv")
#if there are empty cells check and remove
#print(df.isnull().sum().sum())
#df=df.dropna()

x=df.drop(columns=["Class"])
y=df["Class"]
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=0)

model=RandomForestClassifier(n_estimators=100,random_state=0)
model.fit(x_train,y_train)
prediction=model.predict(x_test)
print("\n",accuracy_score(y_test,prediction))
print("\n classification report:",classification_report(y_test,prediction,target_names=["not fraud","fraud"]))

new_data=pd.DataFrame([{
   "Time":0,
    "V1" :-0.021053053,
    "V2" :-1.01545471,
    "V3" :0.324504731,
    "V4":-0.058132823,
    "V5":0.40399296,
    "V6":-0.470400525,
    "V7":-0.65311889,
    "V8":1.505617082,
    "V9":0.525668092,
    "V10":-0.222486536,
    "V11":-1.409009452,
    "V12":0.988255366,
    "V13":-0.40486721,
    "V14":-0.212180192,
    "V15":0.551737263,
    "V16":0.372832935,
    "V17":-0.119547733,
    "V18":0.060785635,
    "V19":-1.23971373,
    "V20":0.648088501,
    "V21":-0.904013395,
    "V22":0.662310958,
    "V23":0.47871309,
    "V24":0.238845153,
    "V25":0.198053718,
    "V26":-0.270755181,
    "V27":0.595487402,
    "V28":0.392156548,
    "Amount":200.5
}])
new_prediction=model.predict(new_data)
print(f"\n prediction :{'fraud' if new_prediction[0]==1 else 'fair' }")
probability=model.predict_proba(new_data)
print(f"the probability of fraud: {probability[0][1]*100:.2f}%")
#importance -graph
plt.figure(figsize=(8,8))
sns.barplot




