import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report

df=pd.read_csv('IRIS.csv')
x=df.drop(columns=['species'])
y=df['species']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
model = RandomForestClassifier(n_estimators=100,random_state=0)
model.fit(x_train, y_train)
y_predict  = model.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
print(f"\n accuracy: {accuracy*100:.2f}%")
print("\nmodel report")
print(classification_report(y_test,y_predict,target_names=["Iris-setosa","Iris-versicolor","Iris-verginica"]))
print("\ntest")
print("\n enter the details of new flower")
sp_lenght=float(input("\nenter the sepal_length of new flower: "))
sp_width=float(input("\nenter the sepal_width of new flower: "))
pt_lenght=float(input("\nenter the petal_length of new flower: "))
pt_width=float(input("\nenter the petal_width of new flower: "))
new_flower=pd.DataFrame([{
    "sepal_length": sp_lenght,
    "sepal_width": sp_width,
    "petal_length": pt_lenght,
    "petal_width": pt_width

}])
prediction = model.predict(new_flower)
if prediction[0]==0:
    print("iris-setosa")
elif prediction[0]==1:
    print("iris-versicolor")
else:
    print("iris-verginica")

importance=model.feature_importances_
feature_names=x.columns
df_feature=pd.DataFrame({
    "feature": feature_names,
    "Importance": importance
})
plt.figure(figsize=(8, 5))
sns.barplot(x="feature", y="Importance", data=df_feature,palette="magma")
plt.title("importance of features given")
plt.xlabel("feature")
plt.ylabel("importance")
plt.savefig("iris feature importance.png")
plt.show()







