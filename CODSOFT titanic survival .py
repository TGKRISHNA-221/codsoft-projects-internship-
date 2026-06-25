
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#we chose the random forest as it is the problem for classification and random forest has better accuracy than most other algorithms
from sklearn.metrics import accuracy_score, classification_report

#csv file given to the pandas

df = pd.read_csv("Titanic-Dataset.csv")

print("\n the survived count")
print(df["Survived"].value_counts())

df.drop(columns=["Name", "Ticket", "PassengerId","Cabin","Embarked"], inplace=True)
#there is no use for the columns name ,ticket , passenger id for the survival of the passengers it only gives the identity of the passenger
#the cabin has a lot of empty values and not necessary for survival prediction
#the embarked is the place of boarding it doesnt matter as the ship sank after boarding at every station

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
#as the sex is defined as male and female but for system we make them as binary so
#male=0,female=1

df["familySize"] = df["SibSp"] + df["Parch"] + 1
#having only one column for one factor(family on board)

df["isAlone"] = (df["familySize"] == 1).astype(int)
#if alone then so consider this factor

#for the model training dividing the data into two parts x=factors,y=solutions

X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split( X, y,test_size=0.2,random_state=42  )


#initializing the model of random forest for training and testing:
model = RandomForestClassifier( n_estimators=100, random_state=0)

model.fit(X_train, y_train)

#model trained

#Testing phase:
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n model accuracy: {accuracy*100:.2f}%")

# Details
print("\n report on prediction")
print(classification_report(y_test, y_pred,target_names=["Did not survive", "Survived"]))
# details will be on precision ,recall, f1-score and accuracy details

# prediction on new passenger (testing):

print("testing by new data")
print("enter passenger details below:\n")
pclass = int(input("Passenger Class (1=Rich, 2=Middle, 3=Poor): "))
sex = input("Sex (male/female): ").strip().lower()
age = float(input("Age: "))
sibsp = int(input("Number of Siblings/Spouse on board: "))
parch = int(input("Number of Parents/Children on board: "))
fare = float(input("Fare paid: "))

sex_num = 1 if sex == "female" else 0
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0
#new data set
new_passenger = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex_num,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "familySize": family_size,
    "isAlone": is_alone
}])

#prediction :

prediction = model.predict(new_passenger)
probability = model.predict_proba(new_passenger)
print(f"\n prediction : {'survived' if prediction[0] == 1 else 'did not survive'}")
print(f"survival probability : {probability[0][1]*100:.1f}%")

#features importance in prediction :

importances = model.feature_importances_
feature_names = X.columns

df_feature = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=df_feature, x="importance", y="feature",palette="plasma")
plt.title("Which Features Matter Most?", fontsize=13, fontweight="bold")
plt.xlabel("Importance Score")
plt.ylabel("Feature name")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
