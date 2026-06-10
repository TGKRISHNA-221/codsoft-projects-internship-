import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

#giving the file
df=pd.read_csv('advertising.csv')

#data cleaning:
df.dropna(subset=['Sales'], inplace=True)

#splitting the data for training :
x=df.drop(columns=['Sales'])
y=df['Sales']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#scaling:
scaling= StandardScaler()
scaled_x_train=scaling.fit_transform(x_train)
scaled_x_test=scaling.transform(x_test)

#initializing the model:
model = RandomForestRegressor(n_estimators=100,random_state=0)
model.fit(scaled_x_train, y_train)
prediction=model.predict(scaled_x_test)

#metrics:
mae=mean_absolute_error(y_test,prediction)
rmse=np.sqrt(mean_squared_error(y_test,prediction))
r2=r2_score(y_test,prediction)
print(f"\nmean absolute error: {mae:.3f}")
print(f"\nroot of mean squared error: {rmse:.3f}")
print(f"\nR2 score :{r2:.3f}")

#new test data:
tv_adv=float(input("\nEnter the tv advertisment value: "))
radio_adv=float(input("\nEnter the radio advertisment value: "))
newspaper_adv=float(input("\nEnter the newspaper advertisment value: "))

new_test_data=pd.DataFrame([{
    'TV': tv_adv,
    'Radio': radio_adv,
    'Newspaper': newspaper_adv
}])

scaled_new_data=scaling.transform(new_test_data)

test_prediction=model.predict(scaled_new_data)
print(f"the new unseen data prediction is:{test_prediction[0]:.3f}")

#factor-importance graph:
imp_factors=pd.DataFrame({
    "features": x.columns,
    "importance":model.feature_importances_
})
plt.figure(figsize=(5, 5))
sns.barplot(x="features", y="importance", data=imp_factors,palette="plasma")
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("features-Importance.png")
plt.show()



