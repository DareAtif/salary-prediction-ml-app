# PREPROCESSING START

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('D:\salary-prediction-ML\data\salary_dataset.csv')
data.head()
# Encode categorical columns
le_edu = LabelEncoder()
le_role = LabelEncoder()
le_loc = LabelEncoder()
le_size = LabelEncoder()

data['Education'] = le_edu.fit_transform(data['Education'])
data['JobRole'] = le_role.fit_transform(data['JobRole'])
data['Location'] = le_loc.fit_transform(data['Location'])
data['CompanySize'] = le_size.fit_transform(data['CompanySize'])
X = data.drop("Salary(LPA)", axis=1)
y = data["Salary(LPA)"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape
X_test.shape


#-----models------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
}
best_score = 0
best_model = None
for name,model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name} R^2 Score: {score}")
    if score > best_score:
        best_score = score
        best_model = model
print("\nBest Model Selected:", best_model)
# --- PREPROCESSING END ---

# Save best model
import pickle
import os

os.makedirs("models",exist_ok=True)

pickle.dump(best_model,open("models/model.pkl","wb"))
pickle.dump(le_edu,open("models/le_edu.pkl","wb"))
pickle.dump(le_role,open("models/le_role.pkl","wb"))
pickle.dump(le_loc,open("models/le_loc.pkl","wb"))
pickle.dump(le_size,open("models/le_size.pkl","wb"))

print("All models saved Successfully")


