import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib


data = pd.read_csv("fitness_data.csv")

# Preprocessing
label_encoder = LabelEncoder()
data['Fitness Level'] = label_encoder.fit_transform(data['Fitness Level'])

X = data.iloc[:, :-1]  # Features (excluding 'Fitness Level')
y = data['Fitness Level']  # Target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model and encoder
joblib.dump(model, 'fitness_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
