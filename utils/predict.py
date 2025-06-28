import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib
import glob

# Load test data
df = pd.read_csv('model_1/datasets/Test_samples.csv')
df = df.drop(columns=['ip.session_id', 'meta.direction'], errors='ignore')
X = df.drop(columns=['malware'])
y_true = df['malware']

# Load the scaler (find the correct .pkl file)
scaler_path = glob.glob('model_1/results/scaler_*.pkl')[0]  # get the first matching scaler
scaler = joblib.load(scaler_path)
X_scaled = scaler.transform(X)

# Load model
model = load_model('model_1/model_1.h5')

# Predict
raw_pred = model.predict(X_scaled)
y_pred = (raw_pred > 0.5).astype(int).flatten()

# Evaluate
cm = confusion_matrix(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
print('Confusion matrix:')
print(cm)
print('Accuracy:', acc)

# Save confusion matrix as CSV (with labels)
cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
cm_df.to_csv('confusion_matrix.csv')
