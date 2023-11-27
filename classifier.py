import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from tensorflow import keras
from tensorflow.keras import layers

import joblib

import config as cfg

from scipy.io import arff
import pandas as pd

from firewall import firewall_normal

# Load ARFF file
data = arff.loadarff(cfg.TRAIN_DATASET)

# Convert to Pandas DataFrame
df = pd.DataFrame(data[0])


# Convert 'result' column values from bytes to string and then to numeric
df['result'] = pd.to_numeric(df['result'].str.decode('utf-8'), errors='coerce')

# Define features (X) and target (y)
features = df[['comm_read_function', 'comm_write_fun', 'resp_read_fun', 'resp_write_fun', 'sub_function','command_length','resp_length', 'measurement']]
target = df['result']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Stacked Autoencoder (SAE)
sae_model = keras.Sequential([
    layers.InputLayer(input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(X_train_scaled.shape[1], activation='linear')
])

sae_model.compile(optimizer='adam', loss='mean_squared_error')
sae_model.fit(X_train_scaled, X_train_scaled, epochs=10, batch_size=32)

# Use SAE for feature extraction
sae_encoder = keras.Model(inputs=sae_model.input, outputs=sae_model.layers[2].output)
X_train_sae = sae_encoder.predict(X_train_scaled)
X_test_sae = sae_encoder.predict(X_test_scaled)

# Deep Neural Network (DNN)
dnn_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=1000, random_state=42)
dnn_model.fit(X_train_sae, y_train)

# Evaluate models
y_pred_dnn = dnn_model.predict(X_test_sae)
accuracy_dnn = accuracy_score(y_test, y_pred_dnn)

print(f'DNN Accuracy: {accuracy_dnn:.4f}')

# transfer class 0 to class 0 and other classes to class 1
dnn_predictions = dnn_model.predict(X_test_sae)
y_pred_classifier = [0 if pred == 0 else 1 for pred in dnn_predictions]

y_test_classifier = [0 if pred == 0 else 1 for pred in y_test]

accuracy_classifier = accuracy_score(y_test_classifier, y_pred_classifier)

print(f'Classifier Accuracy: {accuracy_classifier:.4f}')

# Save model
joblib.dump(sae_model, 'classifier_sae.joblib')
joblib.dump(dnn_model, 'classifier_dnn.joblib')


