import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow import keras
import joblib
import config as cfg
from scipy.io import arff
import pandas as pd

from firewall import firewall_normal
def clf(normal_commands):

    df = normal_commands
    # Define features (X) for the new dataset
    features = df[['comm_read_function', 'comm_write_fun', 'resp_read_fun', 'resp_write_fun', 'sub_function','command_length','resp_length', 'measurement']]

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Load SAE model
    sae_model = joblib.load('classifier_sae.joblib')

    # Use SAE for feature extraction on new dataset
    sae_encoder = keras.Model(inputs=sae_model.input, outputs=sae_model.layers[2].output)
    new_sae_features = sae_encoder.predict(features_scaled)

    # Load DNN model
    dnn_model = joblib.load('classifier_dnn.joblib')

    # Use DNN for prediction on new dataset
    dnn_predictions = dnn_model.predict(new_sae_features)


    # Print or use the predictions as needed
    print("DNN Predictions:")
    print(dnn_predictions)

    # Define a threshold for classifying normal and attack (adjust as needed)
    threshold = 0.5

    # Classify instances based on the threshold
    normal_instances = dnn_predictions < threshold
    attack_instances = ~normal_instances

    # Print or use the results as needed
    print("Normal Instances:")
    print(df[normal_instances])

    print("\nAttack Instances:")
    print(df[attack_instances])

    return df[normal_instances], df[attack_instances]


normal_commands = firewall_normal(cfg.TEST_DATASET)
clf(normal_commands)