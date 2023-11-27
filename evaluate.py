import config as cfg
from classification import clf
from firewall import firewall_normal
from empirical import emp
from scipy.io import arff
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def eval(data_path):

    data = arff.loadarff(data_path)
    # Convert to Pandas DataFrame
    all_commands = pd.DataFrame(data[0])
    # Convert 'result' column values from bytes to string and then to numeric
    all_commands['result'] = pd.to_numeric(all_commands['result'].str.decode('utf-8'), errors='coerce')


    normal_commands = firewall_normal(data_path)
    normal_commands = clf(normal_commands)[0]
    normal_commands = emp(normal_commands)
    print(normal_commands)

    # Find rows that are in df1 but not in df2
    df1 = all_commands
    df2 = normal_commands
    attack_commands = pd.merge(df1, df2, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)

    normal_commands['result'] = [0 if label == 0 else 1 for label in normal_commands['result']]
    attack_commands['result'] = [0 if label == 0 else 1 for label in attack_commands['result']]

    actual = pd.concat([normal_commands['result'], attack_commands['result']], ignore_index=True) # Actual labels (ground truth)
    predicted = pd.concat([normal_commands['result'].map(lambda x: 0), attack_commands['result'].map(lambda x: 1)], ignore_index=True)  # Predicted labels

    # Calculate TP, FP, FN, TN
    tp = sum((a == 1) and (p == 1) for a, p in zip(actual, predicted))
    fp = sum((a == 0) and (p == 1) for a, p in zip(actual, predicted))
    fn = sum((a == 1) and (p == 0) for a, p in zip(actual, predicted))
    tn = sum((a == 0) and (p == 0) for a, p in zip(actual, predicted))

    # Display the results
    print("True Positives (TP):", tp)
    print("False Positives (FP):", fp)
    print("False Negatives (FN):", fn)
    print("True Negatives (TN):", tn)

    # Assuming 'actual' and 'predicted' are your actual and predicted lists, respectively
    # These could be obtained from the concat operation you performed earlier

    # Calculate Accuracy
    accuracy = accuracy_score(actual, predicted)

    # Calculate Precision
    precision = precision_score(actual, predicted)

    # Calculate Recall
    recall = recall_score(actual, predicted)

    # Calculate F1 Score
    f1 = f1_score(actual, predicted)

    # Display the results
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


eval(cfg.TEST_DATASET)