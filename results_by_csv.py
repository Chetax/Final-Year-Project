import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
import csv


def restructure_classifier_results(input_csv_path, output_csv_path):
    """
    Restructures classifier results CSV to have classifiers as columns and filenames as rows.
    """
    try:
        with open(input_csv_path, 'r', newline='') as infile, \
             open(output_csv_path, 'w', newline='') as outfile:
            reader = csv.DictReader(infile)
            if reader.fieldnames:
                classifiers = []
                data = {}
                for row in reader:
                    filename = row.get('File')
                    classifier = row.get('Classifier')
                    accuracy = row.get('Accuracy')
                    if filename and classifier and accuracy:
                        if filename not in data:
                            data[filename] = {}
                        data[filename][classifier] = accuracy
                        if classifier not in classifiers:
                            classifiers.append(classifier)

                writer = csv.writer(outfile)
                header = ['Filename'] + classifiers
                writer.writerow(header)

                for filename in data:
                    row_data = [filename]
                    for classifier in classifiers:
                        row_data.append(data[filename].get(classifier, 'N/A'))
                    writer.writerow(row_data)

                print(f"Successfully restructured data from '{input_csv_path}' to '{output_csv_path}'")
            else:
                print("Error: Input CSV file has no header row.")

    except FileNotFoundError:
        print(f"Error: Input file '{input_csv_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def load_data_from_directory(directory):
    file_data_labels = []
    print(f"Listing files in directory: {directory}")
    try:
        files_in_dir = os.listdir(directory)
        print(f"Files found: {files_in_dir}")
        for file in files_in_dir:
            print(f"Checking file: {file}")
            if file.endswith(".csv"):
                file_path = os.path.join(directory, file)
                df = pd.read_csv(file_path)
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
                file_data_labels.append((X, y, file))
    except FileNotFoundError:
        print(f"Error: Directory not found: {directory}")
    return file_data_labels

def evaluate_classifiers(X, y, classifiers):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
    return results

if __name__ == "__main__":
    dir_name = "output"
    directory = dir_name
    absolute_directory = os.path.join(os.getcwd(), directory)
    file_data_labels = load_data_from_directory(absolute_directory)
    print(f"Number of files loaded: {len(file_data_labels)}")

    classifiers = {

        'NaiveBayes': GaussianNB(),
        'Multilayer Perceptron': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42),
        'Logistic': LogisticRegression(max_iter=1000, random_state=42),
        'SGD': SGDClassifier(max_iter=1000, random_state=42),
        'SMO': SVC(kernel='linear', random_state=42),  # Using SVC with linear kernel for SMO
        'Voted Perceptron': Perceptron(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'J48': DecisionTreeClassifier(random_state=42),  # Using DecisionTreeClassifier for J48
        'K Nearest': KNeighborsClassifier(),
    }

    all_results = []

    for X, y, filename in file_data_labels:
        print(f"Evaluating classifiers for {filename}...")
        file_results = evaluate_classifiers(X, y, classifiers)
        for clf_name, accuracy in file_results.items():
            all_results.append({'File': filename, 'Classifier': clf_name, 'Accuracy': accuracy})

    results_df = pd.DataFrame(all_results)
    results_csv_file = "classifier_results.csv"
    results_df.to_csv(results_csv_file, index=False)
    print(f"Results saved to {results_csv_file}")


    input_csv_path = results_csv_file
    output_csv_path = 'results.csv'
    restructure_classifier_results(input_csv_path, output_csv_path)
