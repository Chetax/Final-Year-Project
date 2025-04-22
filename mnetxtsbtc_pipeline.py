

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# For data preprocessing and splitting
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)

# For building the MLP model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------------------
# 1. Load Data from CSV
# -----------------------------------------
# Specify the file path for the fused features CSV file.
file_path = '/content/drive/MyDrive/fyproject/mobilenet_features.csv'

# Note: The CSV file contains float16 features to reduce size.
# When reading, we let pandas infer the dtypes; later we will convert feature columns if needed.
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame to verify successful loading
print("Data sample:")
print(df.head())

# -----------------------------------------
# 2. Data Preprocessing
# -----------------------------------------
# Separate features and labels. We assume the last column is 'label'.
if 'label' not in df.columns:
    raise ValueError("The CSV file must contain a 'label' column.")

X = df.drop(columns=['label'])
y = df['label']

# Convert features from float16 to float32 for numerical stability during training
X = X.astype('float32')

# Check for missing values and fill them if necessary.
missing_values = X.isnull().sum().sum()
if missing_values > 0:
    print(f"Found {missing_values} missing values in features. Filling missing values with column mean.")
    X.fillna(X.mean(), inplace=True)
else:
    print("No missing values found in the features.")

# Feature Scaling using StandardScaler
# StandardScaler standardizes features by removing the mean and scaling to unit variance.
# This is generally preferred for MLPs which are sensitive to the scale of input data.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------
# 3. Data Splitting: Train, Validation, Test
# -----------------------------------------
# We use 70% for training, and the remaining 30% is split equally into validation and test sets.
# Stratification is applied to preserve class proportions.
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.30, 
                                                    stratify=y, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, 
                                                stratify=y_temp, random_state=42)

print(f"Train set size: {X_train.shape[0]} samples")
print(f"Validation set size: {X_val.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# -----------------------------------------
# 4. MLP Model Definition
# -----------------------------------------
# We define a Multilayer Perceptron (MLP) model using Keras.
# The model includes:
#  - Two hidden layers with ReLU activation.
#  - Batch Normalization to stabilize and accelerate training.
#  - Dropout for regularization to help prevent overfitting.
#  - L2 regularization added to the Dense layers.
#
# The number of input neurons equals the number of features.
input_dim = X_train.shape[1]

# Define the model architecture
model = Sequential()

# First hidden layer with 512 neurons, L2 regularization, and ReLU activation.
model.add(Dense(512, activation='relu', input_dim=input_dim, kernel_regularizer=l2(1e-4)))
model.add(BatchNormalization())
model.add(Dropout(0.5))  # Dropout rate of 50%

# Second hidden layer with 256 neurons
model.add(Dense(256, activation='relu', kernel_regularizer=l2(1e-4)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Output layer with 1 neuron and sigmoid activation for binary classification.
model.add(Dense(1, activation='sigmoid'))

# Compile the model with Binary Cross-Entropy loss and Adam optimizer.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# -----------------------------------------
# 5. Model Training
# -----------------------------------------
# We set up early stopping to monitor the validation loss.
# This stops training when the model performance on the validation set stops improving,
# reducing the risk of overfitting.
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Set training parameters: number of epochs and batch size.
epochs = 100
batch_size = 32

history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stop],
                    verbose=1)

# Plot training history for visual analysis of loss and accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss During Training')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy During Training')
plt.legend()
plt.show()

# -----------------------------------------
# 6. Model Evaluation
# -----------------------------------------
# Predict probabilities on the test set.
y_pred_prob = model.predict(X_test).ravel()

# Convert probabilities to binary predictions with threshold 0.5.
y_pred = (y_pred_prob >= 0.5).astype(int)

# Calculate evaluation metrics.
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_pred_prob)

# Specificity: TN / (TN + FP)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)

print("\nTest Set Evaluation Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC Score: {auc:.4f}")

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# -----------------------------------------
# 7. Hyperparameter Tuning (Optional)
# -----------------------------------------
# A strategy for hyperparameter tuning could involve using Keras Tuner or sklearn's RandomizedSearchCV
# in combination with a KerasClassifier wrapper.
#
# For example, one can use Keras Tuner to search over:
#  - Number of hidden layers and neurons per layer
#  - Activation functions (e.g., 'relu', 'tanh')
#  - Learning rate of the optimizer
#  - Regularization strengths and dropout rates
#
# Here is a simple example using Keras Tuner:
#
# from kerastuner import RandomSearch
#
# def build_model(hp):
#     model = Sequential()
#     model.add(Dense(units=hp.Int('units_input', min_value=256, max_value=1024, step=128),
#                     activation='relu', input_dim=input_dim, kernel_regularizer=l2(hp.Float('l2_reg', 1e-5, 1e-2, sampling='LOG'))))
#     model.add(BatchNormalization())
#     model.add(Dropout(hp.Float('dropout_rate', 0.3, 0.7, step=0.1)))
#     # Additional hidden layer
#     model.add(Dense(units=hp.Int('units_hidden', min_value=128, max_value=512, step=64),
#                     activation='relu', kernel_regularizer=l2(hp.Float('l2_reg', 1e-5, 1e-2, sampling='LOG'))))
#     model.add(BatchNormalization())
#     model.add(Dropout(hp.Float('dropout_rate', 0.3, 0.7, step=0.1)))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     return model
#
# tuner = RandomSearch(build_model,
#                      objective='val_accuracy',
#                      max_trials=10,
#                      executions_per_trial=2,
#                      directory='my_dir',
#                      project_name='mlp_breast_cancer_tuning')
#
# tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val))
#
# best_model = tuner.get_best_models(num_models=1)[0]
#
# -----------------------------------------
# 8. Model Saving
# -----------------------------------------
# Save the trained model to Google Drive.
model_save_path = '/content/drive/MyDrive/fyproject/mlp_breast_cancer_model.h5'
model.save(model_save_path)
print(f"\nModel saved to: {model_save_path}")

# -----------------------------------------
# 9. Addressing Overfitting and Unrealistic Accuracy
# -----------------------------------------
# Several strategies have been implemented to prevent overfitting:
#  - Data splitting into training, validation, and test sets ensures the model is evaluated on unseen data.
#  - Batch normalization and dropout layers help in regularizing the model.
#  - L2 regularization is applied in Dense layers.
#  - Early stopping monitors the validation loss to avoid over-training.
#
# It is important to note that achieving exactly 100% accuracy in real-world medical datasets is usually a red flag.
# Perfect accuracy can indicate issues such as data leakage or that the dataset is too simplistic.
# The goal should be to achieve high and realistic performance while ensuring that the model generalizes well
# to new, unseen data. Ethical implications must be considered when deploying AI in medical diagnostics, and
# models should always be validated thoroughly before any clinical application.
