import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Paths
base_dir = 'C:/Users/danie/Desktop/IR/Information_retrieval/exe2/IR-files'
output_dir = 'C:/Users/danie/Desktop/IR/Information_retrieval/exe2/clustering_results'
os.makedirs(output_dir, exist_ok=True)

# Load Data
def load_combined_data(group_name):
    """ Load combined matrix for a group """
    file_path = os.path.join(output_dir, f"{group_name}_clustering_results.csv")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    return pd.read_csv(file_path)

# Train ANN Model
def train_ann(X, y, activation='relu', embedding_output=128):
    num_classes = len(np.unique(y))  # Number of output classes
    input_dim = X.shape[1]          # Number of input features

    # Ensure labels are integers for sparse_categorical_crossentropy
    if y.ndim != 1 or y.dtype != np.int32:
        print("Converting labels to integers for sparse_categorical_crossentropy.")
        y = y.astype(np.int32)

    # ANN model
    model = Sequential([
        Dense(embedding_output, input_dim=input_dim, activation=activation),  # Input layer
        Dense(10, activation=activation),  # First hidden layer
        Dense(10, activation=activation),  # Second hidden layer
        Dense(7, activation=activation),   # Third hidden layer
        Dense(num_classes, activation='softmax')  # Output layer
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Train and Evaluate Classifiers
def train_and_evaluate_classifiers(X, y, group_name):
    classifiers = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'SVM': SVC(kernel='linear', probability=True),
        'Random Forest': RandomForestClassifier(n_estimators=100)
    }
    feature_importance = {}

    for clf_name, clf in classifiers.items():
        print(f"\nTraining {clf_name} for group: {group_name}")
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')
        print(f"{clf_name} Accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

        clf.fit(X, y)
        if hasattr(clf, 'coef_'):
            importance = np.abs(clf.coef_[0])
        elif hasattr(clf, 'feature_importances_'):
            importance = clf.feature_importances_
        else:
            importance = np.zeros(X.shape[1])

        feature_importance[clf_name] = importance

    # Save feature importance
    feature_df = pd.DataFrame(feature_importance, index=[f"Feature {i+1}" for i in range(X.shape[1])])
    feature_df['Mean_Importance'] = feature_df.mean(axis=1)
    feature_df = feature_df.sort_values(by='Mean_Importance', ascending=False).head(20)
    feature_df.to_excel(os.path.join(output_dir, f"{group_name}_feature_importance.xlsx"))
    print(f"Feature importance for {group_name} saved to Excel.")

# Main Execution
groups = ['bert-sbert', 'doc2vec', 'glove', 'word2vec']
for group in groups:
    print(f"\nProcessing group: {group}")
    data = load_combined_data(group)
    if data is None:
        continue

    X = data.iloc[:, :-1].to_numpy()  # Features
    y = data.iloc[:, -1].to_numpy()   # Labels

    # Validate labels are integers
    if y.dtype != np.int32:
        print("Converting labels to integers.")
        y = y.astype(np.int32)

    # Split Data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # ANN Training (Topology 1)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    model_path = os.path.join(output_dir, f"best_ann_relu_{group}.h5")
    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True)

    ann_model = train_ann(X_train, y_train, activation='relu')
    history = ann_model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=15, batch_size=32, callbacks=[early_stop, checkpoint])
    ann_loss, ann_acc = ann_model.evaluate(X_test, y_test)
    print(f"ANN (ReLU) Test Accuracy for {group}: {ann_acc:.4f}")

    # ANN Training (Topology 2)
    model_path = os.path.join(output_dir, f"best_ann_gelu_{group}.h5")
    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True)

    ann_model = train_ann(X_train, y_train, activation='gelu')
    history = ann_model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=15, batch_size=32, callbacks=[early_stop, checkpoint])
    ann_loss, ann_acc = ann_model.evaluate(X_test, y_test)
    print(f"ANN (GELU) Test Accuracy for {group}: {ann_acc:.4f}")

    # Train Other Classifiers
    train_and_evaluate_classifiers(X, y, group)

print("All tasks completed.")
