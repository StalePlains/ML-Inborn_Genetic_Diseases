import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from keras import Sequential
from imblearn.over_sampling import SMOTE

df = pd.read_excel('C:\\Users\\inspi\\Desktop\\Research\\final12unbalanced.xlsx')
validation_data = pd.read_excel('C:\\Users\\inspi\\Desktop\\Research\\Testing3noorigin.xlsx')

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import pandas as pd

# Separate features and labels
X = df.drop(columns=['CLNDN_ENC', 'CLNSIG_ENC'])
y = df['CLNDN_ENC']


# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values for both training and validation sets
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)

# Apply SMOTE to both the training and validation sets
smote = SMOTE(sampling_strategy='auto')
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_imputed, y_train)
X_val_resampled, y_val_resampled = smote.fit_resample(X_val_imputed, y_val)

print(sum(y_train_resampled == 1), sum(y_val_resampled == 1))

nan_indices = np.isnan(X_train_imputed)
nan_count = np.sum(nan_indices)
print("Number of NaN values in X_train_imputed:", nan_count)


from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

clf = HistGradientBoostingClassifier(learning_rate=0.15, max_iter=110)

clf.fit(X_train_resampled, y_train_resampled)

# Evaluate the model
accuracy = clf.score(X_val_resampled, y_val_resampled)
print("Validation Accuracy:", accuracy)

precision = precision_score(y_val_resampled, y_pred)
recall = recall_score(y_val_resampled, y_pred)
f1 = f1_score(y_val_resampled, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

search_space = {
    'learning_rate' : [0.1, 0.25, 0.5],
    'max_iter': [50, 100, 150, 200],
    'max_leaf_nodes': [10, 20, 30],
    'max_depth': [10, 50, 100],
    'min_samples_leaf': [20, 30, 40],
    'l2_regularization': [0, 0.1, 0.5],
    'max_bins': [100, 200, 300],
    'warm_start': [True, False]
}

GS = GridSearchCV(estimator = clf, param_grid=search_space, scoring=['accuracy', 'f1'], refit='accuracy', cv=5)
GS.fit(X_train_resampled, y_train_resampled)

print(GS.best_estimator_)
print(GS.best_params_)
print(GS.best_score_)

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

print(X_train_resampled)
print(y_train_resampled)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_val_resampled)

# Initialize and train the SVM model
svm_model = SVC(kernel='linear')  # You can choose different kernels (e.g., linear, RBF)
svm_model.fit(X_train_resampled, y_train_resampled)

search_space = {
    'C': [0.5, 1],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [3, 6],
    'gamma': [0.5, 'scale', 'auto'],
    'coef0': [1, 0],
    'shrinking': [False, True],
}

GS = GridSearchCV(estimator=svm_model, param_grid=search_space, scoring=['accuracy', 'f1'], refit='accuracy', cv=5)
GS.fit(X_train_resampled, y_train_resampled)
print(GS.best_estimator_)
print(GS.best_params_)
print(GS.best_score_)

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

scores = []
hgb_model_1 = HistGradientBoostingClassifier(learning_rate=0.5, max_iter=100, max_bins=200, max_depth=10, max_leaf_nodes=30, min_samples_leaf=40, warm_start=True)
hgb_model_2 = HistGradientBoostingClassifier(learning_rate=0.5, max_iter=100, max_bins=200, max_depth=10, max_leaf_nodes=30, min_samples_leaf=40, warm_start=True)
hgb_model_3 = HistGradientBoostingClassifier(learning_rate=0.5, max_iter=100, max_bins=200, max_depth=10, max_leaf_nodes=30, min_samples_leaf=40, warm_start=True)

hgb_model_1.fit(X_train_resampled, y_train_resampled)

y_pred = hgb_model_1.predict(X_val_resampled)
accuracy = accuracy_score(y_val_resampled, y_pred)
accuracy = accuracy_score(y_val_resampled, y_pred)
precision = precision_score(y_val_resampled, y_pred)
recall = recall_score(y_val_resampled, y_pred)
f1 = f1_score(y_val_resampled, y_pred)
conf_matrix = confusion_matrix(y_val_resampled, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# # Combine base models into a voting classifier
voting_classifier = VotingClassifier(estimators=[('hgb1', hgb_model_1), ('hgb2', hgb_model_2), ('hgb3', hgb_model_3)], voting='hard')

# Train the voting classifier
voting_classifier.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred = voting_classifier.predict(X_val_resampled)
print('Voting classifier')
accuracy = accuracy_score(y_val_resampled, y_pred)
accuracy = accuracy_score(y_val_resampled, y_pred)
precision = precision_score(y_val_resampled, y_pred)
recall = recall_score(y_val_resampled, y_pred)
f1 = f1_score(y_val_resampled, y_pred)
conf_matrix = confusion_matrix(y_val_resampled, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

hgb_model_1.fit(X_train_resampled, y_train_resampled)
hgb_model_2.fit(X_train_resampled, y_train_resampled)
hgb_model_3.fit(X_train_resampled, y_train_resampled)

# Generate predictions on the validation set
predictions_1 = hgb_model_1.predict(X_val_resampled)
predictions_2 = hgb_model_2.predict(X_val_resampled)
predictions_3 = voting_classifier.predict(X_val_resampled)

# Create a meta-model
meta_model = LogisticRegression(solver='newton-cholesky')

# Create a stacked classifier
stacked_clf = StackingClassifier(estimators=[
    ('hgb1', hgb_model_1), 
    ('hgb2', hgb_model_2), 
    ('voting', voting_classifier)], 
    final_estimator=meta_model)

# Train the stacked classifier on the predictions

stacked_clf.fit(np.column_stack((predictions_1, predictions_2, predictions_3)), y_val_resampled)

search_space = {
    'estimators': [[
    ('hgb1', hgb_model_1), 
    ('hgb2', hgb_model_2), 
    ('voting', voting_classifier)],
    [
    ('hgb1', hgb_model_1), 
    ('hgb2', hgb_model_2), 
    ('hgb3', hgb_model_3)]],
    'final_estimator': [meta_model],
    'cv': [5],
    'stack_method': ['auto', 'predict_proba', 'decision_function', 'predict'],
    'passthrough': [True, False]
}
GS = GridSearchCV(estimator=stacked_clf, param_grid=search_space, scoring=['accuracy', 'f1'], refit='accuracy', cv=5)
GS.fit(X_train_resampled, y_train_resampled)
print(GS.best_estimator_)
print(GS.best_params_)
print(GS.best_score_)

# Make predictions on the test set
test_predictions_1 = hgb_model_1.predict(X_val_resampled)
test_predictions_2 = hgb_model_2.predict(X_val_resampled)
test_predictions_3 = voting_classifier.predict(X_val_resampled)
test_predictions_voting = voting_classifier.predict(X_val_resampled)


y_pred = stacked_clf.predict(np.column_stack((test_predictions_1, test_predictions_2, test_predictions_3)))


# Calculate accuracy
print('Stacked Classifier')
accuracy_stacked = accuracy_score(y_val_resampled, y_pred)
precision = precision_score(y_val_resampled, y_pred)
recall = recall_score(y_val_resampled, y_pred)
f1 = f1_score(y_val_resampled, y_pred)
conf_matrix = confusion_matrix(y_val_resampled, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(stacked_clf, X_train_resampled, y_train_resampled, cv=2)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)




# # Get the names of the features
feature_names = X.columns

# Create lists to store feature importances for each model
relative_importances_hgb1 = []
relative_importances_hgb2 = []
relative_importances_hgb3 = []

# Loop over each model
for model in [hgb_model_1, hgb_model_2, hgb_model_3]:
    # Calculate permutation importances
    perm_importance = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=10)
    
    # Normalize permutation importances
    relative_importance = perm_importance.importances_mean / np.sum(perm_importance.importances_mean)

    abs_importance = np.abs(relative_importance)
    
    # Append relative importances to respective lists
    if model == hgb_model_1:
        relative_importances_hgb1.extend(relative_importance)
    elif model == hgb_model_2:
        relative_importances_hgb2.extend(relative_importance)
    elif model == hgb_model_3:
        relative_importances_hgb3.extend(relative_importance)

# Plot relative importances for each model
plt.figure(figsize=(10, 6))

# HGB Model 1
plt.bar(feature_names, relative_importances_hgb1, color='b', alpha=0.7, label='HGB Model 1')

# HGB Model 2
plt.bar(feature_names, relative_importances_hgb2, color='g', alpha=0.7, label='HGB Model 2')

# HGB Model 3
plt.bar(feature_names, relative_importances_hgb3, color='r', alpha=0.7, label='HGB Model 3')

plt.show()

model = RandomForestClassifier(criterion='log_loss', max_depth=None, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0, n_estimators=50)
model.fit(X_train_resampled, y_train_resampled)

search_space = {
    'n_estimators': [50, 100, 150],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [50, 200, None],
    'min_samples_split': [1, 2, 5],
    'min_samples_leaf': [1, 5, 10],
    'min_weight_fraction_leaf': [0, 1.5, 3],
    'max_features': ['sqrt', 'log2', None],
    'max_leaf_nodes': [None, 10, 30],
    'min_impurity_decrease': [0, 0.5, 2]
}

GS = GridSearchCV(estimator=model, param_grid=search_space, scoring=['accuracy', 'f1'], refit='accuracy', cv=5)
GS.fit(X_train_resampled, y_train_resampled)
print(GS.best_estimator_)
print(GS.best_params_)
print(GS.best_score_)

from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_val_resampled)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Example evaluation metrics
accuracy = accuracy_score(y_val_resampled, y_pred)
precision = precision_score(y_val_resampled, y_pred)
recall = recall_score(y_val_resampled, y_pred)
f1 = f1_score(y_val_resampled, y_pred)
conf_matrix = confusion_matrix(y_val_resampled, y_pred)

print("Accuracy:", accuracy)
print(conf_matrix)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

input_dim = X.shape[1]

model = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train_resampled, y_train_resampled, epochs=1000, validation_data=(X_val_resampled, y_val_resampled))

val_loss, val_accuracy = model.evaluate(X_val_resampled, y_val_resampled)
print("Validation Accuracy:", val_accuracy)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Plot loss
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot accuracy
plt.figure(figsize=(10, 6))
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

knn = KNeighborsClassifier(algorithm='auto', leaf_size=15, metric='nan_euclidean', n_neighbors=5, p=1)

# Train the model
knn.fit(X_train_resampled, y_train_resampled)

search_space = {
    'n_neighbors': [5, 10, 20],
    'algorithm': ['auto', 'brute', 'kd_tree', 'ball_tree'],
    'leaf_size': [15, 30, 60],
    'p': [1, 2, 5],
    'metric': ['cityblock', 'cosine', 'l2', 'euclidean', 'haversine', 'l1', 'manhattan', 'nan_euclidean', 'minkowski'],
}

GS = GridSearchCV(estimator=knn, param_grid=search_space, scoring=['accuracy', 'f1'], refit='accuracy', cv=5)
GS.fit(X_train_resampled, y_train_resampled)
print(GS.best_estimator_)
print(GS.best_params_)
print(GS.best_score_)
exit()

Make predictions on the testing data
y_pred = knn.predict(X_val_resampled)

# Evaluate the model
accuracy = accuracy_score(y_val_resampled, y_pred)
precision = precision_score(y_val_resampled, y_pred)
recall = recall_score(y_val_resampled, y_pred)
f1 = f1_score(y_val_resampled, y_pred)

print(accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

