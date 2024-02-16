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

# Assuming df is your dataframe with features (X) and labels (y)
# Replace this with your actual dataframe

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

print(sum(y_train_resampled ==1), sum(y_val_resampled == 1))

# nan_indices = np.isnan(X_train_imputed)
# nan_count = np.sum(nan_indices)
# print("Number of NaN values in X_train_imputed:", nan_count)

# clf = HistGradientBoostingClassifier()
# clf.fit(X_train_resampled, y_train_resampled)

# # Evaluate the model
# accuracy = clf.score(X_val_resampled, y_val_resampled)
# print("Validation Accuracy:", accuracy)

# class_weights = {0: 1, 1: 2.25}

# model = RandomForestClassifier(max_depth=200, class_weight=class_weights)
# model.fit(X_train_resampled, y_train_resampled)

# from sklearn.metrics import accuracy_score, classification_report
# # Assuming 'model' is your trained machine learning model
# y_pred = model.predict(X_val_resampled)
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# # Example evaluation metrics
# accuracy = accuracy_score(y_val_resampled, y_pred)
# precision = precision_score(y_val_resampled, y_pred)
# recall = recall_score(y_val_resampled, y_pred)
# f1 = f1_score(y_val_resampled, y_pred)
# conf_matrix = confusion_matrix(y_val_resampled, y_pred)

# print("Accuracy:", accuracy)
# print(conf_matrix)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1)


input_dim = X.shape[1]

# custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0015)

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

# from sklearn.neighbors import KNeighborsClassifier

# knn = KNeighborsClassifier(n_neighbors=80)

# # Train the model
# knn.fit(X_train_imputed, y_train)

# # Make predictions on the testing data
# y_pred = knn.predict(X_val_imputed)

# # Evaluate the model
# accuracy = accuracy_score(y_val, y_pred)
# print("Accuracy:", accuracy)

# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler

# # Assuming X is your feature matrix
# # Standardize the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_train_imputed)

# # Instantiate KMeans with k=3 (number of clusters)
# kmeans = KMeans(n_clusters=3, random_state=42)

# # Fit the model to the scaled data
# kmeans.fit(X_scaled)

# # Predict the cluster labels
# cluster_labels = kmeans.predict(X_scaled)

# # Analyze the clusters (e.g., centroids, cluster sizes)
# centroids = kmeans.cluster_centers_
# cluster_sizes = {i: sum(cluster_labels == i) for i in range(kmeans.n_clusters)}

# from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# # Silhouette Score
# silhouette = silhouette_score(X_scaled, cluster_labels)
# print("Silhouette Score:", silhouette)

# # Davies-Bouldin Index
# db_index = davies_bouldin_score(X_scaled, cluster_labels)
# print("Davies-Bouldin Index:", db_index)

# # Calinski-Harabasz Index
# ch_index = calinski_harabasz_score(X_scaled, cluster_labels)
# print("Calinski-Harabasz Index:", ch_index)
