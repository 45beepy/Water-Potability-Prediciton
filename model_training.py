import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from lime.lime_tabular import LimeTabularExplainer

# Step 1: Load Data
file_path = 'water_potability.csv'  # Update with your file path
data = pd.read_csv(file_path)

# Step 2: Handle Missing Values with KNN Imputer
knn_imputer = KNNImputer(n_neighbors=5)
data_imputed = pd.DataFrame(knn_imputer.fit_transform(data), columns=data.columns)

# Step 3: Split Data into Features and Target
X = data_imputed.drop('Potability', axis=1)
y = data_imputed['Potability']

# Step 4: Handle Imbalanced Data using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled)

# Step 6: Data Transformation - Apply Power Transformation for Normality
pt = PowerTransformer()

# Fit PowerTransformer on the training set
X_train_transformed = pt.fit_transform(X_train)

# Apply the transformation to the test set
X_test_transformed = pt.transform(X_test)

# Step 7: Model Training with Hyperparameter Tuning

# RandomForest Hyperparameter Tuning
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
rf_model = RandomForestClassifier(random_state=42)
rf_search = RandomizedSearchCV(rf_model, param_distributions=rf_param_grid, n_iter=20, cv=3, scoring='accuracy', verbose=1, random_state=42)
rf_search.fit(X_train_transformed, y_train)
rf_model = rf_search.best_estimator_

# Step 8: Evaluate the RandomForest model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

rf_accuracy, rf_report = evaluate_model(rf_model, X_test_transformed, y_test)

# Step 9: Save the Model for Deployment
import joblib
joblib.dump(rf_model, 'best_model.pkl')
joblib.dump(pt, 'power_transformer.pkl')

# Step 10: LIME Explainer Setup and Explanation Generation
explainer = LimeTabularExplainer(X_train_transformed, feature_names=X.columns, class_names=['Not Potable', 'Potable'], discretize_continuous=True)

# LIME Explanation Function
def get_lime_explanation(input_data):
    input_transformed = pt.transform([input_data])  # Transform input using PowerTransformer
    exp = explainer.explain_instance(input_transformed[0], rf_model.predict_proba, num_features=5)
    return exp.as_html()

# After transformation in training script
joblib.dump(X_train_transformed, 'X_train_transformed.pkl')
