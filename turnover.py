
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("HR_comma_sep.csv")

#Data cleaning

data = data.dropna()

#Undestanding factors that contribute most to employee turnover.

correlation_matrix = data.select_dtypes(include='number').corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation matrix of all numerical features.")
plt.show()

sns.histplot(data['satisfaction_level'], bins=20, kde=True)
plt.title("Distribution Plot of employee satisfaction")
plt.show()

sns.histplot(data['last_evaluation'], bins=20, kde=True)
plt.title("Distribution Plot of employee satisfaction")
plt.show()

sns.histplot(data['average_montly_hours'], bins=20, kde=True)
plt.title("Distribution Plot of employee satisfaction")
plt.show()

project_numbers = data['number_project'].unique()

left = data.groupby('number_project')['left'].sum().unique()

plt.bar(project_numbers, left, color='blue', edgecolor='black', label='Statistics')

plt.xlabel('Project numbers')
plt.ylabel('Number of people who left')
plt.title('Project count of both employees who left and stayed in the organization')

plt.legend()

plt.show()

#Perform clustering of employees who left based on their satisfaction and evaluation

from sklearn.cluster import KMeans

columns = data[['satisfaction_level', 'last_evaluation', 'left']].values

wcss = []

for i in range(1, 4):
  model = KMeans(n_clusters = i, n_init=10, init = 'k-means++', random_state=42)
  model.fit(columns)
  wcss.append(model.inertia_)

plt.plot(range(1,4), wcss)
plt.title('Clustering of employees who left based on their satisfaction and evaluation')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Handle the left Class Imbalance using the SMOTE technique

data_categorical = data.select_dtypes(exclude='number')
data_numerical = data.select_dtypes(include='number')

data_categorical_encoded = pd.get_dummies(data_categorical)

new_data = pd.concat([data_categorical_encoded, data_numerical], axis=1).drop('left', axis=1)

y = data['left'].astype(int)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(new_data, y, test_size=0.2, random_state=123, stratify=y)

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

plt.figure(figsize=(10,5))
plt.scatter(X_train_sc[:, 0], X_train_sc[:, 1], c = y_train, alpha=0.5, cmap='viridis', marker='o')
plt.title('Imbalanced Data')

plt.show()

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train_sc, y_train)

plt.figure(figsize=(10,5))
plt.scatter(X_train_smote[:, 0], X_train_smote[:, 1], c=y_train_smote, alpha=0.5, cmap='viridis', marker='o')
plt.title('Balanced Data')

plt.show()

#Perform 5-fold cross-validation model training and evaluate performance

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_curve, auc


pipeline = Pipeline([
  ('log_reg', LogisticRegression(max_iter=10000, random_state=42))
])

pipeline.fit(X_train_smote, y_train_smote)

y_pred_test_log_reg = pipeline.predict(X_test_sc)

print("Logistic Regression classification report\n")

print(classification_report(y_test, y_pred_test_log_reg), "\n")

cv_scores = cross_val_score(pipeline, X_train_smote, y_train_smote, cv=5)

print("Cross Validation Scores\n")

print(cv_scores, "\n")

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train_smote, y_train_smote)

y_pred_test_rf = rf_model.predict(X_test_sc)

print("Random Forest classification report\n")

print(classification_report(y_test, y_pred_test_rf), "\n")

cv_scores = cross_val_score(rf_model, X_train_smote, y_train_smote, cv=5)

print("Cross Validation Scores\n")

print(cv_scores, "\n")

from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier(n_estimators=100, random_state=7)

gb_model.fit(X_train_smote, y_train_smote)

y_pred_test_gb = gb_model.predict(X_test_sc)

print("Gradient Boosting classification report\n")

print(classification_report(y_test, y_pred_test_gb), "\n")

cv_scores = cross_val_score(gb_model, X_train_smote, y_train_smote, cv=5)

print("Cross Validation Scores\n")

print(cv_scores, "\n")

#Identify the best model and justify the evaluation metrics used.

