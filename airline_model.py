import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss, roc_auc_score, classification_report, ConfusionMatrixDisplay
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.models import Sequential
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

pink_colors = ["#ffc0cb", "#ff69b4", "#ff1493", "#db7093", "#c71585"]

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

train_data.head()

train_data.info()

train_data.describe()

train_data.isnull().sum()

sns.boxplot(x='Arrival Delay in Minutes',data = train_data,color='pink')

train_data.dropna(inplace=True)
train_data.isnull().sum()

train_data.duplicated().sum()


test_data.duplicated().sum()

test_data.isnull().sum()

test_data.dropna(inplace=True)
test_data.isnull().sum()

train_data.columns

numeric_columns = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
boxplot = plt.boxplot([train_data[col] for col in numeric_columns],
                      labels=numeric_columns,
                      vert=False,
                      patch_artist=True)
plt.title("Box Plot for Each Column")
for patch in boxplot['boxes']:
    patch.set_color('hotpink')

plt.show()


# Remove outliers using IQR
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

train_data = remove_outliers_iqr(train_data, numeric_columns)
test_data = remove_outliers_iqr(test_data, numeric_columns)

categorical_cols = train_data.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('satisfaction')

label = LabelEncoder()
for col in categorical_cols:
    train_data[col] = label.fit_transform(train_data[col])
    test_data[col] = label.transform(test_data[col])

target_le = LabelEncoder()
train_data['satisfaction'] = target_le.fit_transform(train_data['satisfaction'])
test_data['satisfaction'] = target_le.transform(test_data['satisfaction'])

plt.figure(figsize=(6, 6))
plt.pie(train_data['satisfaction'].value_counts(),
        labels=["Neutral or dissatisfied", "Satisfied"],
        colors=pink_colors,
        autopct='%1.1f%%')

plt.title("Satisfaction Distribution")
plt.show()

numerical_features = ['Age', 'Flight Distance',
                      'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
                      'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment',
                      'On-board service', 'Leg room service', 'Checkin service', 'Inflight service', 'Cleanliness',
                      'Departure Delay in Minutes', 'Arrival Delay in Minutes']

plt.figure(figsize=(20,20))

for i, feature in enumerate(numerical_features, 1):
    plt.subplot(5, 4, i)
    sns.histplot(train_data[feature],kde=True,color='hotpink')
    plt.title(feature)

plt.tight_layout()
plt.show()

heatmap_df=train_data.copy()
heatmap_df['Satisfaction']=heatmap_df['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})


plt.figure(figsize=(20,8),dpi=150)
corr_matrix = heatmap_df._get_numeric_data().corr()
sns.heatmap(corr_matrix, annot=True, cmap=pink_colors, fmt='.2f')

train_data.drop(columns=[
    "id",
    "Gender",
    "Gate location",
    "Arrival Delay in Minutes",
    "Departure Delay in Minutes",
    "Unnamed: 0",
    "Departure/Arrival time convenient"
], inplace=True)

test_data.drop(columns=[
    "id",
    "Gender",
    "Gate location",
    "Arrival Delay in Minutes",
    "Departure Delay in Minutes",
    "Unnamed: 0",
    "Departure/Arrival time convenient"
], inplace=True)

X_train = train_data.drop('satisfaction', axis=1)
y_train = train_data['satisfaction']
X_test = test_data.drop('satisfaction', axis=1)
y_test = test_data['satisfaction']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(" GradientBoosting:")
GradientBoosting_model = GradientBoostingClassifier()
GradientBoosting_model.fit(X_train_scaled, y_train)

y_pred_GradientBoosting = GradientBoosting_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred_GradientBoosting)
print(f"Gradient Boosting Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test,y_pred_GradientBoosting))

xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)


y_pred_xg = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_xg)
print(f"XG Boosting Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_xg))

print("-------------------------------------------------------")

print("LogisticRegression")
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)
print("Number of iterations used:", log_model.n_iter_)
print("Score for training data : ",log_model.score(X_train_scaled, y_train))
print("Score for test data : ",log_model.score(X_test_scaled, y_test))
accuracy = accuracy_score(y_test,y_pred_log)
print(f"Logistic Regrission Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test,y_pred_log))
print("Log Loss:", log_loss(y_test, log_model.predict_proba(X_test)))

print("---------------------------------------------")

print("RandomForestClassifier")
RandomForestClassifier_model = RandomForestClassifier()
RandomForestClassifier_model.fit(X_train_scaled, y_train)

y_pred_RandomForestClassifier = RandomForestClassifier_model.predict(X_test_scaled)
print("Number of estimators (trees):", RandomForestClassifier_model.n_estimators)
print("Score for training data : ", RandomForestClassifier_model.score(X_train_scaled, y_train))
accuracy = accuracy_score(y_test, y_pred_RandomForestClassifier)
print(f"RandomForest Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_RandomForestClassifier))

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred_knn)
print(f"KNN Accuracy: {acc:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn ))

dt_model = DecisionTreeClassifier(criterion='entropy',random_state=42)  # ممكن تحددي max_depth لو عايزة
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
acc = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {acc:.2f}")
print("\nClassification Report:\n", classification_report(y_test,y_pred_dt))

svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)

y_pred_svm = svm_model.predict(X_test_scaled)
print("Score for training data : ", svm_model.score(X_train_scaled, y_train))
print("Score for test data : ", svm_model.score(X_test_scaled, y_test))
accuracy = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_svm))

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU if errors persist

model = Sequential([
    InputLayer(shape=(X_train.shape[1],)),  # Changed from input_shape to shape
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2)

y_pred_nn = model.predict(X_test)
y_pred_classes = (y_pred_nn > 0.5).astype(int)

print("Neural Network Accuracy ", accuracy_score(y_test, y_pred_classes))
print("Classification Report:\n", classification_report(y_test, y_pred_classes))


models = [
    ("Random Forest", y_test, y_pred_RandomForestClassifier),
    ("XGBoost", y_test, y_pred_xg),
    ("Logistic Regression", y_test, y_pred_log),
    ("KNN", y_test, y_pred_knn),
    ("Neural Network", y_test, y_pred_classes),
    ("Gradient Boosting", y_test, y_pred_GradientBoosting),
    ("SVM", y_test, y_pred_svm),
    ("Decision Tree",y_test,y_pred_dt)

]

plt.figure(figsize=(20, 10))
for i, (name, y_true, y_pred) in enumerate(models):
    plt.subplot(2, 4, i+1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=pink_colors, cbar=False, linewidths=0.5)
    plt.title(name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

plt.tight_layout()
plt.show()


results = []

def store_results(model,y_pred, name):
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred)
    })

store_results(RandomForestClassifier_model, y_pred_RandomForestClassifier, 'Random Forest')
store_results(xgb_model,y_pred_xg, 'XGBoost')
store_results(log_model,y_pred_log, 'Logistic Regression')
store_results(knn, y_pred_knn, 'KNN')
store_results(model,  y_pred_classes, 'Neural Network')
store_results(GradientBoosting_model,y_pred_GradientBoosting, 'Gradient Boosting')
store_results(svm_model,y_pred_svm, 'SVM')
store_results(dt_model,y_pred_dt, 'Decision Tree')


df_results = pd.DataFrame(results)
print(df_results.sort_values(by='F1 Score', ascending=False))

df_melted = df_results.melt(id_vars='Model', var_name='Metric', value_name='Score')

plt.figure(figsize=(10,6))
sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric',palette=pink_colors)
plt.xticks(rotation=45)
plt.title("Model Performance Comparison")
plt.tight_layout()
plt.show()



#Saved xgb_model.pkl and scaler.pkl
import joblib
joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Saved xgb_model.pkl and scaler.pkl")
