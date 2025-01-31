import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score

# 1. Загрузка данных
train_df = pd.read_csv("Sources/heart_adapt_train.csv")
test_df = pd.read_csv("Sources/heart_adapt_test.csv")

# Описание структуры набора данных
def describe_data(df):
    print("Описание структуры набора данных:")
    print(df.describe(include='all'))
    for column in df.columns:
        print(f"\nАтрибут: {column}")
        print(f"Текстовое описание: {column}")
        print(f"Цифровое описание: {df[column].describe()}")
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], kde=False)
        plt.title(f'Графическое описание: {column}')
        plt.savefig(f'Diagrams/{column}_histogram.png')
        plt.close()

describe_data(train_df)

# 2. Предобработка данных
def preprocess_data(df, encoder=None, scaler=None):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

    numeric_imputer = SimpleImputer(strategy="mean")
    df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

    categorical_imputer = SimpleImputer(strategy="most_frequent")
    df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, drop="first")
        encoded = pd.DataFrame(encoder.fit_transform(df[categorical_cols]))
        encoded.columns = encoder.get_feature_names_out(categorical_cols)
    else:
        encoded = pd.DataFrame(encoder.transform(df[categorical_cols]))
        encoded.columns = encoder.get_feature_names_out(categorical_cols)

    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, encoded], axis=1)

    continuous_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    if scaler is None:
        scaler = StandardScaler()
        df[continuous_cols] = scaler.fit_transform(df[continuous_cols])
    else:
        df[continuous_cols] = scaler.transform(df[continuous_cols])

    return df, encoder, scaler

train_df, encoder, scaler = preprocess_data(train_df)
test_df, _, _ = preprocess_data(test_df, encoder, scaler)

X_train, y_train = train_df.drop("HeartDisease", axis=1), train_df["HeartDisease"]
X_test, y_test = test_df.drop("HeartDisease", axis=1), test_df["HeartDisease"]

# 3.1 Проверка баланса классов

# Дисбаланс принадлежности к классам отсутствует
print("Распределение классов в обучающих данных:")
print(y_train.value_counts())

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# 3.2 Кластеризация
print("\nКластеризация данных на сбалансированной выборке:")
kmeans = KMeans(n_clusters=3, random_state=42)
dbscan = DBSCAN()
agglomerative = AgglomerativeClustering(n_clusters=3)

labels_kmeans = kmeans.fit_predict(X_train_balanced)
labels_dbscan = dbscan.fit_predict(X_train_balanced)
labels_agglomerative = agglomerative.fit_predict(X_train_balanced)

# Оценка качества кластеризации
silhouette_kmeans = silhouette_score(X_train_balanced, labels_kmeans)
silhouette_dbscan = silhouette_score(X_train_balanced, labels_dbscan)
silhouette_agglomerative = silhouette_score(X_train_balanced, labels_agglomerative)

dbi_kmeans = davies_bouldin_score(X_train_balanced, labels_kmeans)
dbi_dbscan = davies_bouldin_score(X_train_balanced, labels_dbscan)
dbi_agglomerative = davies_bouldin_score(X_train_balanced, labels_agglomerative)

# Визуализация распределения данных по кластерам
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.scatter(X_train_balanced.iloc[:, 0], X_train_balanced.iloc[:, 1], c=labels_kmeans, cmap='viridis')
plt.title('KMeans Clustering')

plt.subplot(1, 3, 2)
plt.scatter(X_train_balanced.iloc[:, 0], X_train_balanced.iloc[:, 1], c=labels_dbscan, cmap='viridis')
plt.title('DBSCAN Clustering')

plt.subplot(1, 3, 3)
plt.scatter(X_train_balanced.iloc[:, 0], X_train_balanced.iloc[:, 1], c=labels_agglomerative, cmap='viridis')
plt.title('Agglomerative Clustering')

plt.savefig('clustering.png')
plt.close()

# Определение лучшего алгоритма кластеризации
best_algorithm = "KMeans"
if silhouette_dbscan > silhouette_kmeans and silhouette_dbscan > silhouette_agglomerative:
    best_algorithm = "DBSCAN"
elif silhouette_agglomerative > silhouette_kmeans and silhouette_agglomerative > silhouette_dbscan:
    best_algorithm = "AgglomerativeClustering"

print(f"Лучший алгоритм кластеризации: {best_algorithm}")

# 3.3 Обучение и тестирование моделей
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train_balanced, y_train_balanced)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\n{name}")
    print("Точность:", accuracy)
    print(classification_report(y_test, predictions))

    if accuracy > best_accuracy:
        best_model = model
        best_accuracy = accuracy

# Сохранение лучшей модели
joblib.dump(best_model, "best_model.pkl")
print(f"\nЛучшая модель: {type(best_model).__name__} с точностью {best_accuracy}")

# 3.4 Непрерывное обучение
print("\nСимуляция непрерывного обучения:")
new_data = pd.DataFrame({
    'Age': [50, 60],
    'Sex': ['M', 'F'],
    'ChestPainType': ['ASY', 'NAP'],
    'RestingBP': [130, 140],
    'Cholesterol': [200, 210],
    'FastingBS': [0, 1],
    'RestingECG': ['Normal', 'ST'],
    'MaxHR': [150, 160],
    'ExerciseAngina': ['N', 'Y'],
    'Oldpeak': [1.0, 2.5],
    'ST_Slope': ['Flat', 'Up'],
    'HeartDisease': [1, 0]
})

new_data, _, _ = preprocess_data(new_data, encoder, scaler)
X_new, y_new = new_data.drop("HeartDisease", axis=1), new_data["HeartDisease"]

for col in X_train.columns:
    if col not in X_new.columns:
        X_new[col] = 0

model = joblib.load("best_model.pkl")
model.fit(X_new, y_new)
joblib.dump(model, "updated_model.pkl")

updated_predictions = model.predict(X_test)
print("\nОбновленная модель")
print("Точность:", accuracy_score(y_test, updated_predictions))
print(classification_report(y_test, updated_predictions))
