import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pacmap
import umap
import trimap
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # ConfusionMatrixDisplay
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# Загрузка данных
data = pd.read_csv('horse-colic.data', header=None, sep=r'\s+', na_values='?')
X_train, y_train = data.drop(columns=[23]).fillna(data.mean()).values, data.iloc[:, 23]

test = pd.read_csv('horse-colic.test', header=None, sep=r'\s+', na_values='?')
X_test, y_test = test.drop(columns=[23]).fillna(test.mean()).values, test[23]

# Обработка некорректных значений
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)


# Функции для оценки модели
def evaluate_model(model: BaseEstimator) -> None:
    evaluate_model_per_sample(model, 'Обучающая', X_train, y_train)
    evaluate_model_per_sample(model, 'Тестовая', X_test, y_test)

    if isinstance(model[-1], SVC):
        support_vectors_count = model[-1].n_support_
        print(f"Число опорных векторов для каждого класса: {support_vectors_count}")


def evaluate_model_per_sample(model: BaseEstimator,
                              sample_name: str,
                              X: np.ndarray,
                              y: np.ndarray) -> None:
    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')

    print(
        f"[{sample_name}] Точность: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
        f"F1-мера: {f1:.4f}")


# Функции для визуализации
def visualize(X: np.ndarray, y: np.ndarray, method_name: str, support_vectors=None, ax=None) -> None:
    classes = np.unique(y)
    colors = plt.colormaps['Set1']

    for clazz in classes:
        idx = y == clazz
        ax.scatter(X[idx, 0], X[idx, 1], alpha=0.45, c=[colors(clazz)], label=f"Класс {clazz}")
        if support_vectors is not None:
            support_vector_idx = support_vectors[np.isin(support_vectors, np.where(idx)[0])]
            ax.scatter(X[support_vector_idx, 0], X[support_vector_idx, 1],
                       marker='D', alpha=0.6, c=[colors(clazz)],
                       edgecolors='black', label=f"Опорные векторы класса {clazz}")

    ax.set_title(f"Визуализация методом {method_name}")
    ax.legend()


def perform_visualization(X: np.ndarray, y: np.ndarray, model: BaseEstimator, support_vectors=None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Сравнение методов уменьшения размерности", fontsize=16)

    tsne_reducer = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    X_tsne = tsne_reducer.fit_transform(X)
    visualize(X_tsne, y, "t-SNE", support_vectors, ax=axes[0, 0])

    umap_reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
    X_umap = umap_reducer.fit_transform(X)
    visualize(X_umap, y, "UMAP", support_vectors, ax=axes[0, 1])

    trimap_reducer = trimap.TRIMAP(n_inliers=10, n_outliers=5, n_random=5)
    X_trimap = trimap_reducer.fit_transform(X)
    visualize(X_trimap, y, "TriMAP", support_vectors, ax=axes[1, 0])

    pacmap_reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0)
    X_pacmap = pacmap_reducer.fit_transform(X)
    visualize(X_pacmap, y, "PacMAP", support_vectors, ax=axes[1, 1])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_classification_results(X: np.ndarray, y: np.ndarray, model: BaseEstimator) -> None:
    colors = plt.colormaps['Set1']
    y_pred = model.predict(X)

    pacmap_reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0)
    X = pacmap_reducer.fit_transform(X)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for clazz in np.unique(y):
        idx = y == clazz
        plt.scatter(X[idx, 0], X[idx, 1],
                    label=f"Класс {clazz}",
                    alpha=0.6,
                    s=50,
                    c=[colors(clazz)])

    plt.title('Известные метки')
    plt.legend()

    plt.subplot(1, 2, 2)
    for clazz in np.unique(y_pred):
        idx = y_pred == clazz
        plt.scatter(X[idx, 0], X[idx, 1],
                    label=f"Класс {clazz}",
                    alpha=0.6,
                    s=50,
                    c=[colors(clazz)])

    plt.title('Предсказанные метки')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Основная программа с выбором классификатора
while True:
    print("\nВыберите классификатор:")
    print("1. KNN-классификатор")
    print("2. SVM-классификатор")
    print("3. Random Forest-классификатор")

    choose = int(input("Введите номер (1, 2 или 3): "))

    if choose == 1:
        parameters = {
            'kneighborsclassifier__n_neighbors': [3, 5, 7, 10],
            'kneighborsclassifier__metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
            'kneighborsclassifier__weights': ['uniform', 'distance']
        }
        clf = make_pipeline(MinMaxScaler(), KNeighborsClassifier())
        print("Запуск KNN-классификатора...")

    elif choose == 2:
        parameters = {
            'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'svc__C': [0.1, 1, 10, 100],
            'svc__gamma': ['scale', 'auto'],
            'svc__degree': [2, 3, 4]
        }
        clf = make_pipeline(MinMaxScaler(), SVC())
        print("Запуск SVM-классификатора...")

    elif choose == 3:
        parameters = {
            'randomforestclassifier__n_estimators': [50, 100, 200],
            'randomforestclassifier__max_depth': [None, 10, 20, 30],
            'randomforestclassifier__min_samples_split': [2, 5, 10],
            'randomforestclassifier__min_samples_leaf': [1, 2, 4],
            'randomforestclassifier__bootstrap': [True, False]
        }
        clf = make_pipeline(MinMaxScaler(), RandomForestClassifier())
        print("Запуск Random Forest-классификатора...")

    elif choose == 0:
        print("Завершение работы программы.")
        break

    else:
        print("Неверный выбор, попробуйте снова.")
        continue

    grid = GridSearchCV(clf, parameters, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print(f"\nЛучшие параметры: {grid.best_params_}\n")

    evaluate_model(best_model)
    perform_visualization(X_train, best_model.predict(X_train), best_model)
    plot_classification_results(X_train, y_train, best_model)
