from ucimlrepo import fetch_ucirepo
# import pandas as pd
from sklearn.model_selection import train_test_split  # Для обучающей и тестовых выборках
from sklearn.svm import SVC  # Для обучения модели с линейным ядром
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV  # Для: Перебор по сетке (grid search).
from sklearn.impute import SimpleImputer  # Для заполнения пропущенных значений
import time  # Для измерения времени выполнения


# Загружаю сам набор данных
horse_colic = fetch_ucirepo(id=47)

# Разделяю тут данные на признаки Х и целевые метки У для дальнейшего использования
X = horse_colic.data.features
y = horse_colic.data.targets

# Используем медиану для замены пропущенных значений (NaN), ибо программа не работала с этими пустыми значениями
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Преобразую в одномерный массив (там тоже прога из-за этого отказывалась, ибо у меня, мол, матрица (пошла куда
# подальше) )
y = y.values.ravel()

# Тут реализую пункт "Обучить, проверить качество классификатора на обучающей и тестовой
# выборках". То есть разделил данные на 70 % обучающей и тестовую 30 % выборки
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)

# # Измеряем время обучения модели
start_time = time.time()

# Как раз-таки содание SVM классификатора с линейным ядром (linear) и обучением его
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train, y_train)

# Прогнозирование на тест. выборке
y_pred = svm_model.predict(X_test)

# ОЦЕНКА КАЧЕСТВА КЛАССИФИКАТОРА
accuracy = accuracy_score(y_test, y_pred)  # Точность
recall = recall_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Сделал, чтобы просто посмотреть результат метрик
print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"F1 Score: {f1}")

# Тут из методички "Оценить число опорных векторов."
print("Number of support vectors per class:", svm_model.n_support_)

# Вывод времени обучения модели
print("Training and evaluation took %.2f seconds" % (time.time() - start_time))

# Перебор по сетке (grid search).

# Определяем параметры для поиска (тут и линейное, RBF, полиномиальное,
# сигмоидное)
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

# Создаю объект для дальнейшей работы
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
# Поиск по сетке
start_time_grid = time.time()
grid_search.fit(X_train, y_train)

# Вывод лучших параметров (потом можно переделать)
print("Best parameters:", grid_search.best_params_)

# Вывод времени поиска по сетке
print("Grid search took %.2f seconds" % (time.time() - start_time_grid))
