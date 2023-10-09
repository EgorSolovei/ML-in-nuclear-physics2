import pickle
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier


# Функция уменьшения входящего DataFrame до размера size. Отбрасываются события 0 класса
def decrease_data(df, size):
    data_class1 = df[df["class"] == 1]
    data_class0 = df[df["class"] == 0][:(size - data_class1.shape[0])]
    return pd.concat([data_class0, data_class1], ignore_index=True)


# Задаём пространство для поиска параметров
def define_params(name_alg):
    if name_alg == "DecisionTree":
        model = DecisionTreeClassifier()
        params = {"max_depth": range(5, 21), "criterion": ['gini', 'entropy', 'log_loss'],
                  "class_weight": [{0: 1, 1: 5}, {0: 1, 1: 10}]}
        return model, params
    elif name_alg == "RandomForest":
        model = RandomForestClassifier()
        params = {"n_estimators": range(100, 301, 10), "max_depth": range(5, 11), "n_jobs": [-1],
                  "class_weight": [{0: 1, 1: 5}, {0: 1, 1: 10}]}
        return model, params
    elif name_alg == "CatBoost":
        model = CatBoostClassifier(verbose=False)
        params = {"iterations": range(100, 1001, 50), "max_depth": range(2, 9, 1), "learning_rate": [5e-4, 1e-3],
                  "class_weights": [{0: 1, 1: 5}, {0: 1, 1: 10}]}
        return model, params


def write_metrics(y_true, y_score, path, title, train):
    # True positive
    tp = np.sum(y_true * y_score)
    # False positive
    fp = np.sum((y_true == 0) * y_score)
    # True negative
    tn = np.sum((y_true == 0) * (y_score == 0))
    # False negative
    fn = np.sum(y_true * (y_score == 0))

    # True positive rate (sensitivity or recall)
    tpr = tp / (tp + fn)
    # False positive rate (fall-out)
    fpr = fp / (fp + tn)
    # Precision
    precision = tp / (tp + fp)
    # True negatvie tate (specificity)
    tnr = 1 - fpr
    # F1 score
    f1 = 2 * tp / (2 * tp + fp + fn)
    # ROC-AUC for binary classification
    auc = (tpr + tnr) / 2
    # MCC. Коэффициент корреляции Мэтьюса. [-1, 1]
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    with open(path + f"/Result {title}.txt", "a") as file:
        if train:
            file.write("Result on train data\n")
        else:
            file.write("-" * 30 + "\n")
            file.write("Result on test data\n")
        file.write(f"True positive: {tp}\n")
        file.write(f"False positive: {fp}\n")
        file.write(f"True negative: {tn}\n")
        file.write(f"False negative: {fn}\n\n")

        file.write(f"True positive rate (recall): {tpr:.3f}\n")
        file.write(f"False positive rate: {fpr:.3f}\n")
        file.write(f"Precision: {precision:.3f}\n")
        file.write(f"True negative rate: {tnr:.3f}\n")
        file.write(f"F1: {f1:.3f}\n")
        file.write(f"ROC-AUC: {auc:.3f}\n")
        file.write(f"MCC: {mcc:.3f}\n\n")


def gridCV(x_train, y_train, model, params, cv=5):
    modelCV = GridSearchCV(model, params, scoring="f1", cv=cv)
    modelCV.fit(x_train, y_train)

    best_model = modelCV.best_estimator_
    history = pd.DataFrame(modelCV.cv_results_)
    return history, best_model


def randomCV(x_train, y_train, model, params, cv=5):  # пока не используется
    modelCV = RandomizedSearchCV(model, params, scoring="f1", cv=cv)
    modelCV.fit(x_train, y_train)

    best_model = modelCV.best_estimator_
    history = pd.DataFrame(modelCV.cv_results_)
    return history, best_model


def searchCV_model(name_exp, size_data):
    data = pd.read_csv(name_exp + "/data.csv")
    data["class"] = data["class"].astype(int)

    new_data = decrease_data(data, size_data).sample(frac=1)  # перемешивает полученный DataFrame
    X = new_data.drop(["class"], axis=1)
    y = new_data["class"]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    list_name = ["DecisionTree", "RandomForest", "CatBoost"]
    for name in list_name:
        estimator, model_params = define_params(name)
        model_hist, best_estimator = gridCV(x_train, y_train, estimator, model_params)

        path = name_exp + f"/{name}"
        os.mkdir(path)  # отдельная папка для каждой модели
        model_hist.to_csv(path + f"/history_{name}.csv", index=False)  # сохраним историю обучения модели

        best_estimator.fit(x_train, y_train)
        write_metrics(y_train, best_estimator.predict(x_train), path, f"{name}", True)
        write_metrics(y_test, best_estimator.predict(x_test), path, f"{name}", False)

        with open(path + f"/{name}.pkl", "wb") as f:
            pickle.dump(best_estimator, f)


if __name__ == "__main__":
    searchCV_model("4meters_1ring_4angle", 10000)
