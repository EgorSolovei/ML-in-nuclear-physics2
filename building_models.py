import optuna
import pickle
import pandas as pd
import numpy as np
import sklearn.metrics as metrics

from optuna.pruners import MedianPruner, PatientPruner
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier


def decrease_data(df, size=10000):
    # Функция уменьшения входящего DataFrame до размера size. Отбрасываются события 0 класса
    data_class1 = df[df["class"] == 1]
    data_class0 = df[df["class"] == 0][:(size - data_class1.shape[0])]
    return pd.concat([data_class0, data_class1], ignore_index=True)


def search_best_model(name_exp):
    data = pd.read_csv(name_exp + "/data.csv")
    data["class"] = data["class"].astype(int)
    small_data = decrease_data(data)

    X = small_data.drop(["class"], axis=1)
    y = small_data["class"]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    def study_best_model(name_model, params):
        # обучим модель с наилучшими параметрами
        best_model = 0
        if name_model == "DecisionTree":
            best_model = DecisionTreeClassifier(**params)
        elif name_model == "RandomForest":
            best_model = RandomForestClassifier(**params)
        elif name_model == "CatBoost":
            best_model = CatBoostClassifier(**params)
        best_model.fit(x_train, y_train)
        write_metrics(y_train, best_model.predict(x_train), name_model, train=True)
        write_metrics(y_test, best_model.predict(x_test), name_model, train=False)

        # сохраним обученную модель
        with open(name_exp + f"/{name_model}.pickle", "wb") as f:
            pickle.dump(best_model, f)
        print(f"Метрики наилучшей модели {name_model} записаны. Модель сохранена")

    def write_metrics(y_true, y_score, title, train):
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

        with open(name_exp + f"/Result best {title}.txt", "a") as file:
            if train:
                file.write("Result on train data\n")
            else:
                file.write("--------------------------------------\n")
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

    list_name = ["DecisionTree", "RandomForest", "CatBoost"]
    mode = "Usually"

    for name in list_name:
        name_alg = name

        def optuna_optimize(trial):  # Необходимо задать пространство для поиска параметров.
            model = -1
            if name_alg == "DecisionTree":
                model = DecisionTreeClassifier(
                    max_depth=trial.suggest_int("max_depth", 5, 20, 1),
                    criterion=trial.suggest_categorical("criterion", ['gini', 'entropy', 'log_loss'])
                )
            elif name_alg == "RandomForest":
                model = RandomForestClassifier(
                    n_estimators=trial.suggest_int("n_estimators", 10, 300, 5),
                    max_depth=trial.suggest_int("max_depth", 5, 25, 1),
                    n_jobs=-1
                )
            elif name_alg == "CatBoost":
                model = CatBoostClassifier(
                    iterations=trial.suggest_int("iterations", 100, 1001, 10),
                    max_depth=trial.suggest_int("max_depth", 2, 9, 1),
                    learning_rate=trial.suggest_float("learning_rate", 0.0001, 0.0101, 0.0005),
                    verbose=False
                )

            # обучим модель
            score = 0
            if mode == "Usually":
                model.fit(x_train, y_train)
                score = metrics.recall_score(y_test, model.predict(x_test))
            elif mode == "Cross val":
                score = cross_val_score(model, x_train, y_train, cv=3, scoring="recall").mean()
            return score

        # создадим объект исследования
        # если дальнейший перебор параметров бессмысленный, то обучение прерывается - это PatientPruner
        study_model = optuna.create_study(study_name=name_alg, direction='maximize',
                                          pruner=PatientPruner(MedianPruner(), patience=3))

        # запустим этап оптимизации параметров
        study_model.optimize(optuna_optimize, n_trials=50, n_jobs=-1)
        # обучим модель с лучшими параметрами
        study_best_model(name_alg, params=study_model.best_params)


if __name__ == "__main__":
    search_best_model("4meters_1ring_4angle")
