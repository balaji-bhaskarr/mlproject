from src.utils import evaluate_models

class TrainPipeline:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train(self, models, params):
            model_report:dict = evaluate_models(X_train=self.X_train, y_train=self.y_train, X_test=self.X_test, y_test=self.y_test, models=models, params=params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            return best_model_name, best_model_score