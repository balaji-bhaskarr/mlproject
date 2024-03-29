import os
import sys
from dataclasses import dataclass
#from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
#from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
from src.pipeline.train_pipeline import TrainPipeline

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Split training and test input data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                #"XGBRegressor": XGBRegressor(),
                #"CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params={
                "Decision Tree": {
                    'criterion':['entropy', 'log_loss', 'gini'],
                    #'splitter':['best','random'],
                    #'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    #'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    #'criterion':['gini', 'log_loss', 'entropy'],
                    #'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    #'loss':['squared_error', 'huber', 'absolute_error', 'exponential', 'log_loss'],
                    #'learning_rate':[.1,.01,.05,.001],
                    #'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    #'criterion':['squared_error', 'friedman_mse'],
                    #'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "K-Neighbors Classifier":{},
                #"XGBRegressor":{
                    #'learning_rate':[.1,.01,.05,.001],
                    #'n_estimators': [8,16,32,64,128,256]
                #},
                #"CatBoosting Regressor":{
                    #'depth': [6,8,10],
                    #'learning_rate': [0.01, 0.05, 0.1],
                    #'iterations': [30, 50, 100]
                #},
                "AdaBoost Regressor":{
                    #'learning_rate':[.1,.01,0.5,.001],
                    #'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            train_pipeline = TrainPipeline(X_train, y_train, X_test, y_test)   
            best_model_name, best_model_score = train_pipeline.train(models=models, params=params)
            best_model = models[best_model_name] 

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info('Best found model on testing dataset')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            logging.info(f'R2_score on best model {best_model_name} = {r2_square}')
            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)
    def evaluate():
        pass
