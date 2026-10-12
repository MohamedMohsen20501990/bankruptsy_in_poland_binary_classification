import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from dataclasses import dataclass
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from src.logging_config import logger
from src.utils import save_obj

@dataclass
class ModelTrainerConfig:
    trained_model_path:str = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        self.report= {}
        self.best_model_name= None
        self.best_test_recall=  None
        
    def train_model(self,train_data: pd.DataFrame, test_data:pd.DataFrame, target:str):
        
        try:
            report={}
            best_models ={}
            X_train, y_train, X_test, y_test = (
                train_data.drop(columns=[target]),
                train_data[target],
                test_data.drop(columns=[target]),
                test_data[target]
            )
            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "KNN": KNeighborsClassifier(),
                "XGBOOST": XGBClassifier()
            }
            
            params = {
                "Logistic Regression": {"max_iter":[500, 1000]},
                "Decision tree": {"max_depth":range(5,31,5)},
                "Random Forest": {"max_depth":range(5,31,5), "n_estimators":[50, 100, 150]},
                "Gradient Boosting": {"max_depth": range(5,31,5), "n_estimators": [50,100,150], 'learning_rate': [0.01, 0.1, 0.2]},
                "KNN": {"n_neighbors": range(5,16,5)},
                "XGBOOST": {"max_depth": range(5,31,5), "n_estimators": [50,100,150], 'learning_rate': [0.01, 0.1, 0.2]}
            }
            
            for model_name, model in models.items():
                gs_model = GridSearchCV(
                    estimator=model,
                    param_grid=params[model_name],
                    scoring="recall",
                    cv=5,
                    n_jobs=-1,
                    verbose=1
                )
                gs_model.fit(X_train, y_train)
                best_model = gs_model.best_estimator_                
                report[model_name] ={"best_recall": gs_model.best_score_, "best params": gs_model.best_params_}
            
                best_models[model_name]=best_model
                
            self.report=report
                
            best_model_name, best_model_info = max(
                self.report.items(),
                key=lambda item: item[1]["best_recall"] 
            )
            
            best_model = best_models[best_model_name]
            
            recall = classification_report(y_test, best_model.predict(X_test), output_dict=True)["1"]["recall"]
            self.best_model_name= best_model_name
            self.best_test_recall = recall
            
            save_obj(obj=best_model, file_path=self.model_trainer_config.trained_model_path)
            
            print(f"Best model: {best_model_name}")
            print(f"Recall score: {best_model_info['best_recall']}")
            print(f"Test recall: {recall}")            
            
            return best_model

        except Exception:
            logger.exception("Error training Model")
            raise
            
        
        
            
    