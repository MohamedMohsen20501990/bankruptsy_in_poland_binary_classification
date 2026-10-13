import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, recall_score
from src.logging_config import logger
from src.utils import save_obj
from src.preprocess import Preprocessor


@dataclass
class ModelTrainerConfig:
    """Store the trained model artifact path."""
    trained_model_path:str = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    """Train, evaluate, and save the best classification model."""
    
    def __init__(self):
        """Initialize the trainer and its tracking attributes."""
        self.model_trainer_config=ModelTrainerConfig()
        self.report= {}
        self.best_model_name= None
        self.best_test_recall=  None
        
    def train_model(self,train_data: pd.DataFrame, test_data:pd.DataFrame, target:str):
        """Tune multiple models, select the best recall, and save it."""
        
        try:
            logger.info("Starting model training.")
            report={}
            best_models ={}
            X_train, y_train, X_test, y_test = (
                train_data.drop(columns=[target]),
                train_data[target],
                test_data.drop(columns=[target]),
                test_data[target]
            )
            
            logger.info(
                "Training data shape: %s, Test data shape: %s",
                train_data.shape,
                test_data.shape
            )
            
            # Numerical features schems for the peprocessor[ColumnTransformer]
            numerical_features = [col for col in X_train.columns if col.startswith("X")]
            
            logger.info(
                "Number of numerical features selected: %d",
                len(numerical_features)
            )       
            
            # Initializing preprocessor object from preprocess module
            preprocessor = Preprocessor()
            
            preprocessor_obj =preprocessor.build_preprocessor(numerical_features=numerical_features)
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "KNN": KNeighborsClassifier(),
                "XGBOOST": XGBClassifier(random_state=42,eval_metric="logloss",n_jobs=1)
            }
            
            params = {
                "Logistic Regression": {"model__C":[0.01, 0.1, 1, 10]},
                "Decision tree": {"model__max_depth":range(5,31,5)},
                "Random Forest": {"model__max_depth":range(5,21,5), "model__n_estimators":[50, 100, 150]},
                "Gradient Boosting": {"model__max_depth": range(5,21,5), "model__n_estimators": [50,100,150], 'model__learning_rate': [0.01, 0.1, 0.2]},
                "KNN": {"model__n_neighbors": range(5,16,5)},
                "XGBOOST": {"model__max_depth": [5, 10, 15, 20], 'model__learning_rate': [0.01, 0.1, 0.2]}
            }
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            for model_name, model in models.items():
                logger.info("Starting hyperparameter tuning for %s.", model_name)
                pipeline=Pipeline(steps=[
                    ("preprocessor", preprocessor_obj),
                    ("sampler", RandomOverSampler(random_state=42)),
                    ("model", model)
                ])
                
                gs_model = GridSearchCV(
                    estimator=pipeline,
                    param_grid=params[model_name],
                    scoring="recall",
                    cv=cv,
                    n_jobs=-1,
                    verbose=1
                )
                gs_model.fit(X_train, y_train)
                best_model = gs_model.best_estimator_                
                report[model_name] ={"best_recall": gs_model.best_score_, "best params": gs_model.best_params_}
            
                best_models[model_name]=best_model
                
                logger.info(
                    "Completed tuning for %s. Best CV recall: %.4f",
                    model_name,
                    gs_model.best_score_
                )
                logger.info(
                    "Best parameters for %s: %s",
                    model_name,
                    gs_model.best_params_
                )
            self.report=report
                
            best_model_name, best_model_info = max(
                self.report.items(),
                key=lambda item: item[1]["best_recall"] 
            )
            
            best_model = best_models[best_model_name]
            logger.info("Selected best model: %s", best_model_name)
            
            recall = recall_score(y_test, best_model.predict(X_test), pos_label=1)
            # recall = classification_report(y_test, best_model.predict(X_test), output_dict=True)["1"]["recall"]
            self.best_model_name= best_model_name
            self.best_test_recall = recall
            
            logger.info("Test recall: %.4f", recall)
            save_obj(obj=best_model, file_path=self.model_trainer_config.trained_model_path)
            
            logger.info(
                "Best model saved to %s.",
                self.model_trainer_config.trained_model_path
            )
                        
            print(f"Best model: {best_model_name}")
            print(f"Recall score: {best_model_info['best_recall']}")
            print(f"Test recall: {recall}")            
            
            return best_model

        except Exception:
            logger.exception("Error training Model")
            raise
            
        
        
            
    