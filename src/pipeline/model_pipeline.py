"""Model training and evaluation pipeline."""
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from pathlib import Path
import pickle

from src.predictive_model.predictive_model import PredictiveModel, drop_columns
from src.predictive_model.common import ClassificationMethods, get_tensor
from src.hyperparameter_optimisation.common import retrieve_best_model
from src.evaluation.common import evaluate_classifier

logger = logging.getLogger(__name__)

class ModelPipeline:
    """Handles model training, evaluation, and persistence."""
    
    def __init__(self, config: dict, experiment_id: str):
        self.config = config
        self.experiment_id = experiment_id
        self.predictive_model: Optional[PredictiveModel] = None
        self.metrics: Optional[Dict] = None
        
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Train predictive model with hyperparameter optimization."""
        logger.info("=" * 80)
        logger.info(f"Training model for experiment: {self.experiment_id}")
        logger.info("=" * 80)
        
        self.predictive_model = PredictiveModel(
            self.config,
            self.config['predictive_model'],
            train_df,
            val_df
        )
        
        logger.info("Starting hyperparameter optimization")
        self.predictive_model.model, self.predictive_model.config = retrieve_best_model(
            self.predictive_model,
            self.config['predictive_model'],
            max_evaluations=self.config['hyperparameter_optimisation_epochs'],
            target=self.config['hyperparameter_optimisation_target'],
            seed=self.config['seed']
        )
        
        logger.info("Model training complete")
        return self.predictive_model
    
    def evaluate(
        self,
        test_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Evaluate model on test set."""
        if self.predictive_model is None:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info("Evaluating model on test set")
        
        # Get predictions
        if self.predictive_model.model_type == ClassificationMethods.LSTM.value:
            probabilities = self.predictive_model.model.predict(
                get_tensor(self.config, drop_columns(test_df))
            )
            predicted = np.argmax(probabilities, axis=1)
            scores = np.amax(probabilities, axis=1)
            actual = np.array(test_df['label'].to_list())
        else:
            predicted = self.predictive_model.model.predict(drop_columns(test_df))
            scores = self.predictive_model.model.predict_proba(drop_columns(test_df))[:, 1]
            actual = test_df['label'].values
        
        # Compute metrics
        accuracy = np.mean(actual == predicted)
        self.metrics = evaluate_classifier(actual, predicted, scores)
        self.metrics['accuracy'] = accuracy
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Metrics: {self.metrics}")
        
        return actual, predicted, scores, self.metrics
    
    def save(self, output_dir: str = 'models'):
        """Save trained model to disk."""
        if self.predictive_model is None:
            raise ValueError("No model to save")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_file = output_path / f"{self.experiment_id}_model.pkl"
        
        with open(model_file, 'wb') as f:
            pickle.dump({
                'model': self.predictive_model.model,
                'config': self.predictive_model.config,
                'model_type': self.predictive_model.model_type,
                'metrics': self.metrics,
                'experiment_id': self.experiment_id
            }, f)
        
        logger.info(f"Model saved to: {model_file}")
        return model_file
    
    def load(self, model_file: str):
        """Load trained model from disk."""
        logger.info(f"Loading model from: {model_file}")
        
        with open(model_file, 'rb') as f:
            saved_data = pickle.load(f)
        
        # Reconstruct PredictiveModel
        self.predictive_model = PredictiveModel(
            self.config,
            saved_data['model_type'],
            None,  # No training data needed for loaded model
            None
        )
        self.predictive_model.model = saved_data['model']
        self.predictive_model.config = saved_data['config']
        self.predictive_model.model_type = saved_data['model_type']
        self.metrics = saved_data.get('metrics')
        
        logger.info("Model loaded successfully")
        return self.predictive_model