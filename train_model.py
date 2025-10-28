"""Script to train predictive models."""
import logging
import random
import numpy as np
import argparse
from pathlib import Path

from src.pipeline.config import ExperimentConfig, DATASETS
from src.pipeline.data_pipeline import DataPipeline
from src.pipeline.model_pipeline import ModelPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_model(
    dataset_name: str,
    prefix_length: int,
    seed: int = 42,
    save_model: bool = True
):
    """Train a single model."""
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    
    # Create config
    config = ExperimentConfig(
        dataset_name=dataset_name,
        prefix_length=prefix_length,
        seed=seed
    )
    
    # Load and encode data
    data_pipeline = DataPipeline(config.to_dict())
    encoder, full_df = data_pipeline.load_and_encode()
    
    # Split data
    train_df, val_df, test_df = data_pipeline.split_data()
    
    # Train model
    model_pipeline = ModelPipeline(config.to_dict(), config.get_experiment_id())
    model_pipeline.train(train_df, val_df)
    
    # Evaluate
    actual, predicted, scores, metrics = model_pipeline.evaluate(test_df)
    
    # Save if requested
    if save_model:
        model_path = model_pipeline.save()
        logger.info(f"Model saved to: {model_path}")
    
    return model_pipeline, metrics

def main():
    parser = argparse.ArgumentParser(description='Train predictive models')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=list(DATASETS.keys()),
                       help='Dataset name')
    parser.add_argument('--prefix', type=int, required=True,
                       help='Prefix length')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save model')
    
    args = parser.parse_args()
    
    model_pipeline, metrics = train_model(
        dataset_name=args.dataset,
        prefix_length=args.prefix,
        seed=args.seed,
        save_model=not args.no_save
    )
    
    print("\nFinal Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

if __name__ == '__main__':
    main()