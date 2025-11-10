"""Script to generate counterfactuals using pre-trained model."""
import logging
import random
import numpy as np
import argparse
from pathlib import Path

from src.pipeline.config import ExperimentConfig, DATASETS, get_cf_configurations
from src.pipeline.data_pipeline import DataPipeline
from src.pipeline.model_pipeline import ModelPipeline
from src.pipeline.ltl_pipeline import LTLPipeline
from src.pipeline.cf_pipeline import CFPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_cfs(
    dataset_name: str,
    prefix_length: int,
    ltl_percentage: str = '50%',
    model_file: str = None,
    seed: int = 42
):
    """Generate counterfactuals."""
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    
    # Create config
    config = ExperimentConfig(
        dataset_name=dataset_name,
        prefix_length=prefix_length,
        ltl_percentage=ltl_percentage,
        seed=seed
    )
    
    # Load and encode data
    data_pipeline = DataPipeline(config.to_dict())
    encoder, full_df = data_pipeline.load_and_encode()
    train_df, val_df, test_df = data_pipeline.split_data()

    
    # Load or train model
    model_pipeline = ModelPipeline(config.to_dict(), config.get_experiment_id())
    
    if model_file:
        model_pipeline.load(model_file)
        logger.info("Using pre-trained model")
    else:
        logger.info("No model file provided, training new model")
        model_pipeline.train(train_df, val_df)
    
    # Evaluate model
    actual, predicted, scores, metrics = model_pipeline.evaluate(test_df)

    # Parse LTL model
    ltl_pipeline = LTLPipeline()
    ltl_model, dfa, model_names = ltl_pipeline.parse_model(config.get_ltl_model())

    # Optional: Run conformance checking
    event_log = data_pipeline.create_event_log()
    conf_results = ltl_pipeline.check_conformance(event_log)
    
    # Prepare CF pipeline
    cf_pipeline = CFPipeline(
        config=config.to_dict(),
        predictive_model=model_pipeline.predictive_model,
        encoder=encoder,
        ltl_model=ltl_model,
        dfa=dfa,
        model_names=model_names,
        experiment_id=config.get_experiment_id()
    )
    
    # Get query instances
    query_instances = cf_pipeline.prepare_query_instances(test_df, actual, predicted)
    
    # Generate CFs for all configurations
    for cf_config in get_cf_configurations():
        cf_pipeline.generate_counterfactuals(
            query_instances=query_instances,
            full_df=full_df,
            train_df=train_df,
            val_df=val_df,
            cf_config=cf_config,
            ltl_percentage=ltl_percentage
        )
    
    logger.info("All counterfactuals generated!")

def main():
    parser = argparse.ArgumentParser(description='Generate counterfactuals')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=list(DATASETS.keys()),
                       help='Dataset name')
    parser.add_argument('--prefix', type=int, required=True,
                       help='Prefix length')
    parser.add_argument('--ltl-percentage', type=str, default='50%',
                       choices=['10%', '25%', '50%'],
                       help='LTL model complexity percentage')
    parser.add_argument('--model-file', type=str, default=None,
                       help='Path to pre-trained model file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()

    if args.model_file == None:
        args.model_file = f"models/{args.dataset}_prefix{args.prefix}_model.pkl"

    generate_cfs(
        dataset_name=args.dataset,
        prefix_length=args.prefix,
        ltl_percentage=args.ltl_percentage,
        model_file=args.model_file,
        seed=args.seed
    )

if __name__ == '__main__':
    main()