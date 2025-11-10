"""Run the complete LTLf counterfactual pipeline."""
import logging
import warnings
import argparse
from src.pipeline.config import DATASETS

from train_model import train_model
from generate_counterfactuals import generate_cfs

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    filename='pipeline.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_all_experiments(
    datasets: list = None,
    ltl_percentages: list = None,
    seed: int = 42
):
    """Run all experiments."""
    if datasets is None:
        datasets = list(DATASETS.keys())
    
    if ltl_percentages is None:
        ltl_percentages = ['10%', '25%', '50%']
    
    for dataset_name in datasets:
        dataset_def = DATASETS[dataset_name]
        
        for prefix_length in dataset_def.prefix_lengths:
            logger.info("=" * 100)
            logger.info(f"DATASET: {dataset_name}, PREFIX: {prefix_length}")
            logger.info("=" * 100)
            
            try:
                # Train model once per prefix length
                logger.info("TRAINING MODEL")
                model_pipeline, metrics = train_model(
                    dataset_name=dataset_name,
                    prefix_length=prefix_length,
                    seed=seed,
                    save_model=True
                )
                
                logger.info(f"Model metrics: {metrics}")
                
                # Generate CFs for all LTL percentages
                for ltl_pct in ltl_percentages:
                    logger.info(f"GENERATING CFs FOR LTL {ltl_pct}")
                    
                    generate_cfs(
                        dataset_name=dataset_name,
                        prefix_length=prefix_length,
                        ltl_percentage=ltl_pct,
                        model_file=f"models/{dataset_name}_prefix{prefix_length}_model.pkl",
                        seed=seed
                    )
                    
            except Exception as e:
                logger.error(
                    f"Error in {dataset_name}, prefix {prefix_length}: {str(e)}",
                    exc_info=True
                )
                continue
    
    logger.info("All experiments complete!")

def main():
    parser = argparse.ArgumentParser(description='Run full LTLf CF pipeline')
    parser.add_argument('--datasets', nargs='+', default=None,
                       choices=list(DATASETS.keys()),
                       help='Datasets to run (default: all)')
    parser.add_argument('--ltl-percentages', nargs='+', default=['50%'],
                       choices=['10%', '25%', '50%'],
                       help='LTL model percentages to run')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    run_all_experiments(
        datasets=args.datasets,
        ltl_percentages=args.ltl_percentages,
        seed=args.seed
    )

if __name__ == '__main__':
    main()