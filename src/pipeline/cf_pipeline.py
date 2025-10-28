"""Counterfactual generation pipeline."""
import logging
import pandas as pd
from pathlib import Path

from src.explanation.common import explain
from src.pipeline.config import CFMethodConfig

logger = logging.getLogger(__name__)

class CFPipeline:
    """Handles counterfactual generation."""
    
    def __init__(
        self,
        config: dict,
        predictive_model,
        encoder,
        ltl_model,
        dfa,
        model_names: list,
        experiment_id: str
    ):
        self.config = config
        self.predictive_model = predictive_model
        self.encoder = encoder
        self.ltl_model = ltl_model
        self.dfa = dfa
        self.model_names = model_names
        self.experiment_id = experiment_id
        
    def prepare_query_instances(
        self,
        test_df: pd.DataFrame,
        actual: pd.Series,
        predicted: pd.Series
    ) -> pd.DataFrame:
        """Get test instances for CF generation (correctly predicted negatives)."""
        # Filter: predicted correctly AND labeled as negative (0)
        mask = (test_df['label'] == predicted) & (test_df['label'] == 0)
        query_instances = test_df[mask]
        
        logger.info(f"Total test instances: {len(test_df)}")
        logger.info(f"Correctly predicted negatives: {len(query_instances)}")
        
        if len(query_instances) == 0:
            logger.warning("No valid query instances found!")
        
        return query_instances
    
    def generate_counterfactuals(
        self,
        query_instances: pd.DataFrame,
        full_df: pd.DataFrame,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        cf_config: CFMethodConfig,
        ltl_percentage: str
    ):
        """Generate counterfactuals for given configuration."""
        logger.info("=" * 80)
        logger.info(f"Generating CFs: {cf_config}")
        logger.info("=" * 80)
        
        # Prepare CF dataset (train + val)
        cf_dataset = pd.concat([train_df, val_df], ignore_index=True)
        cf_dataset.loc[len(cf_dataset)] = 0  # Add dummy row (legacy requirement?)
        
        # Encode full_df for CF generation
        full_df_encoded = full_df.copy()
        
        # Generate CFs
        explain(
            self.config,
            self.predictive_model,
            encoder=self.encoder,
            query_instances=query_instances.iloc[:, 1:],  # Drop first column
            method=cf_config.method,
            df=full_df_encoded.iloc[:, 1:],  # Drop first column
            optimization=cf_config.optimization,
            heuristic=cf_config.heuristic,
            model_path=self.config.get('model_dir', '../experiments/process_models/process_models_new'),
            random_seed=self.config['seed'],
            adapted=cf_config.adapted,
            ltlf_model=self.ltl_model,
            model_names=self.model_names,
            percentage=ltl_percentage,
            path_results=self.config.get('output', 'results'),
            dfa=self.dfa
        )
        
        logger.info(f"CF generation complete: {cf_config}")