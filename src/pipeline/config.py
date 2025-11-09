"""Configuration management for LTLf CF pipeline."""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from pathlib import Path
import json
from src.labeling.common import LabelTypes

@dataclass
class DatasetDefinition:
    """Definition of a dataset with its experiments."""
    name: str
    prefix_lengths: List[int]
    ltl_models: Dict[str, str]
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

# Dataset definitions
DATASETS = {
    'synthetic_data': DatasetDefinition(
        name='synthetic_data',
        prefix_lengths=[7, 9, 11, 13],
        ltl_models={'10%':'G(contacthospital -> X(acceptclaim | rejectclaim))',
                       '25%':'G(contacthospital -> X(acceptclaim | rejectclaim)) & F(createquestionnaire)',
                        '50%':'(F(contacthospital) -> F(highinsurancecheck)) & G(preparenotificationcontent -> X(sendnotificationbyphone \
                        | sendnotificationbypost)) & G(createquestionnaire-> F(preparenotificationcontent)) & register'
                       }
    ),
    'bpic2012_O_ACCEPTED-COMPLETE': DatasetDefinition(
        name='bpic2012_O_ACCEPTED-COMPLETE',
        prefix_lengths=[20, 25, 30, 35],
        ltl_models={
            '10%':'F(osentcomplete) & G(osentcomplete ->(!(aacceptedcomplete)U(wcompleterenaanvraagcomplete))) & F(osentbackcomplete)',
            '25%':' F(osentcomplete) & G(osentcomplete ->(!(aacceptedcomplete)U(wcompleterenaanvraagcomplete))) &F(osentbackcomplete) &  G(wcompleterenaanvraagstart -> F(aacceptedcomplete)) & (F(wnabellenoffertesstart) & F(wnabellenoffertescomplete)) &  (F(oselectedcomplete) |F(wvaliderenaanvraagstart))',
             '50%': ('F(osentcomplete) & G(osentcomplete -> (!(aacceptedcomplete)U(wcompleterenaanvraagcomplete))) & '
            'G(wcompleterenaanvraagschedule -> F(wcompleterenaanvraagstart)) & '
            '(F(wnabellenoffertesstart) | F(wnabellenoffertescomplete)) & '
            '(F(oselectedcomplete) | F(wvaliderenaanvraagstart)) & '
            'asubmittedcomplete & F(oselectedcomplete | apartlysubmittedcomplete) & '
            'G(ocreatedcomplete -> F(osentbackcomplete)) & '
            '(F(afinalizedcomplete) | F(apreacceptedcomplete) | F(wafhandelenleadscomplete))')
                               }
    ),
    'BPIC17_O_ACCEPTED': DatasetDefinition(
        name='BPIC17_O_ACCEPTED',
        prefix_lengths=[15, 20, 25, 30],
        ltl_models={'10%':'acreateapplication & (!(aconcept)U(wcompleteapplication))',
                         '25%':'acreateapplication & (!(aconcept)U(wcompleteapplication)) & (F(ocreateoffer) -> F(wcallafteroffers)) & F(wcompleteapplication)',
                         '50%':'acreateapplication & (!(aconcept)U(wcompleteapplication)) & (G(ocreateoffer) -> (F(wcallafteroffers) | F(wvalidateapplication))) & (F(ocreated) -> X(osentmailandonline | osentonlineonly)) & G((aincomplete | apending) -> (X(wcallincompletefiles) & F(wvalidateapplication)))' }
    ),
    'sampled_logs': DatasetDefinition(
        name='sampled_logs',
        prefix_lengths=[15, 20, 25, 30],
        ltl_models={'10%':'acreateapplication & (!(aconcept)U(wcompleteapplication))',
                         '25%':'acreateapplication & (!(aconcept)U(wcompleteapplication)) & (F(ocreateoffer) -> F(wcallafteroffers)) & F(wcompleteapplication)',
                         '50%':'acreateapplication & (!(aconcept)U(wcompleteapplication)) & (G(ocreateoffer) -> (F(wcallafteroffers) | F(wvalidateapplication))) & (F(ocreated) -> X(osentmailandonline | osentonlineonly)) & G((aincomplete | apending) -> (X(wcallincompletefiles) & F(wvalidateapplication)))' }
    )
}

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    # Dataset info
    dataset_name: str
    prefix_length: int
    ltl_percentage: str = '50%'
    
    # Data paths
    data_dir: str = '.'
    model_dir: str = '../experiments/process_models/process_models_new'
    output_dir: str = 'results'
    
    # Data split
    train_val_test_split: Tuple[float, float, float] = (0.7, 0.1, 0.2)
    
    # Encoding settings
    prefix_length_strategy: str = 'fixed'
    padding: bool = True
    feature_selection: str = 'simple'
    task_generation_type: str = 'only_this'
    attribute_encoding: str = 'label'
    time_encoding: str = 'none'
    
    # Labeling
    labeling_type: str = LabelTypes.ATTRIBUTE_STRING.value
    target_event: str = None
    
    # Model settings
    predictive_model: str = 'xgboost'
    hyperparameter_optimisation_target: str = 'auc'
    hyperparameter_optimisation_epochs: int = 20
    
    # CF settings
    explanator: str = 'dice_ltlf'
    threshold: int = 13
    top_k: int = 10
    
    # Random seed
    seed: int = 42
    
    def get_data_path(self) -> str:
        """Get full path to dataset."""
        return str(Path(self.data_dir) / self.dataset_name / 'full.xes')
    
    def get_ltl_model(self) -> str:
        """Get LTL model for current configuration."""
        dataset_def = DATASETS[self.dataset_name]
        return dataset_def.ltl_models[self.ltl_percentage]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for legacy compatibility."""
        return {
            'data': self.get_data_path(),
            'prefix_length': self.prefix_length,
            'train_val_test_split': list(self.train_val_test_split),
            'prefix_length_strategy': self.prefix_length_strategy,
            'padding': self.padding,
            'feature_selection': self.feature_selection,
            'task_generation_type': self.task_generation_type,
            'attribute_encoding': self.attribute_encoding,
            'labeling_type': self.labeling_type,
            'predictive_model': self.predictive_model,
            'explanator': self.explanator,
            'threshold': self.threshold,
            'top_k': self.top_k,
            'hyperparameter_optimisation_target': self.hyperparameter_optimisation_target,
            'hyperparameter_optimisation_epochs': self.hyperparameter_optimisation_epochs,
            'time_encoding': self.time_encoding,
            'target_event': self.target_event,
            'seed': self.seed,
            'output': self.output_dir
        }
    
    def get_experiment_id(self) -> str:
        """Get unique identifier for this experiment."""
        return f"{self.dataset_name}_prefix{self.prefix_length}"

@dataclass 
class CFMethodConfig:
    """Configuration for counterfactual generation methods."""
    method: str = 'genetic_ltlf'
    optimization: str = 'baseline'
    heuristic: str = 'heuristic_1'
    adapted: bool = False
    
    def __str__(self):
        return f"{self.method}_{self.optimization}_{self.heuristic}_adapted{self.adapted}"

def get_cf_configurations() -> List[CFMethodConfig]:
    """Get all CF method configurations to run."""
    configs = [
        # Baseline non-adapted
        CFMethodConfig(heuristic='heuristic_1', adapted=False),
    ]
    
    # Adapted versions with all heuristics
    for heuristic in ['heuristic_1', 'heuristic_2', 'mar']:
        configs.append(CFMethodConfig(heuristic=heuristic, adapted=True))
    
    return configs