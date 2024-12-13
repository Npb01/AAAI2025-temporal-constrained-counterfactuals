from src.explanation.wrappers.dice_ltlf import dice_ltlf_explain
from enum import Enum

class ExplainerType(Enum):
    DICE_LTLf = 'dice_ltlf'
    DICE_GENETIC = 'dice_genetic'


def explain(CONF, predictive_model, encoder, df=None, query_instances=None, method=None,
            optimization=None, heuristic=None, support=0.9, timestamp_col_name=None,
            model_path=None, case_ids=None, random_seed=None, adapted=None, filtering=False,
            ltlf_model=None, model_names=None, path_results=None, dfa=None, percentage=None):
    """
    Explain the predictions using the specified explainer method.

    Parameters:
    - CONF: Configuration dictionary containing the necessary settings.
    - predictive_model: The machine learning model used for predictions.
    - encoder: Encoder for transforming data as required by the model.
    - df: DataFrame containing the data for explanations.
    - query_instances: Instances for which explanations are needed.
    - method: Explanation method to be used (e.g., 'method1', 'method2').
    - optimization: Optimization strategy for the explanation method.
    - heuristic: Heuristic strategy to be applied.
    - timestamp_col_name: Name of the timestamp column in the data.
    - model_path: Path to the saved model, if applicable.
    - case_ids: IDs of the cases to be explained.
    - random_seed: Seed for random number generation.
    - adapted: Boolean indicating if the adapted version of the explanation method should be used.
    - filtering: Boolean indicating if filtering should be applied.
    - ltlf_model: LTLf model for compliance checking.
    - model_names: List of model names for the explanation.
    - path_results: Path where the results should be saved.
    - dfa: Deterministic Finite Automaton for LTLf formula compliance.
    - percentage: Percentage value used in explanation methods.

    Returns:
    - Explanation result based on the specified parameters and version.
    """
    return dice_ltlf_explain(CONF, predictive_model, encoder=encoder, df=df, query_instances=query_instances,
                                 method=method, optimization=optimization, heuristic=heuristic,
                                 model_path=model_path, case_ids=case_ids,
                                 random_seed=random_seed, adapted=adapted, ltlf_model=ltlf_model,
                                 path_results=path_results, dfa=dfa, percentage=percentage)

