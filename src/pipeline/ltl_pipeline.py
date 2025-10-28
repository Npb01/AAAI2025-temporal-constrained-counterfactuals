"""LTL model processing and conformance checking."""
import logging
import pandas as pd
import re
from typing import Tuple, List

from Declare4Py.ProcessModels.LTLModel import LTLModel
from Declare4Py.ProcessMiningTasks.ConformanceChecking.LTLAnalyzer import LTLAnalyzer
from Declare4Py.Utils.utils import Utils
from logaut import ltl2dfa

logger = logging.getLogger(__name__)

class LTLPipeline:
    """Handles LTL model operations."""
    
    def __init__(self):
        self.ltl_model = LTLModel(backend='ltlf2dfa')
        self.dfa = None
        self.model_names: List[str] = []
        self.model_string: str = ""
        
    def parse_model(self, model_string: str):
        """Parse LTL formula and create DFA."""
        logger.info("Parsing LTL model")
        logger.debug(f"Formula: {model_string}")
        
        self.model_string = model_string
        
        # Parse
        self.ltl_model.parse_from_string(model_string)
        
        # Convert to DFA
        self.dfa = ltl2dfa(self.ltl_model.parsed_formula, backend='ltlf2dfa')
        
        # Extract activity names
        normalized = Utils.normalize_formula(model_string)
        self.model_names = self._extract_activity_names(normalized)
        
        logger.info(f"Extracted {len(self.model_names)} activities: {self.model_names}")
        
        return self.ltl_model, self.dfa, self.model_names
    
    @staticmethod
    def _extract_activity_names(formula: str) -> List[str]:
        """Extract activity names from LTL formula."""
        # Use regular expression to find text within parentheses
        pattern = r'\((.*?)\)'
        matches = re.findall(pattern, formula)

        # Remove spaces and letters from each match
        cleaned_names = [re.sub(r'[^a-zA-Z]', '', match).lower().strip('f').strip('g').strip('u') for match in matches]

        return cleaned_names

    def check_conformance(
        self,
        event_log,
        jobs: int = 12,
        minimize: bool = False
    ) -> pd.DataFrame:
        """Run conformance checking on event log."""
        if self.ltl_model is None or self.dfa is None:
            raise ValueError("Must parse model before checking conformance")
        
        logger.info("Running conformance checking")
        
        analyzer = LTLAnalyzer(event_log, self.ltl_model)
        conf_results = analyzer.run(
            jobs=jobs,
            minimize_automaton=minimize,
            dfa=self.dfa
        )
        
        # Log statistics
        total = len(conf_results)
        conforming = conf_results[conf_results['accepted'] == True].shape[0]
        conf_rate = 100 * conforming / total if total > 0 else 0
        
        logger.info(f"Conformance: {conforming}/{total} ({conf_rate:.1f}%) traces conform")
        
        return conf_results