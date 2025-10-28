"""Data loading and preprocessing pipeline."""
import logging
import pandas as pd
import numpy as np
from typing import Tuple
from pathlib import Path

from src.encoding.common import get_encoded_df
from src.log.common import get_log
from pm4py import convert_to_event_log, format_dataframe
from Declare4Py.D4PyEventLog import D4PyEventLog

logger = logging.getLogger(__name__)

class DataPipeline:
    """Handles all data loading and preprocessing."""
    
    def __init__(self, config: dict):
        self.config = config
        self.encoder = None
        self.full_df = None
        
    def load_and_encode(self):
        """Load event log and encode to dataframe."""
        logger.info(f"Loading data from: {self.config['data']}")
        
        # Load log
        log = get_log(filepath=self.config['data'])
        
        # Normalize event names for specific datasets
        dataset_name = Path(self.config['data']).parent.name
        if 'bpic2012' in dataset_name or 'synthetic_data' in dataset_name:
            log = self._normalize_event_names(log)
        
        # Encode
        logger.info(f"Encoding with prefix length: {self.config['prefix_length']}")
        self.encoder, self.full_df = get_encoded_df(log=log, CONF=self.config)
        
        logger.info(f"Encoded dataframe shape: {self.full_df.shape}")
        return self.encoder, self.full_df
    
    @staticmethod
    def _normalize_event_names(log):
        """Normalize event names by removing special characters."""
        for trace in log:
            for event in trace:
                event['concept:name'] = (
                    event['concept:name']
                    .replace('_', '').replace('-', '').replace(' ', '')
                    .replace('(', '').replace(')', '').lower()
                )
        return log
    
    def split_data(
        self,
        df: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets."""
        if df is None:
            df = self.full_df
            
        train_size, val_size, test_size = self.config['train_val_test_split']
        
        if not np.isclose(train_size + val_size + test_size, 1.0):
            raise ValueError(
                f"Split ratios must sum to 1.0, got {train_size + val_size + test_size}"
            )
        
        # Sequential split (important for temporal data)
        split_idx_1 = int(train_size * len(df))
        split_idx_2 = int((train_size + val_size) * len(df))
        
        train_df, val_df, test_df = np.split(df, [split_idx_1, split_idx_2])
        
        logger.info(
            f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )
        
        return train_df, val_df, test_df
    
    def create_event_log(self, df: pd.DataFrame = None) -> D4PyEventLog:
        """Convert encoded dataframe back to event log format."""
        if df is None:
            df = self.full_df.copy()
        else:
            df = df.copy()
            
        logger.info("Creating D4Py event log from encoded dataframe")
        logger.debug(f"Input df shape: {df.shape}")
        logger.debug(f"Input df sample before decode:\n{df.iloc[0]}")
        
        # Decode
        decoded_df = df.copy()
        self.encoder.decode(decoded_df)
        
        # Convert to long format
        long_data = pd.wide_to_long(
            decoded_df.drop_duplicates(subset=['trace_id']),
            stubnames=['prefix'],
            i='trace_id',
            j='order',
            sep='_',
            suffix=r'\w+'
        )
        
        # Add timestamps
        timestamps = pd.date_range('1/1/2011', periods=len(long_data), freq='H')
        long_data_sorted = long_data.sort_values(['trace_id', 'order']).reset_index(drop=False)
        long_data_sorted['time:timestamp'] = timestamps
        long_data_sorted.drop(columns=['order'], inplace=True)
        
        # Rename and clean
        long_data_sorted.rename(
            columns={'trace_id': 'case:concept:name', 'prefix': 'concept:name'},
            inplace=True
        )
        long_data_sorted['label'].replace({'regular': 'false', 'deviant': 'true'}, inplace=True)
        long_data_sorted.replace('0', 'other', inplace=True)
        
        # Convert to PM4Py event log
        dataframe = format_dataframe(
            long_data_sorted,
            case_id='case:concept:name',
            activity_key='concept:name',
            timestamp_key='time:timestamp'
        )
        log = convert_to_event_log(dataframe)
        
        # Create D4Py event log
        event_log = D4PyEventLog(case_name="case:concept:name")
        event_log.load_xes_log(log)
        
        logger.info(f"Created event log with {len(event_log.log)} traces")
        
        return event_log