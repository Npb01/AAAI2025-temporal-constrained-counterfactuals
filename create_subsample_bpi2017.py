import pandas as pd
import pm4py
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Configuration of the sampling process
# File paths

LOG_FILE = "BPIC17_O_ACCEPTED\\full.xes"     # The full, original XES log file
OUTPUT_DIR = "sampled_logs"   # Directory to save the new CSVs

# Column Names
# These are standard for pm4py
CASE_ID_COL = 'case:concept:name'
ACTIVITY_COL = 'concept:name'
TIMESTAMP_COL = 'time:timestamp'

# Split & Sample Sizes
# Ratios for the chronological split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
# TEST_RATIO is the remainder (1.0 - 0.70 - 0.10 = 0.20)

# Target number of CASES for your *final* sampled files
N_SAMPLE_CASES = 5000
N_TRAIN_CASES = int(N_SAMPLE_CASES*TRAIN_RATIO)
N_VAL_CASES = int(N_SAMPLE_CASES*VAL_RATIO)
N_TEST_CASES = N_SAMPLE_CASES - N_TRAIN_CASES - N_VAL_CASES

# Dataset-Specific Logic
# This function defines what a case 'outcome' is.
# 'concept:name' of the very last event in the trace.
def get_case_outcomes(df_full):
    print("  Determining case outcomes (BPI_2017 logic)...")
    # Sort by case and time
    df_sorted = df_full.sort_values(by=[CASE_ID_COL, TIMESTAMP_COL])
    
    # Group by case, get the last activity
    df_case_outcomes = df_sorted.groupby(CASE_ID_COL) \
                                .agg(outcome=(ACTIVITY_COL, 'last')) \
                                .reset_index()
    return df_case_outcomes


def perform_stratified_sampling(df_events_full, df_cases_full, n_sample_cases):
    """
    Helper function to perform stratified sampling.

    :param df_events_full: The full DataFrame of events for this split (e.g., all train events)
    :param df_cases_full: The DataFrame of cases for this split (ID, start_time, outcome)
    :param n_sample_cases: The target number of cases for the sample
    :return: A DataFrame of events for the sampled cases
    """

    # Get the stratification column
    strata_col = 'outcome'
    
    # Check if sample size is larger than available cases
    if n_sample_cases >= len(df_cases_full):
        print(f"    Warning: Target sample size ({n_sample_cases}) is >= available cases ({len(df_cases_full)}). Using all available cases.")
        sampled_case_ids = df_cases_full[CASE_ID_COL]
    else:
        # Use train_test_split to get a stratified sample
        # We only care about the 'train' part of the split, which is our sample
        try:
            sample_cases_df, _ = train_test_split(
                df_cases_full,
                train_size=n_sample_cases,
                stratify=df_cases_full[strata_col], # This is the magic!
                random_state=42
            )
            sampled_case_ids = sample_cases_df[CASE_ID_COL]
        except ValueError as e:
            print(f"    Warning: Stratified sampling failed (e.g., some outcomes have < 2 members). {e}")
            print("    Falling back to simple random sampling for this split.")
            sampled_case_ids = df_cases_full[CASE_ID_COL].sample(n=n_sample_cases, random_state=42)

    # Filter the *event* log to get the final sample
    df_event_sample = df_events_full[df_events_full[CASE_ID_COL].isin(sampled_case_ids)]
    return df_event_sample


def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 2. Load and Convert Full Log ---
    print(f"Loading log: {LOG_FILE}...")
    if not os.path.exists(LOG_FILE):
        print(f"--- ERROR ---")
        print(f"Log file '{LOG_FILE}' not found.")
        print("Please download the log and place it in the same directory as this script.")
        print(f"--- --- ---")
        return
    
    log = pm4py.read_xes(LOG_FILE)
    df_full = pm4py.convert_to_dataframe(log)
    print(f"Log loaded. Total events: {len(df_full)}")

    # Pre-process: Get Case-Level Information
    print("Processing all cases to get start times and outcomes...")
    
    # Ensure correct datetime type for sorting
    df_full[TIMESTAMP_COL] = pd.to_datetime(df_full[TIMESTAMP_COL], utc=True)

    # Get start time (min timestamp) for each case
    df_case_starts = df_full.groupby(CASE_ID_COL) \
                            .agg(case_start_time=(TIMESTAMP_COL, 'min')) \
                            .reset_index()
    
    # Get dataset-specific outcomes
    df_case_outcomes = get_case_outcomes(df_full)

    # Combine into a single case-level DataFrame
    df_case_master = pd.merge(df_case_starts, df_case_outcomes, on=CASE_ID_COL)
    
    print(f"\nFound {len(df_case_master)} unique cases.")
    print("Full log outcome distribution:")
    print(df_case_master['outcome'].value_counts(normalize=True).head())


    # --- 4. Chronological Split (on Cases) ---
    print("\nPerforming chronological split...")
    
    # Sort by start time *first*
    df_case_master = df_case_master.sort_values(by='case_start_time')

    # Get the case IDs as a list
    all_case_ids = df_case_master[CASE_ID_COL].values
    
    # Calculate split indices
    n_total = len(all_case_ids)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    # Split the list of IDs
    train_case_ids, val_case_ids, test_case_ids = np.split(
        all_case_ids,
        [n_train, n_train + n_val]
    )

    print(f"Split complete:")
    print(f"  Full Train cases: {len(train_case_ids)}")
    print(f"  Full Val cases:   {len(val_case_ids)}")
    print(f"  Full Test cases:  {len(test_case_ids)}")

    # --- 5. Filter Full Log into Temporal Splits ---
    # This is memory-intensive, but it's the only
    # way to correctly sample from the temporal splits.
    
    print("\nFiltering full event log into train/val/test splits...")
    
    # Get the full event logs for each split
    df_train_full = df_full[df_full[CASE_ID_COL].isin(train_case_ids)]
    df_val_full = df_full[df_full[CASE_ID_COL].isin(val_case_ids)]
    df_test_full = df_full[df_full[CASE_ID_COL].isin(test_case_ids)]
    
    # Get the case-level info for each split
    outcomes_train = df_case_master[df_case_master[CASE_ID_COL].isin(train_case_ids)]
    outcomes_val = df_case_master[df_case_master[CASE_ID_COL].isin(val_case_ids)]
    outcomes_test = df_case_master[df_case_master[CASE_ID_COL].isin(test_case_ids)]


    # --- 6. Stratified Sampling on Each Split ---
    print("\nPerforming stratified sampling on each split...")

    # --- Sample Train ---
    print(f"  Sampling train set (target: {N_TRAIN_CASES} cases)...")
    df_train_sample = perform_stratified_sampling(
        df_train_full, outcomes_train, N_TRAIN_CASES
    )

    # --- Sample Val ---
    print(f"  Sampling validation set (target: {N_VAL_CASES} cases)...")
    df_val_sample = perform_stratified_sampling(
        df_val_full, outcomes_val, N_VAL_CASES
    )

    # --- Sample Test ---
    print(f"  Sampling test set (target: {N_TEST_CASES} cases)...")
    df_test_sample = perform_stratified_sampling(
        df_test_full, outcomes_test, N_TEST_CASES
    )
    
    # --- 7. Save Sampled Logs ---
        # Option 1: Save each split separately as CSV
    print("\nSaving sampled logs as CSV files...")
    
    train_path = os.path.join(OUTPUT_DIR, 'train_sample.csv')
    val_path = os.path.join(OUTPUT_DIR, 'val_sample.csv')
    test_path = os.path.join(OUTPUT_DIR, 'test_sample.csv')

    df_train_sample.to_csv(train_path, index=False)
    df_val_sample.to_csv(val_path, index=False)
    df_test_sample.to_csv(test_path, index=False)

        # Option 2: Merge all splits and save as XES
    print("\nMerging all splits into one dataframe...")

    # Add a split label column to each split (optional, for tracking)
    # df_train_sample['split'] = 'train'
    # df_val_sample['split'] = 'val'
    # df_test_sample['split'] = 'test'

    # Concatenate all splits
    df_merged = pd.concat([df_train_sample, df_val_sample, df_test_sample], ignore_index=True)

    print(f"Merged dataframe: {len(df_merged)} events, {df_merged[CASE_ID_COL].nunique()} cases")

    # Convert back to event log format and save as XES
    print("Converting to XES format...")
    event_log = pm4py.convert_to_event_log(df_merged)

    xes_path = os.path.join(OUTPUT_DIR, 'full.xes')
    pm4py.write_xes(event_log, xes_path)

    print(f"\n--- SUCCESS ---")
    print(f"Sampled logs saved in '{OUTPUT_DIR}':")
    print(f"  Train: {train_path} ({len(df_train_sample)} events, {df_train_sample[CASE_ID_COL].nunique()} cases)")
    print(f"  Val:   {val_path} ({len(df_val_sample)} events, {df_val_sample[CASE_ID_COL].nunique()} cases)")
    print(f"  Test:  {test_path} ({len(df_test_sample)} events, {df_test_sample[CASE_ID_COL].nunique()} cases)")
    print(f"  Merged XES: {xes_path} ({len(df_merged)} events)")


if __name__ == "__main__":
    main()