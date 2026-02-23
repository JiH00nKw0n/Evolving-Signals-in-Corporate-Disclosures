"""
Moving Targets Score Calculation Module

This module calculates Moving Targets (MT) scores based on the methodology from
Cohen & Nguyen (2024). It processes earnings call transcripts to extract targets
and calculates the conditional Moving Targets score:

MT_t = Σ(Missing Targets_t | Targets_{t-4}) / Σ Targets_{t-4}

The score measures how many targets mentioned 4 quarters ago are no longer
mentioned in the current quarter, as a proportion of all targets from 4 quarters ago.
"""
import asyncio
import os
import re
import warnings
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

warnings.filterwarnings('ignore')
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from multiprocessing import cpu_count
from src.embedding import get_embedding_with_cache, get_similarity


def get_analyze_function(extraction_method: str = "base"):
    """
    Get the appropriate analyze_earnings_call function based on extraction method

    Args:
        extraction_method: "base" for spaCy-based extraction, "llm" for LLM-based extraction

    Returns:
        analyze_earnings_call function
    """
    if extraction_method == "llm":
        from src.llm_target import analyze_earnings_call
        return analyze_earnings_call
    else:  # default to base
        from src.base_target import analyze_earnings_call
        return analyze_earnings_call


def extract_isin_date_from_filename(filename: str) -> Optional[Tuple[str, str, str]]:
    """
    Extract ISIN, date, and quarter from earnings call HTML filename.

    Expected format: {ISIN}={Quarter}_{Year}_{Company_Name}={Date}.html
    Example: BRPSSAACNOR7=Q1_2012_Marathon_Petroleum_Corporation=2012-05-01.html

    Args:
        filename: HTML filename

    Returns:
        Tuple of (isin, date_str, quarter) or None if parsing fails
    """
    try:
        # Remove .html extension
        basename = filename.replace('.html', '')

        # Split by '=' to get parts
        parts = basename.split('=')
        if len(parts) < 3:
            return None

        isin = parts[0]
        quarter_company = parts[1]  # e.g., Q1_2012_Marathon_Petroleum_Corporation
        date_str = parts[2]  # e.g., 2012-05-01

        # Extract quarter from quarter_company part
        quarter_match = re.match(r'(Q[1-4]_\d{4})', quarter_company)
        if quarter_match:
            quarter = quarter_match.group(1)
        else:
            quarter = "Unknown"

        return isin, date_str, quarter

    except Exception:
        return None


def scan_earnings_files(earnings_dir: str,
                        isin_list: List[str],
                        date_range: Tuple[str, str]) -> List[Dict]:
    """
    Scan earnings call HTML files and filter by ISIN and date range.

    Args:
        earnings_dir: Directory containing HTML files
        isin_list: List of ISIN codes to filter
        date_range: Tuple of (start_date, end_date) strings

    Returns:
        List of file info dictionaries
    """
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])

    file_info_list = []

    def first_eq_followed_by_Q(path: str) -> bool:
        i = path.find("=")
        return i != -1 and i + 1 < len(path) and path[i + 1] == "Q"

    # Get all HTML files
    html_files = [f for f in os.listdir(earnings_dir)
                  if f.endswith('.html') and first_eq_followed_by_Q(os.path.join(earnings_dir, f))]

    for filename in html_files:
        # Extract ISIN, date, quarter from filename
        parsed = extract_isin_date_from_filename(filename)
        if parsed is None:
            continue

        isin, date_str, quarter = parsed

        # Filter by ISIN
        if isin not in isin_list:
            continue

        # Parse and filter by date
        try:
            file_date = pd.to_datetime(date_str)
            if not (start_date <= file_date <= end_date):
                continue
        except Exception:
            continue

        file_info = {
            'filename': filename,
            'filepath': os.path.join(earnings_dir, filename),
            'isin': isin,
            'date': file_date,
            'date_str': date_str,
            'quarter': quarter
        }
        file_info_list.append(file_info)

    return file_info_list


def process_single_earnings_file_sync(file_info: Dict, extraction_method: str = "base") -> Optional[Dict]:
    """
    Synchronous version for base extraction method.
    """
    try:
        # Read HTML content
        with open(file_info['filepath'], 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Get the appropriate analyze function
        analyze_earnings_call = get_analyze_function(extraction_method)

        # Analyze with selected method (base is synchronous)
        analysis_result = analyze_earnings_call(html_content)

        # Extract target counts
        num_total_targets = analysis_result['total_targets']
        num_presentation_targets = analysis_result['presentation_targets']
        num_qa_targets = analysis_result['qa_targets']
        num_total_chunks = analysis_result['total_chunks']

        # Extract target sets
        targets_by_section = analysis_result['targets_by_section']
        targets_all_set = set(targets_by_section['all'])
        targets_presentation_set = set(targets_by_section['presentation'])
        targets_qa_set = set(targets_by_section['qa'])

        # Create result row
        result_row = {
            'date': file_info['date'],
            'isin': file_info['isin'],
            'quarter': file_info['quarter'],
            'filename': file_info['filename'],

            # Target counts (clear naming)
            'num_total_targets': num_total_targets,
            'num_presentation_targets': num_presentation_targets,
            'num_qa_targets': num_qa_targets,
            'num_total_chunks': num_total_chunks,

            # Target sets for detailed analysis
            'targets_all_set': targets_all_set,
            'targets_presentation_set': targets_presentation_set,
            'targets_qa_set': targets_qa_set,

        }

        return result_row

    except Exception as e:
        print(f"Warning: Error processing {file_info['filename']}: {e}")
        return None


async def process_single_earnings_file(file_info: Dict, extraction_method: str = "base",
                                       semaphore: Optional[asyncio.Semaphore] = None) -> Optional[Dict]:
    """
    Process a single earnings file and return target analysis results.

    Args:
        file_info: Dictionary with file information (filepath, filename, isin, date, quarter)
        extraction_method: "base" for spaCy-based extraction, "llm" for LLM-based extraction
        semaphore: Semaphore to limit concurrent processing

    Returns:
        Dictionary with analysis results or None if processing fails
    """
    async with semaphore:
        try:
            # Read HTML content
            with open(file_info['filepath'], 'r', encoding='utf-8') as f:
                html_content = f.read()

            # Get the appropriate analyze function
            analyze_earnings_call = get_analyze_function(extraction_method)

            # Analyze with selected method
            if extraction_method == "llm":
                analysis_result = await analyze_earnings_call(html_content)
            else:
                analysis_result = analyze_earnings_call(html_content)

            # Extract target counts
            num_total_targets = analysis_result['total_targets']
            num_presentation_targets = analysis_result['presentation_targets']
            num_qa_targets = analysis_result['qa_targets']
            num_total_chunks = analysis_result['total_chunks']

            # Extract target sets
            targets_by_section = analysis_result['targets_by_section']
            targets_all_set = set(targets_by_section['all'])
            targets_presentation_set = set(targets_by_section['presentation'])
            targets_qa_set = set(targets_by_section['qa'])

            # Create result row
            result_row = {
                'date': file_info['date'],
                'isin': file_info['isin'],
                'quarter': file_info['quarter'],
                'filename': file_info['filename'],

                # Target counts (clear naming)
                'num_total_targets': num_total_targets,
                'num_presentation_targets': num_presentation_targets,
                'num_qa_targets': num_qa_targets,
                'num_total_chunks': num_total_chunks,

                # Target sets for detailed analysis
                'targets_all_set': targets_all_set,
                'targets_presentation_set': targets_presentation_set,
                'targets_qa_set': targets_qa_set,
            }

            return result_row

        except Exception as e:
            print(f"Warning: Error processing {file_info['filename']}: {e}")
            return None


def calculate_raw_mt_data_base(isin_list: List[str],
                               date_range: Tuple[str, str],
                               earnings_dir: str = "data/earnings",
                               max_workers: Optional[int] = 8) -> pd.DataFrame:
    """
    Calculate raw MT data (target counts and sets) using base spaCy extraction method.

    Args:
        isin_list: List of ISIN codes
        date_range: Tuple of (start_date, end_date) strings
        earnings_dir: Directory containing HTML earnings files
        max_workers: Maximum number of parallel workers (default: CPU count or file count, whichever is smaller)

    Returns:
        DataFrame with raw target data and sets
    """

    print(
        f"Calculating raw MT data for {len(isin_list)} ISINs from {date_range[0]} to {date_range[1]} using base method"
    )

    # Scan for relevant files
    file_info_list = scan_earnings_files(earnings_dir, isin_list, date_range)
    print(f"Found {len(file_info_list)} earnings call files to process")

    if len(file_info_list) == 0:
        print("Warning: No files found matching criteria")
        return pd.DataFrame()

    # Determine number of workers
    if max_workers is None:
        max_workers = min(cpu_count(), len(file_info_list))

    print(f"Processing files with {max_workers} parallel workers...")

    results = []

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_earnings_file_sync, file_info, "base"): file_info
            for file_info in file_info_list
        }

        # Process completed tasks
        completed = 0
        for future in as_completed(future_to_file):
            file_info = future_to_file[future]
            completed += 1

            print(f"Processing {completed}/{len(file_info_list)}: {file_info['filename']}")

            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                else:
                    # Add a row with NaN/empty values for failed processing
                    result_row = {
                        'date': file_info['date'],
                        'isin': file_info['isin'],
                        'quarter': file_info['quarter'],
                        'filename': file_info['filename'],
                        'num_total_targets': np.nan,
                        'num_presentation_targets': np.nan,
                        'num_qa_targets': np.nan,
                        'num_total_chunks': np.nan,
                        'targets_all_set': set(),
                        'targets_presentation_set': set(),
                        'targets_qa_set': set(),
                    }
                    results.append(result_row)
            except Exception as e:
                print(f"Error processing {file_info['filename']}: {e}")
                # Add a row with NaN/empty values for failed processing
                result_row = {
                    'date': file_info['date'],
                    'isin': file_info['isin'],
                    'quarter': file_info['quarter'],
                    'filename': file_info['filename'],
                    'num_total_targets': np.nan,
                    'num_presentation_targets': np.nan,
                    'num_qa_targets': np.nan,
                    'num_total_chunks': np.nan,
                    'targets_all_set': set(),
                    'targets_presentation_set': set(),
                    'targets_qa_set': set(),
                }
                results.append(result_row)

    # Create DataFrame
    mt_data_df = pd.DataFrame(results)

    if len(mt_data_df) == 0:
        print("Warning: No successful analysis results")
        return mt_data_df

    # Sort by date and ISIN
    mt_data_df = mt_data_df.sort_values(['isin', 'date']).reset_index(drop=True)

    print(f"Raw MT data calculation completed. Generated {len(mt_data_df)} records.")
    print(f"Date range: {mt_data_df['date'].min()} to {mt_data_df['date'].max()}")
    print(f"Unique ISINs: {mt_data_df['isin'].nunique()}")

    return mt_data_df


async def calculate_raw_mt_data_llm(isin_list: List[str],
                                    date_range: Tuple[str, str],
                                    earnings_dir: str = "data/earnings",
                                    max_concurrent: int = 48) -> pd.DataFrame:
    """
    Calculate raw MT data (target counts and sets) using LLM extraction method with async processing.

    Args:
        isin_list: List of ISIN codes
        date_range: Tuple of (start_date, end_date) strings
        earnings_dir: Directory containing HTML earnings files
        max_concurrent: Maximum number of concurrent LLM calls (default: 48)

    Returns:
        DataFrame with raw target data and sets
    """
    print(
        f"Calculating raw MT data for {len(isin_list)} ISINs from {date_range[0]} to {date_range[1]} using LLM method"
    )

    # Scan for relevant files
    file_info_list = scan_earnings_files(earnings_dir, isin_list, date_range)
    print(f"Found {len(file_info_list)} earnings call files to process")

    if len(file_info_list) == 0:
        print("Warning: No files found matching criteria")
        return pd.DataFrame()

    print(f"Processing files with {max_concurrent} max concurrent operations...")

    results = []
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create tasks for all files
    tasks = [
        process_single_earnings_file(file_info, "llm", semaphore)
        for file_info in file_info_list
    ]

    # Process all tasks concurrently while preserving order
    print("Processing files concurrently...")
    task_results = await tqdm_asyncio.gather(*tasks, desc="Processing files", total=len(file_info_list))

    # Process results in original order
    for i, result in enumerate(task_results):
        file_info = file_info_list[i]

        print(f"Processing {i + 1}/{len(file_info_list)}: {file_info['filename']}")

        if isinstance(result, Exception):
            print(f"Error processing {file_info['filename']}: {result}")
            result = None

        if result is not None:
            results.append(result)
        else:
            # Add a row with NaN/empty values for failed processing
            result_row = {
                'date': file_info['date'],
                'isin': file_info['isin'],
                'quarter': file_info['quarter'],
                'filename': file_info['filename'],
                'num_total_targets': np.nan,
                'num_presentation_targets': np.nan,
                'num_qa_targets': np.nan,
                'num_total_chunks': np.nan,
                'targets_all_set': set(),
                'targets_presentation_set': set(),
                'targets_qa_set': set(),
            }
            results.append(result_row)

    # Create DataFrame
    mt_data_df = pd.DataFrame(results)

    if len(mt_data_df) == 0:
        print("Warning: No successful analysis results")
        return mt_data_df

    # Sort by date and ISIN
    mt_data_df = mt_data_df.sort_values(['isin', 'date']).reset_index(drop=True)

    print(f"Raw MT data calculation completed. Generated {len(mt_data_df)} records.")
    print(f"Date range: {mt_data_df['date'].min()} to {mt_data_df['date'].max()}")
    print(f"Unique ISINs: {mt_data_df['isin'].nunique()}")
    os.makedirs("output", exist_ok=True)
    mt_data_df.to_csv("output/log.csv", index=False)

    return mt_data_df


def calculate_conditional_mt_scores_base(df: pd.DataFrame, score_scope: str = "presentation") -> pd.DataFrame:
    """
    Calculate MT scores using 4-quarter lag conditional formula:
    MT_t = Σ(Missing Targets_t | Targets_{t-4}) / Σ Targets_{t-4}

    Args:
        df: DataFrame with raw MT data including target sets
        score_scope: "presentation" (default) for presentation-only targets,
                    "all" for presentation + Q&A targets combined

    Returns:
        DataFrame with MT scores added
    """
    print(f"Calculating conditional MT scores (4-quarter lag, scope: {score_scope})...")

    # Validate score_scope parameter
    if score_scope not in ["presentation", "all"]:
        raise ValueError(f"score_scope must be 'presentation' or 'all', got '{score_scope}'")

    # Select appropriate target set column based on scope
    if score_scope == "presentation":
        target_set_col = 'targets_presentation_set'
    else:  # scope == "all"
        target_set_col = 'targets_all_set'

    df = df.copy()
    df['mt_score'] = np.nan

    # Sort by ISIN and date for easier processing
    df = df.sort_values(['isin', 'date']).reset_index(drop=True)

    mt_scores_calculated = 0

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Find same company data, sorted by date
        same_company = df[df['isin'] == row['isin']].copy()
        same_company = same_company.sort_values('date').reset_index(drop=True)

        # Find current row position in same_company data
        current_date_matches = same_company[same_company['date'] == row['date']]
        if len(current_date_matches) == 0:
            continue

        current_pos = current_date_matches.index[0]

        # Look for 4 quarters back (4 positions back in sorted data)
        if current_pos >= 4:
            prev_row = same_company.iloc[current_pos - 4]

            prev_targets = prev_row[target_set_col]
            curr_targets = row[target_set_col]

            if len(prev_targets) > 0:
                missing_targets = prev_targets - curr_targets
                df.loc[idx, 'mt_score'] = len(missing_targets) / len(prev_targets)
                mt_scores_calculated += 1
            else:
                df.loc[idx, 'mt_score'] = np.nan  # No previous targets

    print(f"MT score calculation completed. Calculated scores for {mt_scores_calculated} observations.")

    return df


def calculate_conditional_mt_scores_llm(df: pd.DataFrame, score_scope: str = "presentation",
                                        state_dict: Optional[OrderedDict] = None) -> pd.DataFrame:
    """
    Calculate MT scores using LLM-specific logic (4-quarter lag conditional formula).
    Optimized version that pre-groups data and batches embedding calculations.

    Args:
        df: DataFrame with raw MT data including target sets
        score_scope: "presentation" or "all"
        state_dict: Embedding cache state dictionary

    Returns:
        DataFrame with MT scores added
    """
    print(f"Calculating conditional MT scores (4-quarter lag, scope: {score_scope})...")

    # Validate score_scope parameter
    if score_scope not in ["presentation", "all"]:
        raise ValueError(f"score_scope must be 'presentation' or 'all', got '{score_scope}'")

    # Select appropriate target set column based on scope
    if score_scope == "presentation":
        target_set_col = 'targets_presentation_set'
    else:  # scope == "all"
        target_set_col = 'targets_all_set'

    df = df.copy()
    df['mt_score'] = np.nan

    # Sort by ISIN and date for easier processing
    df = df.sort_values(['isin', 'date']).reset_index(drop=True)

    # Group by ISIN to avoid repeated filtering
    grouped = df.groupby('isin')
    mt_scores_calculated = 0

    # Collect all unique targets for batch embedding
    all_targets = set()
    target_pairs = []  # [(prev_targets, curr_targets, df_idx)]

    print("Collecting target pairs...")
    for isin, group in tqdm(grouped, desc="Processing companies"):
        # Keep original index before sorting and reset
        original_indices = group.index.tolist()
        group_sorted = group.sort_values('date').reset_index(drop=True)

        for i in range(4, len(group_sorted)):  # Start from index 4 (5th row)
            curr_row_idx = original_indices[i]  # Get original DataFrame index
            prev_targets = group_sorted.iloc[i-4][target_set_col]
            curr_targets = group_sorted.iloc[i][target_set_col]

            if len(prev_targets) > 0 and len(curr_targets) > 0:
                target_pairs.append((prev_targets, curr_targets, curr_row_idx))
                all_targets.update(prev_targets)
                all_targets.update(curr_targets)

    if not target_pairs:
        print("No valid target pairs found for MT score calculation.")
        return df

    print(f"Found {len(target_pairs)} target pairs to process")
    print(f"Total unique targets: {len(all_targets)}")

    # Batch compute embeddings for all unique targets
    print("Computing embeddings for all unique targets...")
    all_targets_list = list(all_targets)

    async def compute_embeddings_with_progress():
        return await get_embedding_with_cache(all_targets_list, state_dict=state_dict, return_list=True)

    all_embeddings = asyncio.run(compute_embeddings_with_progress())

    # Create target to index mapping for fast lookup
    target_to_idx = {target: i for i, target in enumerate(all_targets_list)}

    print("Calculating MT scores...")

    # Parallel processing function
    def compute_single_score(args):
        prev_targets, curr_targets, df_idx = args
        try:
            # Get embeddings using index lookup
            prev_embeddings = torch.stack([all_embeddings[target_to_idx[target]] for target in prev_targets])
            curr_embeddings = torch.stack([all_embeddings[target_to_idx[target]] for target in curr_targets])

            similarity = get_similarity(prev_embeddings, curr_embeddings)
            max_vals, _ = torch.max(similarity, 1)

            # Similarity threshold: <=0.4 = 0 (missing), >=0.6 = 1 (matched), linear interpolation between
            max_vals = torch.clamp(max_vals, min=0)
            max_vals = torch.where(max_vals <= 0.4, torch.zeros_like(max_vals),
                       torch.where(max_vals >= 0.6, torch.ones_like(max_vals),
                                   (max_vals - 0.4) / 0.2))
            missing_targets_score = 1 - (torch.sum(max_vals).item() / len(prev_targets))

            return (missing_targets_score, df_idx)
        except Exception as e:
            print(f"Error processing pair {df_idx}: {e}")
            return (None, df_idx)

    # Use ThreadPoolExecutor for parallel processing (better for I/O bound tasks with shared memory)
    max_workers = min(cpu_count(), 8)  # Limit to avoid memory issues
    print(f"Processing {len(target_pairs)} target pairs with {max_workers} workers...")

    scores = []
    indices = []
    failed_pairs = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_pair = {executor.submit(compute_single_score, pair): pair for pair in target_pairs}

        # Process results with progress bar
        for future in tqdm(as_completed(future_to_pair), total=len(target_pairs), desc="Computing scores"):
            try:
                score, df_idx = future.result()
                if score is not None:
                    scores.append(score)
                    indices.append(df_idx)
                    mt_scores_calculated += 1
                else:
                    failed_pairs.append(future_to_pair[future])
            except Exception as e:
                failed_pairs.append(future_to_pair[future])
                print(f"Error processing target pair: {e}")

    # Report failed pairs
    if failed_pairs:
        print(f"Warning: Failed to process {len(failed_pairs)} target pairs out of {len(target_pairs)}")
        print(f"Success rate: {(len(target_pairs) - len(failed_pairs)) / len(target_pairs) * 100:.1f}%")

    # Batch update DataFrame using vectorized operation
    if indices:
        df.loc[indices, 'mt_score'] = scores

    print(f"MT score calculation completed. Calculated scores for {mt_scores_calculated} observations.")
    return df


def calculate_mt_scores(isin_list: List[str],
                        date_range: Tuple[str, str],
                        earnings_dir: str = "data/earnings",
                        extraction_method: str = "base",
                        score_scope: str = "presentation") -> pd.DataFrame:
    """
    Main wrapper function to calculate Moving Targets scores using specified extraction method.

    This function:
    1. Scans and processes earnings call HTML files
    2. Extracts target counts and sets using the specified method (base or llm)
    3. Calculates conditional MT scores using 4-quarter lag

    Args:
        isin_list: List of ISIN codes
        date_range: Tuple of (start_date, end_date) strings
        earnings_dir: Directory containing HTML earnings files
        extraction_method: "base" for spaCy-based extraction, "llm" for LLM-based extraction
        score_scope: "presentation" (default) for presentation-only targets,
                    "all" for presentation + Q&A targets combined

    Returns:
        DataFrame with MT scores and all supporting data
    """

    # Step 1: Calculate raw MT data using the specified method
    if extraction_method == "llm":
        embedding_state_dict = OrderedDict(torch.load("output/mt_embeddings.pt")) if os.path.exists(
            "output/mt_embeddings.pt"
        ) else OrderedDict()
        mt_data_df = asyncio.run(calculate_raw_mt_data_llm(isin_list, date_range, earnings_dir))
        calculate_scores_func = lambda df: calculate_conditional_mt_scores_llm(df, score_scope, embedding_state_dict)

    else:  # default to base
        mt_data_df = calculate_raw_mt_data_base(isin_list, date_range, earnings_dir)
        calculate_scores_func = lambda df: calculate_conditional_mt_scores_base(df, score_scope)

    if len(mt_data_df) == 0:
        return mt_data_df

    # Step 2: Calculate conditional MT scores using method-specific logic
    mt_score_df = calculate_scores_func(mt_data_df)

    if extraction_method == "llm":
        os.makedirs("output", exist_ok=True)
        torch.save(embedding_state_dict, "output/mt_embeddings.pt")
    return mt_score_df


def get_mt_score_summary(mt_score_df: pd.DataFrame) -> str:
    """
    Generate a summary report of MT scores.

    Args:
        mt_score_df: DataFrame with MT scores

    Returns:
        Formatted summary string
    """
    if len(mt_score_df) == 0:
        return "No MT score data available."

    summary = "## Moving Targets Score Summary\n\n"
    summary += f"**Data Overview:**\n"
    summary += f"- Total observations: {len(mt_score_df)}\n"
    summary += f"- Unique companies (ISINs): {mt_score_df['isin'].nunique()}\n"
    summary += f"- Date range: {mt_score_df['date'].min()} to {mt_score_df['date'].max()}\n"
    summary += f"- Unique quarters: {mt_score_df['quarter'].nunique()}\n\n"

    # Target statistics
    summary += f"**Target Statistics:**\n"
    summary += f"- Average total targets per call: {mt_score_df['num_total_targets'].mean():.2f}\n"
    summary += f"- Average presentation targets: {mt_score_df['num_presentation_targets'].mean():.2f}\n"
    summary += f"- Average Q&A targets: {mt_score_df['num_qa_targets'].mean():.2f}\n\n"

    # MT Score statistics
    mt_scores = mt_score_df['mt_score'].dropna()
    if len(mt_scores) > 0:
        summary += f"**MT Score Statistics:**\n"
        summary += f"- Observations with MT scores: {len(mt_scores)}\n"
        summary += f"- Mean MT score: {mt_scores.mean():.4f}\n"
        summary += f"- Std MT score: {mt_scores.std():.4f}\n"
        summary += f"- Min MT score: {mt_scores.min():.4f}\n"
        summary += f"- Max MT score: {mt_scores.max():.4f}\n"
        summary += f"- Median MT score: {mt_scores.median():.4f}\n\n"
    else:
        summary += f"**No MT scores calculated** (need at least 4 quarters of data per company)\n\n"

    return summary


if __name__ == "__main__":
    # Example usage
    print("Moving Targets Score Calculation Module")
    print("See run.py for usage examples.")
