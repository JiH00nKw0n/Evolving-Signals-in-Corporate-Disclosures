#!/usr/bin/env python3
"""
End-to-End Moving Targets Pipeline

This script runs the complete pipeline from ISIN list and date range
to MT score calculation, regression analysis, and backtesting.

Pipeline Steps:
1. ISIN List + Date Range -> MT Score Calculation (src/mt_score.py)
2. MT Score DataFrame -> Fama-MacBeth Regression (src/regression.py)
3. MT Score DataFrame -> Portfolio Backtest (src/backtest.py)

Usage:
    python run.py --mode sample     # Quick single-company test (base NER)
    python run.py --mode full       # Full analysis with auto-detected data
    python run.py --mode demo       # MT score calculation demo only
    python run.py --mode compare    # LLM vs NER comparison
    python run.py --method llm      # Use LLM extraction (default: base)
    python run.py --cache path.csv  # Load cached MT scores
"""

import argparse
import os
import warnings
from collections import OrderedDict
from datetime import datetime
from typing import List, Tuple

import pandas as pd
import torch

warnings.filterwarnings('ignore')

# Import our modules
from src.mt_score import calculate_mt_scores, get_mt_score_summary, calculate_conditional_mt_scores_llm, calculate_conditional_mt_scores_base
from src.regression import run_mt_score_analysis, run_multi_specification_analysis
from src.backtest import run_moving_targets_backtest, MovingTargetsBacktest, compare_sharpe_ratios_paired


def load_train_data(extraction_method: str = "base") -> pd.DataFrame:
    """Load stock returns data."""

    # Always use SNP100 data for fair comparison
    file_path = "data/TRAIN_DATA_FMP_SNP100.csv"
    print(f"Loading TRAIN_DATA_FMP_SNP100.csv for {extraction_method} method...")

    # Read CSV with proper separator
    df = pd.read_csv(file_path, sep='|')

    # Clean up column names (remove extra characters if any)
    df.columns = df.columns.str.strip()

    # Convert date column
    df['DATE'] = pd.to_datetime(df['DATE'])

    print(f"Loaded {len(df)} observations from {df['DATE'].min()} to {df['DATE'].max()}")
    print(f"Unique ISINs: {df['ISIN'].nunique()}")

    return df


def save_mt_score_data(mt_score_df: pd.DataFrame,
                       isin_list: List[str],
                       date_range: Tuple[str, str]) -> None:
    """
    Save MT score DataFrame to output folder with timestamp and metadata.

    Args:
        mt_score_df: DataFrame with MT scores
        isin_list: List of ISINs analyzed
        date_range: Date range used for analysis
    """

    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create filename with metadata
    num_isins = len(isin_list)
    start_date = date_range[0].replace('-', '')
    end_date = date_range[1].replace('-', '')

    # Save full MT score DataFrame (with sets converted to strings for CSV)
    mt_score_clean = mt_score_df.copy()

    # Convert sets to string representation for CSV storage
    for col in ['targets_all_set', 'targets_presentation_set', 'targets_qa_set']:
        if col in mt_score_clean.columns:
            mt_score_clean[col] = mt_score_clean[col].apply(lambda x: '|'.join(sorted(x)) if x else '')

    csv_filename = f"mt_scores_{num_isins}companies_{start_date}to{end_date}_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    mt_score_clean.to_csv(csv_path, index=False)
    print(f"MT Score DataFrame saved to: {csv_path}")

    # Save summary statistics
    mt_scores = mt_score_df['mt_score'].dropna()

    summary_data = {
        'analysis_timestamp': datetime.now().isoformat(),
        'isin_list': isin_list,
        'date_range': date_range,
        'total_observations': len(mt_score_df),
        'valid_mt_scores': len(mt_scores),
        'unique_companies': mt_score_df['isin'].nunique(),
        'unique_quarters': mt_score_df['quarter'].nunique() if 'quarter' in mt_score_df.columns else 0,
        'mt_score_mean': mt_scores.mean() if len(mt_scores) > 0 else None,
        'mt_score_std': mt_scores.std() if len(mt_scores) > 0 else None,
        'mt_score_min': mt_scores.min() if len(mt_scores) > 0 else None,
        'mt_score_max': mt_scores.max() if len(mt_scores) > 0 else None,
        'avg_total_targets': mt_score_df['num_total_targets'].mean(),
        'avg_presentation_targets': mt_score_df['num_presentation_targets'].mean(),
        'avg_qa_targets': mt_score_df['num_qa_targets'].mean()
    }

    # Save metadata as JSON
    import json
    json_filename = f"mt_scores_metadata_{num_isins}companies_{start_date}to{end_date}_{timestamp}.json"
    json_path = os.path.join(output_dir, json_filename)

    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)

    print(f"MT Score metadata saved to: {json_path}")

    # Save a simplified CSV with key columns only (for easier analysis)
    key_columns = ['date', 'isin', 'quarter', 'num_total_targets', 'num_presentation_targets',
                   'num_qa_targets', 'mt_score', 'filename']

    if all(col in mt_score_df.columns for col in key_columns):
        simple_df = mt_score_df[key_columns].copy()
        simple_filename = f"mt_scores_simple_{num_isins}companies_{start_date}to{end_date}_{timestamp}.csv"
        simple_path = os.path.join(output_dir, simple_filename)

        simple_df.to_csv(simple_path, index=False)
        print(f"Simplified MT Score data saved to: {simple_path}")

    print(f"Summary: {len(mt_score_df)} observations, {len(mt_scores)} valid MT scores")


def save_pipeline_results(results: dict,
                          filename: str,
                          analysis_config: dict = None) -> None:
    """
    Save pipeline results to markdown file.

    Args:
        results: Dictionary containing pipeline results
        filename: Output filename for markdown
        analysis_config: Optional analysis configuration info
    """
    print(f"\nSaving pipeline results to '{filename}'...")

    summary_text = "# Moving Targets Pipeline Results\n\n"

    # Add analysis configuration if provided
    if analysis_config:
        summary_text += "**Analysis Configuration:**\n"
        for key, value in analysis_config.items():
            summary_text += f"- {key}: {value}\n"
        summary_text += "\n"

    # MT Score Summary
    if 'mt_summary' in results:
        summary_text += results['mt_summary'] + "\n\n"

    # Multi-Specification Regression Results
    if 'multi_spec_results' in results:
        summary_text += "## Multi-Specification Fama-MacBeth Regression\n\n"
        summary_text += results['multi_spec_results']['combined_table'] + "\n\n"
    elif 'regression_error' in results:
        summary_text += "## Multi-Specification Fama-MacBeth Regression\n\n"
        summary_text += f"**Regression Analysis Failed:** {results['regression_error']}\n\n"
        summary_text += "This typically occurs when there are insufficient observations per month for cross-sectional regressions.\n"
        summary_text += "Consider using more companies or longer time periods for better results.\n\n"

    # Single Specification Results
    if 'single_spec_results' in results:
        summary_text += "## Single Specification Regression\n\n"
        summary_text += results['single_spec_results']['results_table'] + "\n\n"

    # Portfolio Backtest Results
    if 'backtest_table' in results:
        summary_text += "## Portfolio Backtest Results\n\n"
        summary_text += results['backtest_table'] + "\n\n"
    elif 'backtest_error' in results:
        summary_text += "## Portfolio Backtest Results\n\n"
        summary_text += f"**Backtest Failed:** {results['backtest_error']}\n\n"

    # Write to file
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w") as f:
        f.write(summary_text)

    print(f"Results saved to '{output_path}'")


def run_complete_mt_pipeline(isin_list: List[str],
                             date_range: Tuple[str, str],
                             mt_score_df: pd.DataFrame = None,
                             extraction_method: str = "base") -> dict:
    """
    Run the complete Moving Targets pipeline.

    Args:
        isin_list: List of ISIN codes to analyze
        date_range: Tuple of (start_date, end_date) strings
        mt_score_df: Optional pre-computed MT score DataFrame (for caching)
        extraction_method: "base" or "llm"

    Returns:
        Dictionary with all results
    """

    print("=" * 60)
    print("MOVING TARGETS PIPELINE - END-TO-END ANALYSIS")
    print("=" * 60)

    results = {}

    # ========================================
    # Step 1: Calculate or Load MT Scores
    # ========================================
    print("\n" + "=" * 40)
    if mt_score_df is not None:
        print("STEP 1: USING CACHED MT SCORES")
    else:
        print("STEP 1: CALCULATING MT SCORES")
    print("=" * 40)

    if mt_score_df is None:
        mt_score_df = calculate_mt_scores(
            isin_list=isin_list,
            date_range=date_range,
            earnings_dir="data/earnings",
            extraction_method=extraction_method,
            score_scope="presentation"  # Cohen & Nguyen (2024) method
        )

        if len(mt_score_df) == 0:
            print("No MT scores calculated. Pipeline terminated.")
            return {"error": "No MT scores calculated"}

        # Save MT score DataFrame to output folder
        print("\nSaving MT Score DataFrame...")
        save_mt_score_data(mt_score_df, isin_list, date_range)
    else:
        print(f"Using cached MT score data: {len(mt_score_df)} observations")
        if extraction_method == "base":
            mt_score_df = calculate_conditional_mt_scores_base(mt_score_df, score_scope="presentation")
        else:
            mt_score_df = calculate_conditional_mt_scores_llm(
                mt_score_df,
                score_scope="presentation",
                state_dict=OrderedDict(torch.load("output/mt_embeddings.pt")) if os.path.exists(
                    "output/mt_embeddings.pt"
                ) else OrderedDict()
            )

    # Generate MT score summary
    mt_summary = get_mt_score_summary(mt_score_df)
    print(mt_summary)

    results['mt_score_df'] = mt_score_df
    results['mt_summary'] = mt_summary

    # ========================================
    # Step 2: Load Returns Data
    # ========================================
    print("\n" + "=" * 40)
    print("STEP 2: LOADING RETURNS DATA")
    print("=" * 40)

    try:
        returns_df = load_train_data(extraction_method)
        results['returns_df'] = returns_df
    except Exception as e:
        print(f"Error loading returns data: {e}")
        return {"error": f"Failed to load returns data: {e}"}

    # Filter returns data to relevant ISINs and date range
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])

    filtered_returns = returns_df[
        (returns_df['ISIN'].isin(isin_list)) &
        (returns_df['DATE'] >= start_date) &
        (returns_df['DATE'] <= end_date)
        ].copy()

    print(f"Filtered returns data: {len(filtered_returns)} observations")
    results['filtered_returns'] = filtered_returns

    # ========================================
    # Step 3: Multi-Specification Regression Analysis
    # ========================================
    print("\n" + "=" * 40)
    print("STEP 3: MULTI-SPECIFICATION FAMA-MACBETH REGRESSION")
    print("=" * 40)

    try:
        # Run multi-specification analysis (similar to paper's Table IV)
        multi_spec_results = run_multi_specification_analysis(
            mt_score_df=mt_score_df,
            returns_df=filtered_returns,
            lag_months=1
        )

        print(multi_spec_results['combined_table'])

        # Print summary statistics
        summary_stats = multi_spec_results['summary_stats']
        print(f"\nMulti-Specification Summary:")
        print(f"- Total observations: {summary_stats['total_observations']}")
        print(f"- Unique companies: {summary_stats['unique_companies']}")
        print(f"- Unique months: {summary_stats['unique_months']}")
        print(f"- Specifications run: {summary_stats['specifications_run']}")
        print(f"- MT score mean: {summary_stats['mt_score_mean']:.4f}")
        print(f"- Returns mean: {summary_stats['returns_mean']:.4f}")

        results['multi_spec_results'] = multi_spec_results

        # Also run single specification for comparison
        print(f"\n" + "-" * 30)
        print("SINGLE SPECIFICATION (MT Score + All Available Controls)")
        print("-" * 30)

        # Map our column names to control variables
        available_controls = ['log_size', 'log_bm', 'ret_lag1', 'ret_lag12_2', 'beta', 'momentum']

        single_spec_results = run_mt_score_analysis(
            mt_score_df=mt_score_df,
            returns_df=filtered_returns,
            control_variables=available_controls,
            lag_months=1
        )

        print(single_spec_results['results_table'])
        results['single_spec_results'] = single_spec_results

    except Exception as e:
        print(f"Regression analysis failed: {e}")
        results['regression_error'] = str(e)

    # ========================================
    # Step 4: Portfolio Backtest
    # ========================================
    print("\n" + "=" * 40)
    print("STEP 4: PORTFOLIO BACKTEST")
    print("=" * 40)

    try:
        # Run equal-weighted backtest
        backtest = MovingTargetsBacktest(is_value_weighted=False)
        backtest.load_factor_data(
            factor_3_path="data/F-F_Research_Data_Factors.tsv",
            factor_5_path="data/F-F_Research_Data_5_Factors_2x3.tsv"
        )
        portfolio_returns, backtest_results, results_table, sharpe_results = backtest.run_backtest(
            mt_score_df=mt_score_df,
            returns_df=filtered_returns,
            score_col='mt_score'
        )

        print(results_table)

        # Calculate additional portfolio statistics
        if len(portfolio_returns) > 0:
            print(f"\nPortfolio Summary:")
            print(f"- Portfolio periods: {len(portfolio_returns)}")
            print(f"- Date range: {portfolio_returns.index[0]} to {portfolio_returns.index[-1]}")

            # Show average returns by quintile
            print(f"\nAverage Monthly Returns:")
            for col in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q5-Q1']:
                if col in portfolio_returns.columns:
                    avg_ret = portfolio_returns[col].mean()
                    std_ret = portfolio_returns[col].std()
                    print(f"- {col}: {avg_ret:.4f} (+/-{std_ret:.4f})")

        results['portfolio_returns'] = portfolio_returns
        results['backtest_results'] = backtest_results
        results['backtest_table'] = results_table
        results['sharpe_results'] = sharpe_results

    except Exception as e:
        print(f"Backtest failed: {e}")
        results['backtest_error'] = str(e)

    # ========================================
    # Step 5: Pipeline Summary
    # ========================================
    print("\n" + "=" * 40)
    print("PIPELINE COMPLETED")
    print("=" * 40)

    print(f"MT Scores: {len(mt_score_df)} observations")
    if 'multi_spec_results' in results:
        print(
            f"Multi-Spec Regression: {results['multi_spec_results']['summary_stats']['total_observations']} merged observations"
            )
        print(f"Specifications run: {results['multi_spec_results']['summary_stats']['specifications_run']}")
    if 'portfolio_returns' in results:
        print(f"Backtest: {len(results['portfolio_returns'])} portfolio periods")

    return results


def get_available_data_info(extraction_method: str = "base",
                            earnings_dir: str = "data/earnings") -> dict:
    """
    Get available ISINs and date ranges from both returns data and earnings files.

    Returns:
        Dict with available ISINs, date ranges, and intersection info
    """
    print("Analyzing available data...")

    # Always use SNP100 data for fair comparison
    returns_file = "data/TRAIN_DATA_FMP_SNP100.csv"

    # Load returns data
    returns_df = pd.read_csv(returns_file, sep='|')
    returns_df.columns = returns_df.columns.str.strip()
    returns_df['DATE'] = pd.to_datetime(returns_df['DATE'])

    returns_isins = set(returns_df['ISIN'].unique())
    returns_date_min = returns_df['DATE'].min()
    returns_date_max = returns_df['DATE'].max()

    print(f"Returns data ({returns_file.split('/')[-1]}):")
    print(f"   - ISINs: {len(returns_isins)}")
    print(f"   - Date range: {returns_date_min.date()} to {returns_date_max.date()}")

    # Scan earnings files
    from src.mt_score import extract_isin_date_from_filename

    earnings_isins = set()
    earnings_dates = []

    if os.path.exists(earnings_dir):
        html_files = [f for f in os.listdir(earnings_dir) if f.endswith('.html')]
        for filename in html_files:
            parsed = extract_isin_date_from_filename(filename)
            if parsed:
                isin, date_str, quarter = parsed
                earnings_isins.add(isin)
                try:
                    earnings_dates.append(pd.to_datetime(date_str))
                except:
                    continue

    earnings_date_min = min(earnings_dates) if earnings_dates else None
    earnings_date_max = max(earnings_dates) if earnings_dates else None

    print(f"Earnings data:")
    print(f"   - ISINs: {len(earnings_isins)}")
    print(
        f"   - Date range: {earnings_date_min.date() if earnings_date_min else 'N/A'} to {earnings_date_max.date() if earnings_date_max else 'N/A'}"
        )

    # Find intersection
    common_isins = returns_isins.intersection(earnings_isins)

    # Get overlapping date range
    if earnings_date_min and earnings_date_max:
        overlap_start = max(returns_date_min, earnings_date_min)
        overlap_end = min(returns_date_max, earnings_date_max)
    else:
        overlap_start = returns_date_min
        overlap_end = returns_date_max

    print(f"Overlapping data:")
    print(f"   - Common ISINs: {len(common_isins)}")
    print(f"   - Overlap date range: {overlap_start.date()} to {overlap_end.date()}")

    return {
        'returns_isins': returns_isins,
        'earnings_isins': earnings_isins,
        'common_isins': list(common_isins),
        'returns_date_range': (returns_date_min, returns_date_max),
        'earnings_date_range': (earnings_date_min, earnings_date_max),
        'overlap_date_range': (overlap_start, overlap_end),
        'total_returns_obs': len(returns_df),
        'common_isins_count': len(common_isins)
    }


def find_cached_results(output_dir: str = "output") -> Tuple[str, str]:
    """
    Find cached MT score results in output directory.

    Args:
        output_dir: Directory to search for cached results

    Returns:
        Tuple of (csv_cache_path, md_result_path) if found, (None, None) otherwise
    """
    import re
    import glob as glob_module

    if not os.path.exists(output_dir):
        return None, None

    # Find all md files matching the pattern
    md_pattern = os.path.join(output_dir, "mt_scores_*companies_*to*_*.md")
    md_files = glob_module.glob(md_pattern)

    if not md_files:
        return None, None

    # Sort by modification time (newest first)
    md_files.sort(key=os.path.getmtime, reverse=True)

    for md_file in md_files:
        # Extract base pattern from md filename
        basename = os.path.basename(md_file)

        match = re.match(r'(mt_scores_\d+companies_\d+to\d+_\d+_\d+)_\d+_\d+\.md$', basename)

        if match:
            csv_base = match.group(1)
            csv_filename = csv_base + ".csv"
            csv_path = os.path.join(output_dir, csv_filename)

            if os.path.exists(csv_path):
                return csv_path, md_file

            # Also check in subdirectories (e.g., snp100_llm/)
            for subdir in os.listdir(output_dir):
                subdir_path = os.path.join(output_dir, subdir)
                if os.path.isdir(subdir_path):
                    csv_path_sub = os.path.join(subdir_path, csv_filename)
                    if os.path.exists(csv_path_sub):
                        return csv_path_sub, md_file

    return None, None


def run_full_analysis(extraction_method: str = "base", cache_file_path: str = None):
    """
    Run full analysis using all available data from returns CSV and earnings files.

    Args:
        extraction_method: "base" for spaCy-based extraction, "llm" for LLM-based extraction
        cache_file_path: Path to cached MT score CSV file.
    """

    print("Running Full Moving Targets Analysis")
    print("=" * 50)

    # Auto-detect cached results if cache_file_path not provided
    if cache_file_path is None:
        cached_csv, cached_md = find_cached_results("output")
        if cached_csv and cached_md:
            print(f"\nFound cached results:")
            print(f"   - CSV cache: {cached_csv}")
            print(f"   - MD result: {cached_md}")
            cache_file_path = cached_csv

    # Get available data info
    data_info = get_available_data_info(extraction_method)

    if len(data_info['common_isins']) == 0:
        print("No common ISINs found between returns and earnings data")
        return

    # Select ISINs to analyze (limit for performance)
    available_isins = data_info['common_isins']
    overlap_start, overlap_end = data_info['overlap_date_range']

    date_range = (overlap_start.strftime('%Y-%m-%d'), overlap_end.strftime('%Y-%m-%d'))

    # Check for cached MT score data
    mt_score_df = None
    if cache_file_path and os.path.exists(cache_file_path):
        print(f"\nLoading cached MT score data from: {cache_file_path}")
        try:
            mt_score_df = pd.read_csv(cache_file_path)

            # Convert string sets back to actual sets for columns that need them
            for col in ['targets_all_set', 'targets_presentation_set', 'targets_qa_set']:
                if col in mt_score_df.columns:
                    mt_score_df[col] = mt_score_df[col].apply(
                        lambda x: set(x.split('|')) if pd.notna(x) and x != '' else set()
                    )

            print(f"Loaded {len(mt_score_df)} cached MT score observations")
            print(f"   - Unique companies: {mt_score_df['isin'].nunique()}")
            print(f"   - Date range: {mt_score_df['date'].min()} to {mt_score_df['date'].max()}")
        except Exception as e:
            print(f"Failed to load cached data: {e}")
            print("Continuing with fresh MT score calculation...")
            mt_score_df = None

    print(f"\nAnalysis Configuration:")
    print(f"   - Selected ISINs: {len(available_isins)} (out of {len(data_info['common_isins'])} available)")
    print(f"   - Date range: {date_range[0]} to {date_range[1]}")
    print(f"   - Cache file: {cache_file_path if cache_file_path else 'None'}")
    print(f"   - Extraction method: {extraction_method}")
    print(f"   - Selected companies:")

    # Show selected ISINs
    for i, isin in enumerate(available_isins[:5]):  # Show first 5
        print(f"     {i + 1}. {isin}")
    if len(available_isins) > 5:
        print(f"     ... and {len(available_isins) - 5} more")

    try:
        # Run complete pipeline with automatically detected data
        pipeline_results = run_complete_mt_pipeline(
            isin_list=available_isins,
            date_range=date_range,
            mt_score_df=mt_score_df,
            extraction_method=extraction_method
        )

        if 'error' in pipeline_results:
            print(f"Pipeline failed: {pipeline_results['error']}")
            return

        # Save results with configuration info
        config_info = {
            "Companies analyzed": len(available_isins),
            "Date range": f"{date_range[0]} to {date_range[1]}",
            "Total companies available": len(data_info['common_isins']),
            "Returns observations": f"{data_info['total_returns_obs']:,}"
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if cache_file_path:
            base_name = os.path.basename(cache_file_path).replace(".csv", "")
            filename = f"{base_name}_{timestamp}.md"
        else:
            filename = f"mt_pipeline_full_analysis_{timestamp}.md"
        save_pipeline_results(pipeline_results, filename, config_info)

        return pipeline_results

    except Exception as e:
        print(f"Full analysis failed: {e}")
        import traceback
        traceback.print_exc()


def run_sample_analysis(extraction_method: str = "base", cache_file_path: str = None):
    """
    Run a sample analysis with pre-selected data for quick testing.

    Args:
        extraction_method: "base" for spaCy-based extraction, "llm" for LLM-based extraction
        cache_file_path: Path to cached MT score CSV file.
    """

    print("Running Sample Moving Targets Analysis")
    print("=" * 50)

    # Sample ISINs and date range (use actual ISINs from earnings files)
    sample_isins = [
        'US02079K3059',
    ]

    # Date range covering available earnings data
    date_range = ('2010-01-01', '2025-12-31')

    print(f"Using extraction method: {extraction_method}")

    # Check for cached MT score data
    mt_score_df = None
    if cache_file_path and os.path.exists(cache_file_path):
        print(f"\nLoading cached MT score data from: {cache_file_path}")
        try:
            mt_score_df = pd.read_csv(cache_file_path)

            # Convert string sets back to actual sets for columns that need them
            for col in ['targets_all_set', 'targets_presentation_set', 'targets_qa_set']:
                if col in mt_score_df.columns:
                    mt_score_df[col] = mt_score_df[col].apply(
                        lambda x: set(x.split('|')) if pd.notna(x) and x != '' else set()
                    )

            print(f"Loaded {len(mt_score_df)} cached MT score observations")
            print(f"   - Unique companies: {mt_score_df['isin'].nunique()}")
            print(f"   - Date range: {mt_score_df['date'].min()} to {mt_score_df['date'].max()}")
        except Exception as e:
            print(f"Failed to load cached data: {e}")
            print("Continuing with fresh MT score calculation...")
            mt_score_df = None

    print(f"\nSample Analysis Configuration:")
    print(f"   - Selected ISINs: {len(sample_isins)}")
    print(f"   - Date range: {date_range[0]} to {date_range[1]}")
    print(f"   - Cache file: {cache_file_path if cache_file_path else 'None'}")
    print(f"   - Extraction method: {extraction_method}")

    try:
        # Run complete pipeline
        pipeline_results = run_complete_mt_pipeline(
            isin_list=sample_isins,
            date_range=date_range,
            mt_score_df=mt_score_df,
            extraction_method=extraction_method
        )

        if 'error' in pipeline_results:
            print(f"Pipeline failed: {pipeline_results['error']}")
            return

        # Save results without configuration info for sample analysis
        save_pipeline_results(pipeline_results, "mt_pipeline_results.md")

        return pipeline_results

    except Exception as e:
        print(f"Sample analysis failed: {e}")
        import traceback
        traceback.print_exc()


def demo_mt_score_calculation():
    """
    Simple demo of just the MT score calculation step.
    """

    print("Demo: MT Score Calculation Only")
    print("=" * 40)

    # Use a smaller sample for quick testing
    sample_isins = ['BRPSSAACNOR7']  # Marathon Petroleum
    date_range = ('2015-01-01', '2018-12-31')

    try:
        mt_score_df = calculate_mt_scores(
            isin_list=sample_isins,
            date_range=date_range,
            score_scope="presentation"  # Cohen & Nguyen (2024) method
        )

        if len(mt_score_df) > 0:
            print("\nMT Score calculation successful!")
            print(f"Generated {len(mt_score_df)} observations")
            print("\nFirst 5 rows:")
            print(mt_score_df.head())

            # Show MT score statistics
            mt_scores = mt_score_df['mt_score'].dropna()
            if len(mt_scores) > 0:
                print(f"\nMT Score Statistics:")
                print(f"- Valid scores: {len(mt_scores)}")
                print(f"- Mean: {mt_scores.mean():.4f}")
                print(f"- Std: {mt_scores.std():.4f}")
                print(f"- Range: {mt_scores.min():.4f} to {mt_scores.max():.4f}")
            else:
                print("No valid MT scores (need 4+ quarters per company)")

            return mt_score_df

        else:
            print("No MT score data generated")
            return None

    except Exception as e:
        print(f"MT score calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_llm_vs_ner_comparison(
        llm_cache_path: str = "output/snp100_llm/mt_scores_98companies_20100129to20241231_20250929_024754.csv",
        ner_cache_path: str = "output/snp100_base/mt_scores_98companies_20100129to20241231_20250929_163157.csv"
):
    """
    Compare LLM vs NER extraction methods using Sharpe ratio statistical test.

    Uses Ledoit-Wolf (2008) method for paired Sharpe ratio comparison.
    """
    print("\n" + "=" * 60)
    print("LLM vs NER SHARPE RATIO COMPARISON")
    print("=" * 60)

    # Run both methods
    print("\nRunning LLM method...")
    results_llm = run_full_analysis(
        extraction_method="llm",
        cache_file_path=llm_cache_path
    )

    print("\nRunning NER method...")
    results_ner = run_full_analysis(
        extraction_method="base",
        cache_file_path=ner_cache_path
    )

    if results_llm is None or results_ner is None:
        print("One or both analyses failed. Cannot compare.")
        return None

    if 'portfolio_returns' not in results_llm or 'portfolio_returns' not in results_ner:
        print("Portfolio returns not available for comparison.")
        return None

    # Compare Sharpe ratios
    print("\n" + "=" * 60)
    print("SHARPE RATIO COMPARISON (Ledoit-Wolf 2008)")
    print("=" * 60)

    llm_returns = results_llm['portfolio_returns']
    ner_returns = results_ner['portfolio_returns']

    # Get risk-free rate
    factor_data = pd.read_csv("data/F-F_Research_Data_Factors.tsv", sep='\t')
    factor_data['Date'] = pd.to_datetime(factor_data['Date'].astype(str), format='%Y%m')
    factor_data['Date'] = factor_data['Date'] + pd.offsets.MonthEnd(0)
    factor_data.set_index('Date', inplace=True)
    factor_data['RF'] = factor_data['RF'] / 100
    rf = factor_data['RF']

    comparison_results = {}

    # Build comparison table string for saving
    comparison_table = "\n## LLM vs NER Sharpe Ratio Comparison (Ledoit-Wolf 2008)\n\n"
    comparison_table += "| Portfolio | LLM SR | NER SR | Diff (LLM-NER) | z-stat | p-value | Significant |\n"
    comparison_table += "|-----------|--------|--------|----------------|--------|---------|-------------|\n"

    print("\n| Portfolio | LLM SR | NER SR | Diff (LLM-NER) | z-stat | p-value | Significant |")
    print("|-----------|--------|--------|----------------|--------|---------|-------------|")

    for col in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q5-Q1']:
        if col in llm_returns.columns and col in ner_returns.columns:
            result = compare_sharpe_ratios_paired(
                llm_returns[col],
                ner_returns[col],
                rf=rf,
                method='ledoit_wolf'
            )

            if 'error' not in result:
                comparison_results[col] = result
                sig = "Yes***" if result['p_value'] < 0.01 else "Yes**" if result['p_value'] < 0.05 else "Yes*" if result['p_value'] < 0.10 else "No"
                # Display Q5-Q1 as Q1-Q5 with flipped Sharpe ratios
                if col == 'Q5-Q1':
                    display_col = 'Q1-Q5'
                    sr_a = -result['sharpe_a']
                    sr_b = -result['sharpe_b']
                    sr_diff = -result['sharpe_diff']
                    z_stat = -result['z_stat']
                else:
                    display_col = col
                    sr_a = result['sharpe_a']
                    sr_b = result['sharpe_b']
                    sr_diff = result['sharpe_diff']
                    z_stat = result['z_stat']
                row = f"| {display_col:9} | {sr_a:6.3f} | {sr_b:6.3f} | {sr_diff:14.3f} | {z_stat:6.2f} | {result['p_value']:7.4f} | {sig:11} |"
                print(row)
                comparison_table += row + "\n"

    comparison_table += "\n*p<0.10, **p<0.05, ***p<0.01\n"
    comparison_table += f"\nObservations: {comparison_results.get('Q5-Q1', {}).get('n_obs', 'N/A')}\n"
    if 'Q5-Q1' in comparison_results:
        comparison_table += f"Correlation (Q1-Q5): {comparison_results.get('Q5-Q1', {}).get('correlation', 'N/A'):.4f}\n"

    print("\n*p<0.10, **p<0.05, ***p<0.01")
    print(f"\nObservations: {comparison_results.get('Q5-Q1', {}).get('n_obs', 'N/A')}")
    if 'Q5-Q1' in comparison_results:
        print(f"Correlation (Q1-Q5): {comparison_results.get('Q5-Q1', {}).get('correlation', 'N/A'):.4f}")

    # Summary interpretation
    print("\n" + "-" * 40)
    print("INTERPRETATION:")
    print("-" * 40)

    comparison_table += "\n### Interpretation\n"

    if 'Q5-Q1' in comparison_results:
        q5q1 = comparison_results['Q5-Q1']
        # Display as Q1-Q5 (flipped)
        sr_diff_q1q5 = -q5q1['sharpe_diff']
        if q5q1['p_value'] < 0.05:
            winner = "LLM" if sr_diff_q1q5 > 0 else "NER"
            print(f"{winner} method significantly outperforms (p={q5q1['p_value']:.4f})")
            print(f"   Q1-Q5 Sharpe ratio difference: {sr_diff_q1q5:.3f} (annualized)")
            comparison_table += f"- **{winner} method significantly outperforms** (p={q5q1['p_value']:.4f})\n"
            comparison_table += f"- Q1-Q5 Sharpe ratio difference: {sr_diff_q1q5:.3f} (annualized)\n"
        else:
            print(f"No statistically significant difference between methods (p={q5q1['p_value']:.4f})")
            print(f"   Q1-Q5 Sharpe ratio difference: {sr_diff_q1q5:.3f} (annualized)")
            comparison_table += f"- No statistically significant difference between methods (p={q5q1['p_value']:.4f})\n"
            comparison_table += f"- Q1-Q5 Sharpe ratio difference: {sr_diff_q1q5:.3f} (annualized)\n"

    # Save comparison results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_filename = f"llm_vs_ner_comparison_{timestamp}.md"

    # Build full report
    full_report = "# LLM vs NER Method Comparison Report\n\n"
    full_report += f"**Generated:** {datetime.now().isoformat()}\n\n"

    # Add LLM results
    full_report += "---\n\n# LLM Method Results\n\n"
    if 'mt_summary' in results_llm:
        full_report += results_llm['mt_summary'] + "\n\n"
    if 'multi_spec_results' in results_llm:
        full_report += "## Fama-MacBeth Regression Results\n\n"
        full_report += results_llm['multi_spec_results']['combined_table'] + "\n\n"
    if 'backtest_table' in results_llm:
        full_report += results_llm['backtest_table'] + "\n\n"

    # Add NER results
    full_report += "---\n\n# NER (Base) Method Results\n\n"
    if 'mt_summary' in results_ner:
        full_report += results_ner['mt_summary'] + "\n\n"
    if 'multi_spec_results' in results_ner:
        full_report += "## Fama-MacBeth Regression Results\n\n"
        full_report += results_ner['multi_spec_results']['combined_table'] + "\n\n"
    if 'backtest_table' in results_ner:
        full_report += results_ner['backtest_table'] + "\n\n"

    # Add comparison
    full_report += "---\n\n"
    full_report += comparison_table

    # Save to file
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, comparison_filename)
    with open(output_path, 'w') as f:
        f.write(full_report)

    print(f"\nComparison results saved to: {output_path}")

    return {
        'results_llm': results_llm,
        'results_ner': results_ner,
        'comparison': comparison_results
    }


def main():
    parser = argparse.ArgumentParser(
        description="Moving Targets Pipeline - End-to-End Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --mode sample                  # Quick single-company test
  python run.py --mode full --method base      # Full analysis with NER
  python run.py --mode full --method llm       # Full analysis with LLM
  python run.py --mode demo                    # MT score demo only
  python run.py --mode compare                 # LLM vs NER comparison
  python run.py --mode full --cache output/mt_scores.csv  # Use cached scores
        """
    )
    parser.add_argument(
        "--mode",
        choices=["sample", "full", "demo", "compare"],
        default="sample",
        help="Analysis mode (default: sample)"
    )
    parser.add_argument(
        "--method",
        choices=["base", "llm"],
        default="base",
        help="Target extraction method: 'base' (spaCy NER) or 'llm' (GPT/Gemini) (default: base)"
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Path to cached MT score CSV file"
    )
    parser.add_argument(
        "--llm-cache",
        type=str,
        default=None,
        help="LLM cache path for compare mode"
    )
    parser.add_argument(
        "--ner-cache",
        type=str,
        default=None,
        help="NER cache path for compare mode"
    )

    args = parser.parse_args()

    print("Moving Targets Pipeline")
    print(f"Mode: {args.mode}, Method: {args.method}")
    print()

    if args.mode == "sample":
        run_sample_analysis(
            extraction_method=args.method,
            cache_file_path=args.cache
        )
    elif args.mode == "full":
        run_full_analysis(
            extraction_method=args.method,
            cache_file_path=args.cache
        )
    elif args.mode == "demo":
        demo_mt_score_calculation()
    elif args.mode == "compare":
        run_llm_vs_ner_comparison(
            llm_cache_path=args.llm_cache or "output/snp100_llm/mt_scores.csv",
            ner_cache_path=args.ner_cache or "output/snp100_base/mt_scores.csv"
        )


if __name__ == "__main__":
    main()
