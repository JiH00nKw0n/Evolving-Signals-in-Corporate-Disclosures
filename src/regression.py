from dataclasses import dataclass
from typing import Dict, List, TypedDict, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


class RegressionResult(TypedDict):
    coefficients: Dict[str, float]
    t_stats: Dict[str, float]
    p_values: Dict[str, float]
    std_errors: Dict[str, float]
    r_squared: float
    adj_r_squared: float
    n_obs: int


@dataclass
class FamaMacBethResult:
    variable_names: List[str]
    avg_coefficients: Dict[str, float]
    t_stats: Dict[str, float]
    p_values: Dict[str, float]
    std_errors: Dict[str, float]
    n_periods: int
    avg_r_squared: float
    avg_n_obs: float

    def to_dict(self) -> Dict[str, float]:
        result = {}
        for var in self.variable_names:
            result[f'{var}_coef'] = self.avg_coefficients[var]
            result[f'{var}_tstat'] = self.t_stats[var]
            result[f'{var}_pvalue'] = self.p_values[var]
        result['n_periods'] = self.n_periods
        result['r_squared'] = self.avg_r_squared
        result['n_obs'] = self.avg_n_obs
        return result


def basic_regression(data: pd.DataFrame, y_col: str, x_cols: List[str]) -> RegressionResult:
    """Run basic OLS regression using statsmodels"""
    y = data[y_col].dropna()
    X = data[x_cols].loc[y.index]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    return RegressionResult(
        coefficients=model.params.to_dict(),
        t_stats=model.tvalues.to_dict(),
        p_values=model.pvalues.to_dict(),
        std_errors=model.bse.to_dict(),
        r_squared=float(model.rsquared),
        adj_r_squared=float(model.rsquared_adj),
        n_obs=int(model.nobs)
    )


def fama_macbeth_regression(data: pd.DataFrame, y_col: str, x_cols: List[str],
                            time_col: str) -> FamaMacBethResult:
    """Run Fama-MacBeth monthly cross-sectional regressions"""
    time_periods = sorted(data[time_col].unique())
    monthly_results = []

    for period in time_periods:
        period_data = data[data[time_col] == period].copy()

        if len(period_data) < max(len(x_cols) + 2, 5):  # Need at least 5 observations or variables+2
            continue

        try:
            result = basic_regression(period_data, y_col, x_cols)
            monthly_results.append(result)
        except Exception:
            continue

    if not monthly_results:
        raise ValueError("No valid monthly regressions")

    # Convert to DataFrame for easier aggregation
    coef_df = pd.DataFrame([r['coefficients'] for r in monthly_results])
    r_squared_list = [r['r_squared'] for r in monthly_results]
    n_obs_list = [r['n_obs'] for r in monthly_results]

    variable_names = list(coef_df.columns)
    avg_coefficients = coef_df.mean().to_dict()
    std_coefficients = coef_df.std().to_dict()
    n_periods = len(monthly_results)
    avg_r_squared = float(np.mean(r_squared_list))
    avg_n_obs = float(np.mean(n_obs_list))

    # Fama-MacBeth t-statistics
    t_stats = {}
    p_values = {}
    std_errors = {}

    for var in variable_names:
        se = std_coefficients[var] / np.sqrt(n_periods)
        t_stat = avg_coefficients[var] / se
        # 단측검정: MT Score는 음의 방향 가설 (MT↑ → Return↓)
        p_val = 1 - stats.t.cdf(abs(t_stat), n_periods - 1)

        t_stats[var] = t_stat
        p_values[var] = p_val
        std_errors[var] = se

    return FamaMacBethResult(
        variable_names=variable_names,
        avg_coefficients=avg_coefficients,
        t_stats=t_stats,
        p_values=p_values,
        std_errors=std_errors,
        n_periods=n_periods,
        avg_r_squared=avg_r_squared,
        avg_n_obs=avg_n_obs
    )


def create_results_table(results: Dict[str, float], title: str = "Regression Results") -> str:
    """Create markdown table from results"""
    variables = set()
    for key in results.keys():
        if key.endswith('_coef'):
            variables.add(key.replace('_coef', ''))

    variables = sorted(list(variables))

    table = f"## {title}\n\n"
    table += "| Variable | Coefficient | t-statistic | p-value |\n"
    table += "|----------|-------------|-------------|---------|\n"

    for var in variables:
        coef = results.get(f"{var}_coef", 0.0)
        tstat = results.get(f"{var}_tstat", 0.0)
        pvalue = results.get(f"{var}_pvalue", 1.0)

        # Add significance stars
        coef_str = f"{coef:.4f}"
        if pvalue < 0.01:
            coef_str += "***"
        elif pvalue < 0.05:
            coef_str += "**"
        elif pvalue < 0.10:
            coef_str += "*"

        table += f"| {var} | {coef_str} | {tstat:.3f} | {pvalue:.4f} |\n"

    # Add summary statistics
    if 'n_periods' in results:
        table += f"\nPeriods: {results['n_periods']:.0f}\n"
    if 'r_squared' in results:
        table += f"R-squared: {results['r_squared']:.4f}\n"
    if 'n_obs' in results:
        table += f"Avg observations per period: {results['n_obs']:.0f}\n"

    table += "\n*p<0.10, **p<0.05, ***p<0.01\n"
    table += "t-statistics in parentheses; p-values are one-sided based on |t|\n"
    return table


@dataclass
class RegressionAnalyzer:
    """Main class for regression analysis with monthly averaging"""
    results_history: Dict[str, any] = None

    def __post_init__(self):
        if self.results_history is None:
            self.results_history = {}

    def run_basic_regression(self, data: pd.DataFrame, y_col: str, x_cols: List[str],
                             name: str = "basic") -> RegressionResult:
        """Run basic OLS regression"""
        result = basic_regression(data, y_col, x_cols)
        self.results_history[name] = result
        return result

    def run_fama_macbeth(self, data: pd.DataFrame, y_col: str, x_cols: List[str],
                         time_col: str, name: str = "fama_macbeth") -> FamaMacBethResult:
        """Run Fama-MacBeth regression"""
        result = fama_macbeth_regression(data, y_col, x_cols, time_col)
        self.results_history[name] = result
        return result

    def get_results_table(self, name: str, title: str = None) -> str:
        """Generate markdown table for stored results"""
        if name not in self.results_history:
            raise ValueError(f"Results '{name}' not found")

        result = self.results_history[name]
        if title is None:
            title = f"Results: {name}"

        if isinstance(result, FamaMacBethResult):
            return create_results_table(result.to_dict(), title)
        elif isinstance(result, dict):
            # Convert basic regression result to table format
            table_dict = {}
            for var, coef in result['coefficients'].items():
                table_dict[f'{var}_coef'] = coef
                table_dict[f'{var}_tstat'] = result['t_stats'][var]
                table_dict[f'{var}_pvalue'] = result['p_values'][var]
            table_dict['r_squared'] = result['r_squared']
            table_dict['n_obs'] = result['n_obs']
            return create_results_table(table_dict, title)
        else:
            raise ValueError(f"Unknown result type for '{name}'")

    def average_monthly_results(self, monthly_results: List[Dict],
                                title: str = "Monthly Averages") -> str:
        """Average results across months and create table"""
        if not monthly_results:
            raise ValueError("No monthly results provided")

        # Get all unique keys
        all_keys = set()
        for result in monthly_results:
            all_keys.update(result.keys())

        # Calculate averages
        averaged = {}
        for key in all_keys:
            values = [r.get(key, np.nan) for r in monthly_results]
            values = [v for v in values if not np.isnan(v)]
            if values:
                averaged[key] = np.mean(values)

        return create_results_table(averaged, title)


# ========================================
# Moving Targets Score Integration Functions
# ========================================

def merge_mt_scores_with_returns(mt_score_df: pd.DataFrame,
                                 returns_df: pd.DataFrame,
                                 lag_months: int = 1) -> pd.DataFrame:
    """
    Merge MT scores with stock returns data for regression analysis.
    Uses lagged control variables to avoid endogeneity issues.

    Args:
        mt_score_df: DataFrame with MT scores (from mt_score.py)
        returns_df: DataFrame with stock returns (TRAIN_DATA.csv format)
        lag_months: Number of months to lag MT scores (default: 1 month)

    Returns:
        Merged DataFrame ready for regression analysis
    """

    # Prepare returns data
    returns_df = returns_df.copy()
    returns_df['DATE'] = pd.to_datetime(returns_df['DATE'])
    returns_df = returns_df.rename(columns={'ISIN': 'isin', 'RETURNS': 'returns'})

    # Sort returns data by ISIN and DATE for lagging
    returns_df = returns_df.sort_values(['isin', 'DATE'])

    # Use control variables from same period as MT scores
    control_vars = ['RET_1', 'RET_12', 'SIZE', 'LOG_BM']
    available_controls = [col for col in control_vars if col in returns_df.columns]

    if available_controls:
        print(f"Using same-period control variables: {available_controls}")

    # FMP data already includes RET_1 and RET_12, no need to calculate lags

    # Prepare MT score data
    mt_score_df = mt_score_df.copy()
    mt_score_df['date'] = pd.to_datetime(mt_score_df['date'])

    # Prepare control variables from t-period (same period as MT scores)
    # We need control variables from the same period as MT scores, not returns period
    control_vars_t = returns_df[['isin', 'DATE'] + available_controls].copy()
    control_vars_t['year_month_control'] = control_vars_t['DATE'].dt.to_period('M')

    # Add lagged date for MT scores (lag by specified months)
    mt_score_df['date_lagged'] = mt_score_df['date'] + pd.DateOffset(months=lag_months)
    mt_score_df['year_month_mt'] = mt_score_df['date'].dt.to_period('M')

    # First merge: MT scores with t-period control variables
    mt_with_controls = pd.merge(
        mt_score_df[['isin', 'date', 'date_lagged', 'year_month_mt', 'mt_score', 'quarter', 'num_total_targets']],
        control_vars_t[['isin', 'year_month_control'] + available_controls],
        left_on=['isin', 'year_month_mt'],
        right_on=['isin', 'year_month_control'],
        how='inner'
    )

    # Second merge: Add t+1 period returns
    merged_df = pd.merge(
        returns_df[['isin', 'DATE', 'returns']],
        mt_with_controls,
        left_on=['isin', returns_df['DATE'].dt.to_period('M')],
        right_on=['isin', mt_with_controls['date_lagged'].dt.to_period('M')],
        how='inner'
    )

    # Clean up and rename columns
    merged_df = merged_df.drop(['date_lagged'], axis=1)
    merged_df['year_month'] = merged_df['DATE'].dt.to_period('M')

    # Log market cap (SIZE is already log transformed in the data)
    if 'SIZE' in merged_df.columns:
        merged_df['log_size'] = merged_df['SIZE']

    # Book-to-market ratio (LOG_BM column)
    if 'LOG_BM' in merged_df.columns:
        merged_df['log_bm'] = merged_df['LOG_BM']

    # Previous month return (RET_1 column)
    if 'RET_1' in merged_df.columns:
        merged_df['ret_lag1'] = merged_df['RET_1']

    # Long-term momentum (RET_12 column)
    if 'RET_12' in merged_df.columns:
        merged_df['ret_lag12_2'] = merged_df['RET_12']

    print(
        f"Merged {len(merged_df)} observations from {len(mt_score_df)} MT scores and {len(returns_df)} return observations"
    )

    # Print available control variables
    control_cols = [col for col in merged_df.columns if col in ['log_size', 'log_bm', 'ret_lag1', 'ret_lag12_2']]
    if control_cols:
        print(f"Available control variables: {control_cols}")

    return merged_df


def run_mt_score_fama_macbeth(mt_score_df: pd.DataFrame,
                              returns_df: pd.DataFrame,
                              control_variables: Optional[List[str]] = None,
                              lag_months: int = 1) -> FamaMacBethResult:
    """
    Run Fama-MacBeth regression with MT scores predicting stock returns.

    Args:
        mt_score_df: DataFrame with MT scores
        returns_df: DataFrame with stock returns and control variables
        control_variables: List of control variable column names from returns_df
        lag_months: Number of months to lag MT scores

    Returns:
        FamaMacBethResult with regression results
    """

    # Merge MT scores with returns
    merged_df = merge_mt_scores_with_returns(mt_score_df, returns_df, lag_months)

    # Remove observations with missing MT scores
    merged_df = merged_df.dropna(subset=['mt_score'])

    if len(merged_df) == 0:
        raise ValueError("No observations with valid MT scores after merging")

    # Prepare independent variables
    x_cols = ['mt_score']

    # Add control variables if specified
    if control_variables:
        available_controls = [col for col in control_variables if col in merged_df.columns]
        if available_controls:
            x_cols.extend(available_controls)
            print(f"Including control variables: {available_controls}")
        else:
            print("Warning: No specified control variables found in merged data")

    # Run Fama-MacBeth regression
    print(f"Running Fama-MacBeth regression with {len(merged_df)} observations")
    print(f"Independent variables: {x_cols}")

    result = fama_macbeth_regression(
        data=merged_df,
        y_col='returns',
        x_cols=x_cols,
        time_col='year_month'
    )

    return result


def run_mt_score_analysis(mt_score_df: pd.DataFrame,
                          returns_df: pd.DataFrame,
                          control_variables: Optional[List[str]] = None,
                          lag_months: int = 1) -> Dict[str, any]:
    """
    Complete MT score regression analysis with summary statistics.

    Args:
        mt_score_df: DataFrame with MT scores
        returns_df: DataFrame with stock returns and control variables
        control_variables: List of control variable column names
        lag_months: Number of months to lag MT scores

    Returns:
        Dictionary with regression results and summary statistics
    """

    print("=== Moving Targets Score Regression Analysis ===")

    # Run Fama-MacBeth regression
    try:
        fama_macbeth_result = run_mt_score_fama_macbeth(
            mt_score_df, returns_df, control_variables, lag_months
        )

        # Create results table
        results_table = create_results_table(
            fama_macbeth_result.to_dict(),
            "Moving Targets Score - Fama-MacBeth Regression"
        )

        print("Fama-MacBeth regression completed successfully")

    except Exception as e:
        print(f"Error in Fama-MacBeth regression: {e}")
        fama_macbeth_result = None
        results_table = "Regression analysis failed"

    # Calculate summary statistics
    merged_df = merge_mt_scores_with_returns(mt_score_df, returns_df, lag_months)
    merged_df = merged_df.dropna(subset=['mt_score'])

    summary_stats = {
        'total_observations': len(merged_df),
        'unique_companies': merged_df['isin'].nunique(),
        'unique_months': merged_df['year_month'].nunique(),
        'mt_score_mean': merged_df['mt_score'].mean(),
        'mt_score_std': merged_df['mt_score'].std(),
        'returns_mean': merged_df['returns'].mean(),
        'returns_std': merged_df['returns'].std(),
        'mt_returns_correlation': merged_df['mt_score'].corr(merged_df['returns'])
    }

    return {
        'fama_macbeth_result': fama_macbeth_result,
        'results_table': results_table,
        'summary_stats': summary_stats,
        'merged_data': merged_df
    }


def run_multi_specification_analysis(mt_score_df: pd.DataFrame,
                                     returns_df: pd.DataFrame,
                                     lag_months: int = 1) -> Dict[str, any]:
    """
    Run multiple regression specifications similar to the paper's Table IV.

    Args:
        mt_score_df: DataFrame with MT scores
        returns_df: DataFrame with stock returns and control variables
        lag_months: Number of months to lag MT scores

    Returns:
        Dictionary with multiple specification results
    """

    print("=== Multi-Specification Moving Targets Analysis ===")

    # Merge data once
    merged_df = merge_mt_scores_with_returns(mt_score_df, returns_df, lag_months)
    merged_df = merged_df.dropna(subset=['mt_score'])

    if len(merged_df) == 0:
        raise ValueError("No observations with valid MT scores after merging")

    # Define different specifications (similar to paper's Table IV)
    specifications = {
        '(1) MT Score Only': ['mt_score'],
        '(2) + Size & BM': ['mt_score', 'log_size', 'log_bm'],
        '(3) + Returns': ['mt_score', 'log_size', 'log_bm', 'ret_lag1', 'ret_lag12_2'],
        '(4) + All Controls': ['mt_score', 'log_size', 'log_bm', 'ret_lag1', 'ret_lag12_2']
    }

    results = {}
    all_results_data = []

    for spec_name, x_vars in specifications.items():
        print(f"\nRunning specification: {spec_name}")

        # Check which variables are available
        available_vars = [var for var in x_vars if var in merged_df.columns]
        if len(available_vars) != len(x_vars):
            missing_vars = set(x_vars) - set(available_vars)
            print(f"  Warning: Missing variables {missing_vars}, using available: {available_vars}")

        if not available_vars:
            print(f"  Skipping {spec_name} - no variables available")
            continue

        try:
            # Run Fama-MacBeth regression
            result = fama_macbeth_regression(
                data=merged_df,
                y_col='returns',
                x_cols=available_vars,
                time_col='year_month'
            )

            results[spec_name] = result

            # Convert to format for combined table
            result_dict = result.to_dict()
            result_dict['specification'] = spec_name
            all_results_data.append(result_dict)

            print(f"  ✅ {spec_name} completed - {result.n_periods} periods, avg {result.avg_n_obs:.0f} obs/period")

        except Exception as e:
            print(f"  ❌ {spec_name} failed: {e}")
            continue

    # Create combined results table (similar to paper's format)
    combined_table = create_combined_specification_table(all_results_data)

    # Calculate summary statistics
    summary_stats = {
        'total_observations': len(merged_df),
        'unique_companies': merged_df['isin'].nunique(),
        'unique_months': merged_df['year_month'].nunique(),
        'specifications_run': len(results),
        'mt_score_mean': merged_df['mt_score'].mean(),
        'mt_score_std': merged_df['mt_score'].std(),
        'returns_mean': merged_df['returns'].mean(),
        'returns_std': merged_df['returns'].std(),
    }

    return {
        'specifications': results,
        'combined_table': combined_table,
        'summary_stats': summary_stats,
        'merged_data': merged_df,
        'individual_tables': {name: create_results_table(result.to_dict(), name)
                              for name, result in results.items()}
    }


def create_combined_specification_table(all_results: List[Dict]) -> str:
    """
    Create a combined table showing all specifications side by side.
    Similar to the paper's Table IV format.
    """

    if not all_results:
        return "No results to display"

    # Get all unique variables across specifications
    all_variables = set()
    for result in all_results:
        for key in result.keys():
            if key.endswith('_coef'):
                all_variables.add(key.replace('_coef', ''))

    # Remove meta variables
    all_variables = [v for v in sorted(all_variables) if v not in ['specification']]

    # Create table header
    table = "## Fama-MacBeth Regression Results - Multiple Specifications\n\n"
    table += "| Variable |"
    for result in all_results:
        spec_name = result.get('specification', 'Unknown')
        # Shorten specification names for table
        short_name = spec_name.split(')')[0] + ')'
        table += f" {short_name} |"
    table += "\n"

    table += "|----------|"
    for _ in all_results:
        table += "-------------|"
    table += "\n"

    # Add variable rows
    for var in all_variables:
        if var == 'const':
            display_var = 'Constant'
        else:
            display_var = var.replace('_', ' ').title()

        table += f"| {display_var} |"

        for result in all_results:
            coef = result.get(f"{var}_coef", None)
            tstat = result.get(f"{var}_tstat", None)
            pvalue = result.get(f"{var}_pvalue", None)

            if coef is not None and tstat is not None and pvalue is not None:
                # Format coefficient with significance stars
                coef_str = f"{coef:.4f}"
                if pvalue < 0.01:
                    coef_str += "***"
                elif pvalue < 0.05:
                    coef_str += "**"
                elif pvalue < 0.10:
                    coef_str += "*"

                table += f" {coef_str}<br>({tstat:.2f}) |"
            else:
                table += " - |"

        table += "\n"

    # Add summary statistics
    table += "\n"
    table += "| **Summary Statistics** |"
    for _ in all_results:
        table += " |"
    table += "\n"

    table += "|----------|"
    for _ in all_results:
        table += "-------------|"
    table += "\n"

    # R-squared row
    table += "| R-squared |"
    for result in all_results:
        r_sq = result.get('r_squared', None)
        if r_sq is not None:
            table += f" {r_sq:.4f} |"
        else:
            table += " - |"
    table += "\n"

    # N periods row
    table += "| Periods |"
    for result in all_results:
        n_periods = result.get('n_periods', None)
        if n_periods is not None:
            table += f" {n_periods:.0f} |"
        else:
            table += " - |"
    table += "\n"

    # Average observations row
    table += "| Avg Obs/Period |"
    for result in all_results:
        n_obs = result.get('n_obs', None)
        if n_obs is not None:
            table += f" {n_obs:.0f} |"
        else:
            table += " - |"
    table += "\n"

    table += "\n*p<0.10, **p<0.05, ***p<0.01\n"
    table += "t-statistics in parentheses; p-values are one-sided based on |t|\n"

    return table
