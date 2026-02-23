"""
Calendar-Time Portfolio Backtest for Moving Targets Analysis

This module implements the portfolio construction and backtesting methodology 
described in Cohen & Nguyen (2024) "Moving Targets" Table 2.

The backtest follows these key principles:
1. Monthly rebalancing based on moving targets scores from the previous month
2. 3-month holding periods with overlapping portfolios 
3. Quintile portfolios based on moving targets distribution
4. Both equally-weighted and value-weighted options
5. Risk-adjusted returns using Fama-French factors
"""

import warnings
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

from src.regression import RegressionAnalyzer


def calculate_sharpe_ratio(returns: pd.Series, rf: pd.Series = None, annualize: bool = True) -> dict:
    """
    Calculate Sharpe ratio with statistical significance test (Lo 2002).

    Args:
        returns: Series of portfolio returns
        rf: Series of risk-free rates (if None, assumes excess returns)
        annualize: If True, annualize the Sharpe ratio (multiply by sqrt(12) for monthly)

    Returns:
        Dictionary with sharpe_ratio, se, t_stat, p_value, and component details
    """
    if rf is not None:
        aligned = pd.concat([returns, rf], axis=1).dropna()
        if len(aligned) > 0:
            returns_aligned = aligned.iloc[:, 0]
            rf_aligned = aligned.iloc[:, 1]
            excess_returns = returns_aligned - rf_aligned
            mean_rf = rf_aligned.mean()
        else:
            excess_returns = returns.dropna()
            mean_rf = np.nan
    else:
        excess_returns = returns.dropna()
        mean_rf = 0.0

    T = len(excess_returns)

    if T < 2:
        return {
            'sharpe_ratio': np.nan, 'se': np.nan, 't_stat': np.nan, 'p_value': np.nan, 'n_obs': T,
            'mean_return': np.nan, 'std_return': np.nan, 'mean_rf': np.nan, 'mean_excess_return': np.nan
        }

    mean_ret = excess_returns.mean()
    std_ret = excess_returns.std(ddof=1)

    # Calculate raw return mean (before subtracting rf)
    if rf is not None and len(aligned) > 0:
        raw_mean_ret = returns_aligned.mean()
    else:
        raw_mean_ret = returns.dropna().mean()

    if std_ret == 0:
        return {
            'sharpe_ratio': np.nan, 'se': np.nan, 't_stat': np.nan, 'p_value': np.nan, 'n_obs': T,
            'mean_return': raw_mean_ret, 'std_return': std_ret, 'mean_rf': mean_rf, 'mean_excess_return': mean_ret
        }

    # Monthly Sharpe ratio
    sr = mean_ret / std_ret

    # Lo (2002) standard error for i.i.d. returns
    se_sr = np.sqrt((1 + 0.5 * sr**2) / T)

    # t-statistic and p-value
    t_stat = sr / se_sr
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=T-1))

    # Annualize if requested
    if annualize:
        sr_annualized = sr * np.sqrt(12)
        se_annualized = se_sr * np.sqrt(12)
        # Annualize return and std as well
        mean_ret_ann = mean_ret * 12
        std_ret_ann = std_ret * np.sqrt(12)
        raw_mean_ret_ann = raw_mean_ret * 12
        mean_rf_ann = mean_rf * 12 if not np.isnan(mean_rf) else np.nan
    else:
        sr_annualized = sr
        se_annualized = se_sr
        mean_ret_ann = mean_ret
        std_ret_ann = std_ret
        raw_mean_ret_ann = raw_mean_ret
        mean_rf_ann = mean_rf

    return {
        'sharpe_ratio': sr_annualized,
        'sharpe_ratio_monthly': sr,
        'se': se_annualized,
        't_stat': t_stat,
        'p_value': p_value,
        'n_obs': T,
        # Component details (monthly)
        'mean_return_monthly': raw_mean_ret,
        'mean_excess_return_monthly': mean_ret,
        'std_return_monthly': std_ret,
        'mean_rf_monthly': mean_rf,
        # Component details (annualized)
        'mean_return_ann': raw_mean_ret_ann,
        'mean_excess_return_ann': mean_ret_ann,
        'std_return_ann': std_ret_ann,
        'mean_rf_ann': mean_rf_ann
    }


def compare_sharpe_ratios_paired(returns_a: pd.Series, returns_b: pd.Series,
                                  rf: pd.Series = None, method: str = 'ledoit_wolf') -> dict:
    """
    Compare two Sharpe ratios from paired samples (Ledoit-Wolf 2008).

    This is appropriate for comparing LLM vs NER methods on the same data.

    Args:
        returns_a: Series of returns from method A (e.g., LLM)
        returns_b: Series of returns from method B (e.g., NER)
        rf: Series of risk-free rates (if None, assumes excess returns)
        method: 'ledoit_wolf' or 'paired_bootstrap'

    Returns:
        Dictionary with difference, se, z_stat, p_value, and individual Sharpe ratios
    """
    # Align the series
    aligned = pd.concat([returns_a, returns_b], axis=1).dropna()
    if rf is not None:
        aligned_rf = rf.reindex(aligned.index).fillna(method='ffill')
        ret_a = aligned.iloc[:, 0] - aligned_rf
        ret_b = aligned.iloc[:, 1] - aligned_rf
    else:
        ret_a = aligned.iloc[:, 0]
        ret_b = aligned.iloc[:, 1]

    T = len(ret_a)
    if T < 10:
        return {'error': 'Insufficient observations for comparison'}

    # Calculate individual Sharpe ratios
    mu_a, mu_b = ret_a.mean(), ret_b.mean()
    sigma_a, sigma_b = ret_a.std(ddof=1), ret_b.std(ddof=1)

    if sigma_a == 0 or sigma_b == 0:
        return {'error': 'Zero standard deviation'}

    sr_a = mu_a / sigma_a
    sr_b = mu_b / sigma_b
    sr_diff = sr_a - sr_b

    if method == 'ledoit_wolf':
        # Ledoit-Wolf (2008) method for paired Sharpe ratio comparison
        # Compute covariance and correlation
        cov_ab = np.cov(ret_a, ret_b)[0, 1]
        rho = cov_ab / (sigma_a * sigma_b)

        # Variance of the difference (Ledoit-Wolf 2008, Equation 10)
        var_sr_diff = (1/T) * (
            2 - 2*rho + 0.5*(sr_a**2 + sr_b**2 - 2*sr_a*sr_b*rho)
        )
        se_diff = np.sqrt(var_sr_diff)

        # z-statistic (asymptotically normal for large T)
        z_stat = sr_diff / se_diff if se_diff > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    elif method == 'paired_bootstrap':
        # Bootstrap method for robustness
        n_bootstrap = 10000
        bootstrap_diffs = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(T, T, replace=True)
            boot_a = ret_a.iloc[idx]
            boot_b = ret_b.iloc[idx]

            boot_sr_a = boot_a.mean() / boot_a.std(ddof=1) if boot_a.std(ddof=1) > 0 else 0
            boot_sr_b = boot_b.mean() / boot_b.std(ddof=1) if boot_b.std(ddof=1) > 0 else 0
            bootstrap_diffs.append(boot_sr_a - boot_sr_b)

        se_diff = np.std(bootstrap_diffs)
        z_stat = sr_diff / se_diff if se_diff > 0 else 0
        # Two-tailed p-value from bootstrap distribution
        p_value = 2 * min(
            np.mean(np.array(bootstrap_diffs) <= 0),
            np.mean(np.array(bootstrap_diffs) >= 0)
        )
    else:
        return {'error': f'Unknown method: {method}'}

    # Annualized values
    sr_a_ann = sr_a * np.sqrt(12)
    sr_b_ann = sr_b * np.sqrt(12)
    sr_diff_ann = sr_diff * np.sqrt(12)
    se_diff_ann = se_diff * np.sqrt(12)

    return {
        'sharpe_a': sr_a_ann,
        'sharpe_b': sr_b_ann,
        'sharpe_diff': sr_diff_ann,
        'sharpe_a_monthly': sr_a,
        'sharpe_b_monthly': sr_b,
        'sharpe_diff_monthly': sr_diff,
        'se_diff': se_diff_ann,
        'z_stat': z_stat,
        'p_value': p_value,
        'n_obs': T,
        'correlation': rho if method == 'ledoit_wolf' else np.nan,
        'method': method
    }


class MovingTargetsBacktest:
    """
    Main backtesting class for Moving Targets portfolio strategy.
    
    Implements overlapping 3-month holding period portfolios with monthly rebalancing,
    similar to the methodology in Cohen & Nguyen (2024).
    """

    def __init__(self, is_value_weighted: bool = False):
        """
        Initialize the backtest.
        
        Args:
            is_value_weighted: If True, use value-weighted portfolios. 
                             If False, use equally-weighted portfolios.
        """
        self.is_value_weighted = is_value_weighted
        self.portfolio_returns = {}
        self.factor_3_data = None
        self.factor_5_data = None
        self.regression_analyzer = RegressionAnalyzer()

    def load_factor_data(self,
                         factor_3_path: str = "data/F-F_Research_Data_Factors.tsv",
                         factor_5_path: str = "data/F-F_Research_Data_5_Factors_2x3.tsv") -> None:
        """Load Fama-French factor data separately."""

        # Load 3-factor data
        self.factor_3_data = pd.read_csv(factor_3_path, sep='\t')
        self.factor_3_data['Date'] = pd.to_datetime(self.factor_3_data['Date'].astype(str), format='%Y%m')
        # Convert to month end to match portfolio returns
        self.factor_3_data['Date'] = self.factor_3_data['Date'] + pd.offsets.MonthEnd(0)
        self.factor_3_data.set_index('Date', inplace=True)

        # Convert percentage to decimal
        factor_3_cols = ['Mkt-RF', 'SMB', 'HML', 'RF']
        self.factor_3_data[factor_3_cols] = self.factor_3_data[factor_3_cols] / 100

        # Load 5-factor data 
        self.factor_5_data = pd.read_csv(factor_5_path, sep='\t')
        self.factor_5_data['Date'] = pd.to_datetime(self.factor_5_data['Date'].astype(str), format='%Y%m')
        # Convert to month end to match portfolio returns
        self.factor_5_data['Date'] = self.factor_5_data['Date'] + pd.offsets.MonthEnd(0)
        self.factor_5_data.set_index('Date', inplace=True)

        # Convert percentage to decimal
        factor_5_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        self.factor_5_data[factor_5_cols] = self.factor_5_data[factor_5_cols] / 100

        print(f"Loaded 3-factor data: {self.factor_3_data.index[0]} to {self.factor_3_data.index[-1]}")
        print(f"Loaded 5-factor data: {self.factor_5_data.index[0]} to {self.factor_5_data.index[-1]}")

    def create_quintile_portfolios(self,
                                   mt_score_df: pd.DataFrame,
                                   returns_df: pd.DataFrame,
                                   score_col: str = 'mt_score') -> pd.DataFrame:
        """
        Create quintile portfolios based on moving targets scores.
        
        Implements overlapping 3-month holding periods:
        - Each month, form new quintile portfolios based on prior month scores
        - Hold each portfolio for exactly 3 months
        - At any given month, up to 3 portfolios may be active
        
        Args:
            mt_score_df: DataFrame with columns ['date', 'isin', 'mt_score'] (from mt_score.py)
            returns_df: DataFrame with returns data (from TRAIN_DATA.csv format)
            score_col: Column name to use for quintile formation (default: 'mt_score')

        Returns:
            DataFrame with overlapping portfolio holdings by quintile and month
        """

        # Prepare returns data
        returns_df = returns_df.copy()
        returns_df['DATE'] = pd.to_datetime(returns_df['DATE'])
        returns_df = returns_df.dropna(subset=['RETURNS'])

        # Prepare moving targets data
        mt_score_df = mt_score_df.copy()
        mt_score_df['date'] = pd.to_datetime(mt_score_df['date'])

        # Remove observations with missing scores
        mt_score_df = mt_score_df.dropna(subset=[score_col])

        if len(mt_score_df) == 0:
            print(f"Warning: No valid {score_col} data found")
            return pd.DataFrame()

        # Get monthly returns data
        returns_monthly = self._prepare_monthly_returns(returns_df)

        # Get all unique months for portfolio formation
        formation_months = sorted(mt_score_df['date'].dt.to_period('M').unique())

        # Store portfolio compositions for overlapping holding periods
        portfolio_compositions = {}

        # Create portfolio compositions for each formation month
        for month in formation_months:
            month_start = month.start_time
            month_end = month.end_time

            # Get scores for this formation month
            month_scores = mt_score_df[
                (mt_score_df['date'] >= month_start) &
                (mt_score_df['date'] <= month_end)
                ].copy()

            if len(month_scores) == 0:
                continue

            # Create quintiles based on specified score column
            try:
                month_scores['quintile'] = pd.qcut(
                    month_scores[score_col],
                    q=5,
                    labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                    duplicates='drop'
                )

                # Store portfolio composition (ISIN lists for each quintile)
                portfolio_compositions[month] = {}
                for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
                    q_isins = month_scores[month_scores['quintile'] == q]['isin'].tolist()
                    portfolio_compositions[month][q] = q_isins

            except ValueError:
                # Skip if quintile creation fails (e.g., too few unique values)
                continue

        # Calculate overlapping portfolio returns
        all_return_months = sorted(returns_monthly.index)
        overlapping_returns = []

        for return_month in all_return_months:
            month_returns = returns_monthly.loc[return_month]
            month_returns = month_returns.dropna()

            if len(month_returns) == 0:
                continue

            # Find all active portfolios for this return month
            # (portfolios formed in the last 3 months)
            active_portfolios = []

            # Convert return_month to timestamp for comparison
            return_timestamp = pd.Timestamp(return_month.start_time) if hasattr(
                return_month, 'start_time'
                ) else pd.Timestamp(return_month)

            for formation_month in portfolio_compositions.keys():
                # Check if this portfolio is still active (within 3-month holding period)
                formation_timestamp = pd.Timestamp(formation_month.start_time)
                months_since_formation = (return_timestamp - formation_timestamp).days // 30
                if 0 < months_since_formation <= 3:
                    active_portfolios.append(formation_month)

            if len(active_portfolios) == 0:
                continue

            # Calculate returns for each quintile across all active portfolios
            quintile_returns = {}

            for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
                all_returns_for_q = []

                # Collect returns from all active portfolios for this quintile
                for formation_month in active_portfolios:
                    q_isins = portfolio_compositions[formation_month][q]
                    q_returns = month_returns[month_returns.index.isin(q_isins)]

                    if len(q_returns) > 0:
                        if self.is_value_weighted:
                            # Value-weighted returns (would need market cap data)
                            # For now, use equally-weighted as proxy
                            all_returns_for_q.extend(q_returns.tolist())
                        else:
                            # Equally-weighted returns
                            all_returns_for_q.extend(q_returns.tolist())

                # Calculate average return for this quintile
                if len(all_returns_for_q) > 0:
                    quintile_returns[q] = np.mean(all_returns_for_q)
                else:
                    quintile_returns[q] = np.nan

            # Add to results
            result_row = {
                'return_month': return_month,
                'num_active_portfolios': len(active_portfolios),
                **quintile_returns
            }
            overlapping_returns.append(result_row)

        return pd.DataFrame(overlapping_returns)

    def _prepare_monthly_returns(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Convert daily returns to monthly returns by ISIN."""

        returns_df = returns_df.copy()
        returns_df['year_month'] = returns_df['DATE'].dt.to_period('M')

        # Group by ISIN and month, take the last return observation per month
        monthly_returns = returns_df.groupby(['ISIN', 'year_month']).agg(
            {
                'RETURNS': 'last'  # Use last available return in the month
            }
        ).reset_index()

        # Pivot to have ISINs as index and months as columns
        monthly_pivot = monthly_returns.pivot(
            index='ISIN',
            columns='year_month',
            values='RETURNS'
        )

        # Transpose to have months as index and ISINs as columns
        return monthly_pivot.T

    def calculate_portfolio_returns(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process overlapping portfolio returns.
        
        Args:
            portfolio_df: Output from create_quintile_portfolios (already contains overlapping returns)
            
        Returns:
            DataFrame with monthly portfolio returns by quintile
        """

        # The portfolio_df already contains properly calculated overlapping returns
        # Just need to clean up and add long-short portfolio
        monthly_returns = portfolio_df.copy()

        # Convert return_month to datetime if it's not already
        if not isinstance(monthly_returns.index, pd.DatetimeIndex):
            if 'return_month' in monthly_returns.columns:
                monthly_returns.set_index('return_month', inplace=True)

        # Convert PeriodIndex to DatetimeIndex (month end) to match Fama-French data
        if isinstance(monthly_returns.index, pd.PeriodIndex):
            monthly_returns.index = monthly_returns.index.to_timestamp('M')

        # Calculate long-short portfolio (Q5 - Q1)
        monthly_returns['Q5-Q1'] = monthly_returns['Q5'] - monthly_returns['Q1']

        # Drop unnecessary columns
        columns_to_drop = [col for col in ['num_active_portfolios'] if col in monthly_returns.columns]
        monthly_returns = monthly_returns.drop(columns=columns_to_drop)

        return monthly_returns

    def calculate_risk_adjusted_returns(self, portfolio_returns: pd.DataFrame) -> Dict:
        """
        Calculate risk-adjusted returns using Fama-French factors.
        
        Args:
            portfolio_returns: Monthly portfolio returns by quintile
            
        Returns:
            Dictionary containing regression results for 3-factor and 5-factor models
        """

        if self.factor_3_data is None or self.factor_5_data is None:
            raise ValueError("Factor data not loaded. Call load_factor_data() first.")

        results = {}
        quintiles = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q5-Q1']

        for quintile in quintiles:
            if quintile not in portfolio_returns.columns:
                continue

            # 3-factor model
            try:
                aligned_3f = portfolio_returns[[quintile]].join(self.factor_3_data, how='inner')
                aligned_3f = aligned_3f.dropna()

                if len(aligned_3f) >= 10:  # Need sufficient observations
                    aligned_3f[f'{quintile}_excess'] = aligned_3f[quintile] - aligned_3f['RF']

                    result_3f = self.regression_analyzer.run_basic_regression(
                        aligned_3f,
                        f'{quintile}_excess',
                        ['Mkt-RF', 'SMB', 'HML'],
                        name=f'{quintile}_3factor'
                    )
                    results[f'{quintile}_3factor'] = result_3f
            except Exception as e:
                print(f"3-factor regression failed for {quintile}: {e}")

            # 5-factor model
            try:
                aligned_5f = portfolio_returns[[quintile]].join(self.factor_5_data, how='inner')
                aligned_5f = aligned_5f.dropna()

                if len(aligned_5f) >= 10:  # Need sufficient observations
                    aligned_5f[f'{quintile}_excess'] = aligned_5f[quintile] - aligned_5f['RF']

                    result_5f = self.regression_analyzer.run_basic_regression(
                        aligned_5f,
                        f'{quintile}_excess',
                        ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'],
                        name=f'{quintile}_5factor'
                    )
                    results[f'{quintile}_5factor'] = result_5f
            except Exception as e:
                print(f"5-factor regression failed for {quintile}: {e}")

        return results

    def calculate_sharpe_statistics(self, portfolio_returns: pd.DataFrame) -> Dict:
        """
        Calculate Sharpe ratios with statistical significance for all quintiles.

        Args:
            portfolio_returns: Monthly portfolio returns by quintile

        Returns:
            Dictionary with Sharpe ratio statistics for each quintile
        """
        results = {}
        quintiles = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q5-Q1']

        # Get risk-free rate
        rf = None
        if self.factor_3_data is not None:
            rf = self.factor_3_data['RF']

        for quintile in quintiles:
            if quintile not in portfolio_returns.columns:
                continue

            returns = portfolio_returns[quintile]

            # Align with RF if available
            if rf is not None:
                aligned = pd.concat([returns, rf], axis=1).dropna()
                if len(aligned) > 0:
                    aligned_rf = aligned.iloc[:, 1]
                    aligned_ret = aligned.iloc[:, 0]
                    results[quintile] = calculate_sharpe_ratio(aligned_ret, aligned_rf, annualize=True)
                else:
                    results[quintile] = calculate_sharpe_ratio(returns, annualize=True)
            else:
                results[quintile] = calculate_sharpe_ratio(returns, annualize=True)

        return results

    def create_results_table(self,
                             portfolio_returns: pd.DataFrame,
                             regression_results: Dict,
                             sharpe_results: Dict = None) -> str:
        """
        Create a formatted results table similar to Table II in the paper.

        Args:
            portfolio_returns: Monthly portfolio returns
            regression_results: Dictionary of regression results
            sharpe_results: Dictionary of Sharpe ratio statistics (optional)

        Returns:
            Formatted string table
        """

        quintiles = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q5-Q1']

        # Calculate summary statistics
        weight_type = 'Value-Weighted' if self.is_value_weighted else 'Equal-Weighted'
        table = f"\n## Moving Targets Calendar-Time Portfolio Returns ({weight_type} Quintile)\n\n"

        # Create table header (similar to paper's Table II)
        table += "| Moving Targets |"
        for quintile in quintiles:
            if quintile in portfolio_returns.columns:
                table += f" {quintile} |"
        table += "\n"

        table += "|" + "----------------|" + "-------------|" * len(
            [q for q in quintiles if q in portfolio_returns.columns]
            ) + "\n"

        # Pre-calculate all metrics for all quintiles
        def format_coef(coef, tstat):
            """Format coefficient with significance stars"""
            stars = ""
            abs_tstat = abs(tstat)
            if abs_tstat > 2.576:  # 1% significance
                stars = "***"
            elif abs_tstat > 1.96:  # 5% significance
                stars = "**"
            elif abs_tstat > 1.645:  # 10% significance
                stars = "*"
            return f"{coef:.4f}{stars}"

        # Excess Return row
        table += "| **Excess Return** |"
        for quintile in quintiles:
            if quintile not in portfolio_returns.columns:
                continue

            excess_ret = 0
            excess_tstat = 0

            if self.factor_3_data is not None:
                aligned = portfolio_returns[[quintile]].join(self.factor_3_data[['RF']], how='inner')
                aligned = aligned.dropna()
                if len(aligned) > 0:
                    excess_returns = aligned[quintile] - aligned['RF']
                    excess_ret = excess_returns.mean()
                    excess_std = excess_returns.std()
                    excess_tstat = excess_ret / (excess_std / np.sqrt(len(aligned))) if excess_std > 0 else 0

            table += f" {format_coef(excess_ret, excess_tstat)} |"
        table += "\n"

        # t-statistic row for Excess Return
        table += "| |"
        for quintile in quintiles:
            if quintile not in portfolio_returns.columns:
                continue

            excess_tstat = 0
            if self.factor_3_data is not None:
                aligned = portfolio_returns[[quintile]].join(self.factor_3_data[['RF']], how='inner')
                aligned = aligned.dropna()
                if len(aligned) > 0:
                    excess_returns = aligned[quintile] - aligned['RF']
                    excess_ret = excess_returns.mean()
                    excess_std = excess_returns.std()
                    excess_tstat = excess_ret / (excess_std / np.sqrt(len(aligned))) if excess_std > 0 else 0

            table += f" ({excess_tstat:.2f}) |"
        table += "\n"

        # 3-Factor Alpha row
        table += "| **3-Factor Alpha** |"
        for quintile in quintiles:
            if quintile not in portfolio_returns.columns:
                continue

            alpha_3f = 0
            tstat_3f = 0
            if f'{quintile}_3factor' in regression_results:
                result_3f = regression_results[f'{quintile}_3factor']
                alpha_3f = result_3f['coefficients'].get('const', 0)
                tstat_3f = result_3f['t_stats'].get('const', 0)

            table += f" {format_coef(alpha_3f, tstat_3f)} |"
        table += "\n"

        # t-statistic row for 3-Factor Alpha
        table += "| |"
        for quintile in quintiles:
            if quintile not in portfolio_returns.columns:
                continue

            tstat_3f = 0
            if f'{quintile}_3factor' in regression_results:
                result_3f = regression_results[f'{quintile}_3factor']
                tstat_3f = result_3f['t_stats'].get('const', 0)

            table += f" ({tstat_3f:.2f}) |"
        table += "\n"

        # 5-Factor Alpha row
        table += "| **5-Factor Alpha** |"
        for quintile in quintiles:
            if quintile not in portfolio_returns.columns:
                continue

            alpha_5f = 0
            tstat_5f = 0
            if f'{quintile}_5factor' in regression_results:
                result_5f = regression_results[f'{quintile}_5factor']
                alpha_5f = result_5f['coefficients'].get('const', 0)
                tstat_5f = result_5f['t_stats'].get('const', 0)

            table += f" {format_coef(alpha_5f, tstat_5f)} |"
        table += "\n"

        # t-statistic row for 5-Factor Alpha
        table += "| |"
        for quintile in quintiles:
            if quintile not in portfolio_returns.columns:
                continue

            tstat_5f = 0
            if f'{quintile}_5factor' in regression_results:
                result_5f = regression_results[f'{quintile}_5factor']
                tstat_5f = result_5f['t_stats'].get('const', 0)

            table += f" ({tstat_5f:.2f}) |"
        table += "\n"

        table += "\n*p<0.10, **p<0.05, ***p<0.01\n"
        table += "t-statistics in parentheses; stars use two-sided t-stat cutoffs\n\n"

        # Add summary statistics
        table += f"**Summary Statistics:**\n"
        table += f"- Observations: {len(portfolio_returns)}\n"
        table += f"- Date range: {portfolio_returns.index[0].strftime('%Y-%m')} to {portfolio_returns.index[-1].strftime('%Y-%m')}\n"

        # Add average returns by quintile
        table += f"\n**Average Monthly Returns:**\n"
        for quintile in quintiles:
            if quintile in portfolio_returns.columns:
                avg_ret = portfolio_returns[quintile].mean()
                std_ret = portfolio_returns[quintile].std()
                table += f"- {quintile}: {avg_ret:.4f} (σ={std_ret:.4f})\n"

        # Add Sharpe ratio summary (if available)
        if sharpe_results:
            table += f"\n**Annualized Sharpe Ratios (Lo 2002):**\n"
            for quintile in quintiles:
                if quintile in sharpe_results:
                    sr_data = sharpe_results[quintile]
                    sr = sr_data.get('sharpe_ratio', 0)
                    se = sr_data.get('se', 0)
                    p_val = sr_data.get('p_value', 1)
                    stars = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
                    table += f"- {quintile}: {sr:.3f}{stars} (SE={se:.3f}, p={p_val:.3f})\n"

            # Add Sharpe ratio component details (use Q1-Q5 for long-short strategy display)
            table += f"\n**Sharpe Ratio Components (Annualized):**\n\n"
            table += "| Quintile | Mean Return | Mean Excess Return | Std (Volatility) | Risk-Free Rate | Sharpe Ratio |\n"
            table += "|----------|-------------|-------------------|------------------|----------------|-------------|\n"
            for quintile in quintiles:
                if quintile in sharpe_results:
                    sr_data = sharpe_results[quintile]
                    mean_ret = sr_data.get('mean_return_ann', 0)
                    mean_excess = sr_data.get('mean_excess_return_ann', 0)
                    std_ret = sr_data.get('std_return_ann', 0)
                    mean_rf = sr_data.get('mean_rf_ann', 0)
                    sr = sr_data.get('sharpe_ratio', 0)
                    # Convert Q5-Q1 to Q1-Q5 for display (flip sign and keep self-financing RF=0).
                    if quintile == 'Q5-Q1':
                        display_name = 'Q1-Q5'
                        mean_ret = -mean_ret
                        mean_rf = 0.0
                        mean_excess = mean_ret
                        sr = -sr
                    else:
                        display_name = quintile
                    table += f"| {display_name} | {mean_ret:.4f} ({mean_ret*100:.2f}%) | {mean_excess:.4f} ({mean_excess*100:.2f}%) | {std_ret:.4f} ({std_ret*100:.2f}%) | {mean_rf:.4f} ({mean_rf*100:.2f}%) | {sr:.3f} |\n"

            # Add monthly components as well
            table += f"\n**Sharpe Ratio Components (Monthly):**\n\n"
            table += "| Quintile | Mean Return | Mean Excess Return | Std (Volatility) | Risk-Free Rate | Sharpe Ratio |\n"
            table += "|----------|-------------|-------------------|------------------|----------------|-------------|\n"
            for quintile in quintiles:
                if quintile in sharpe_results:
                    sr_data = sharpe_results[quintile]
                    mean_ret = sr_data.get('mean_return_monthly', 0)
                    mean_excess = sr_data.get('mean_excess_return_monthly', 0)
                    std_ret = sr_data.get('std_return_monthly', 0)
                    mean_rf = sr_data.get('mean_rf_monthly', 0)
                    sr = sr_data.get('sharpe_ratio_monthly', 0)
                    # Convert Q5-Q1 to Q1-Q5 for display (flip sign and keep self-financing RF=0).
                    if quintile == 'Q5-Q1':
                        display_name = 'Q1-Q5'
                        mean_ret = -mean_ret
                        mean_rf = 0.0
                        mean_excess = mean_ret
                        sr = -sr
                    else:
                        display_name = quintile
                    table += f"| {display_name} | {mean_ret:.6f} ({mean_ret*100:.4f}%) | {mean_excess:.6f} ({mean_excess*100:.4f}%) | {std_ret:.6f} ({std_ret*100:.4f}%) | {mean_rf:.6f} ({mean_rf*100:.4f}%) | {sr:.4f} |\n"

        return table

    def run_backtest(self,
                     mt_score_df: pd.DataFrame,
                     returns_df: pd.DataFrame,
                     score_col: str = 'mt_score') -> Tuple[pd.DataFrame, Dict, str, Dict]:
        """
        Run the complete backtest analysis.

        Args:
            mt_score_df: DataFrame with MT scores (from mt_score.py)
            returns_df: DataFrame with stock returns data
            score_col: Column name to use for quintile formation (default: 'mt_score')

        Returns:
            Tuple of (portfolio_returns, regression_results, results_table, sharpe_results)
        """

        print("Starting Moving Targets backtest...")
        print(f"Weighting scheme: {'Value-weighted' if self.is_value_weighted else 'Equal-weighted'}")
        print(f"Using score column: {score_col}")

        # Load factor data
        try:
            self.load_factor_data()
        except Exception as e:
            print(f"Warning: Could not load factor data: {e}")

        # Create quintile portfolios
        print("Creating quintile portfolios...")
        portfolio_df = self.create_quintile_portfolios(mt_score_df, returns_df, score_col)
        print(f"Created portfolios for {len(portfolio_df)} portfolio-month combinations")

        # Calculate aggregated portfolio returns
        print("Calculating portfolio returns...")
        portfolio_returns = self.calculate_portfolio_returns(portfolio_df)
        print(f"Portfolio returns calculated for {len(portfolio_returns)} months")

        # Calculate risk-adjusted returns
        regression_results = {}
        sharpe_results = {}
        if self.factor_3_data is not None and self.factor_5_data is not None:
            print("Calculating risk-adjusted returns...")
            regression_results = self.calculate_risk_adjusted_returns(portfolio_returns)
            print(f"Completed {len(regression_results)} regressions")

            # Calculate Sharpe ratios
            print("Calculating Sharpe ratios...")
            sharpe_results = self.calculate_sharpe_statistics(portfolio_returns)
            print(f"Calculated Sharpe ratios for {len(sharpe_results)} portfolios")

        # Create results table
        results_table = self.create_results_table(portfolio_returns, regression_results, sharpe_results)

        print("Backtest completed!")

        return portfolio_returns, regression_results, results_table, sharpe_results


def run_moving_targets_backtest(mt_score_df: pd.DataFrame,
                                returns_df: pd.DataFrame,
                                score_col: str = 'mt_score',
                                is_value_weighted: bool = False) -> Tuple[pd.DataFrame, Dict, str, Dict]:
    """
    Convenience function to run Moving Targets backtest.

    Args:
        mt_score_df: DataFrame with MT scores (from mt_score.py)
        returns_df: DataFrame with stock returns (TRAIN_DATA.csv format)
        score_col: Column name to use for quintile formation (default: 'mt_score')
        is_value_weighted: Whether to use value weighting (default: False)

    Returns:
        Tuple of (portfolio_returns, regression_results, results_table, sharpe_results)
    """

    backtest = MovingTargetsBacktest(is_value_weighted=is_value_weighted)
    return backtest.run_backtest(mt_score_df, returns_df, score_col)


if __name__ == "__main__":
    print("Moving Targets Backtest Module")
    print("See example_usage_backtest.py for usage examples.")
