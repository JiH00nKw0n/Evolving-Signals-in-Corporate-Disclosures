import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()


class FinancialModelingPrepAPI:
    def __init__(self):
        self.api_key = os.getenv("FMP_API_KEY")
        self.base_url = "https://financialmodelingprep.com/api/v3"

    def get_symbol_from_isin(self, isin: str) -> Optional[str]:
        """Find symbol from ISIN"""
        # Use FMP API's search-isin endpoint
        try:
            url = f"https://financialmodelingprep.com/stable/search-isin?isin={isin}&apikey={self.api_key}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            # Return the symbol of the first result if search results exist
            if data and len(data) > 0:
                return data[0].get('symbol')
            else:
                print(f"No results found for ISIN: {isin}")
                return None

        except Exception as e:
            print(f"Error getting symbol from ISIN {isin}: {e}")
            return None

    def get_historical_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical price data"""
        url = f"{self.base_url}/historical-price-full/{symbol}?from={start_date}&to={end_date}&apikey={self.api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if 'historical' not in data:
                return pd.DataFrame()

            df = pd.DataFrame(data['historical'])
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            return df
        except Exception as e:
            print(f"Error getting historical prices for {symbol}: {e}")
            return pd.DataFrame()

    def get_monthly_returns(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Calculate monthly returns based on month-end prices"""
        df = self.get_historical_prices(symbol, start_date, end_date)
        if df.empty:
            return pd.DataFrame()

        # Extract only month-end data
        df['year_month'] = df['date'].dt.to_period('M')
        monthly_data = df.groupby('year_month').last().reset_index()
        monthly_data = monthly_data.sort_values('date')

        # Calculate returns
        monthly_data['ret_0'] = monthly_data['adjClose'].pct_change()  # Current month return
        monthly_data['ret_minus1'] = monthly_data['ret_0'].shift(1)  # Previous month return Ret(-1, 0)

        # Calculate 12-month cumulative return Ret(-12, -1)
        monthly_data['ret_12m'] = (monthly_data['adjClose'] / monthly_data['adjClose'].shift(12)) - 1

        return monthly_data

    def get_historical_market_cap(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical market cap data (request in 1-year intervals)"""
        from datetime import datetime, timedelta
        import time

        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        all_data = []
        current_dt = start_dt

        while current_dt <= end_dt:
            # Request in 1-year intervals
            year_end = min(current_dt + timedelta(days=365), end_dt)

            from_str = current_dt.strftime('%Y-%m-%d')
            to_str = year_end.strftime('%Y-%m-%d')

            url = f"{self.base_url}/historical-market-capitalization/{symbol}?from={from_str}&to={to_str}&apikey={self.api_key}"

            try:
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()

                if data:
                    all_data.extend(data)
                else:
                    pass

                # Brief wait to consider API rate limit
                time.sleep(0.1)

            except Exception as e:
                print(f"Error getting historical market cap for {symbol} ({from_str} to {to_str}): {e}")

            current_dt = year_end + timedelta(days=1)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').drop_duplicates(subset=['date']).reset_index(drop=True)
        return df

    def get_market_cap(self, symbol: str) -> Optional[float]:
        """Get current market cap (Size)"""
        url = f"{self.base_url}/profile/{symbol}?apikey={self.api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return data[0]['mktCap'] if data else None
        except Exception as e:
            print(f"Error getting market cap for {symbol}: {e}")
            return None

    def get_key_metrics(self, symbol: str, period: str = 'quarterly') -> List[Dict]:
        """Get key financial metrics"""
        url = f"{self.base_url}/key-metrics/{symbol}?period={period}&apikey={self.api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting key metrics for {symbol}: {e}")
            return []

    def get_balance_sheet(self, symbol: str, period: str = 'quarter') -> List[Dict]:
        """Get quarterly balance sheet data"""
        url = f"{self.base_url}/balance-sheet-statement/{symbol}?period={period}&limit=200&apikey={self.api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            print(f"Balance sheet API response status: {response.status_code}")
            print(f"First balance sheet entry keys: {list(data[0].keys()) if data else 'No data'}")
            return data
        except Exception as e:
            print(f"Error getting balance sheet for {symbol}: {e}")
            print(f"Full URL: {url}")
            return []

    def calculate_monthly_size_and_logbm(self, symbol: str, monthly_data: pd.DataFrame, start_date: str,
                                         end_date: str) -> pd.DataFrame:
        """Calculate monthly Size and logBM"""
        # 1. Get historical market cap data
        historical_mcap = self.get_historical_market_cap(symbol, start_date, end_date)

        # 2. Get quarterly balance sheet data
        balance_sheets = self.get_balance_sheet(symbol)

        monthly_data_copy = monthly_data.copy()
        monthly_data_copy['size'] = None
        monthly_data_copy['log_bm'] = None

        # Size: Use historical market cap data
        if not historical_mcap.empty:
            print(f"Historical market cap data shape: {historical_mcap.shape}")
            print(f"Market cap date range: {historical_mcap['date'].min()} to {historical_mcap['date'].max()}")

            # Map market cap by month-end dates
            historical_mcap['year_month'] = historical_mcap['date'].dt.to_period('M')
            mcap_monthly = historical_mcap.groupby('year_month').last().reset_index()

            for idx, row in monthly_data_copy.iterrows():
                month_period = row['date'].to_period('M')
                matching_mcap = mcap_monthly[mcap_monthly['year_month'] == month_period]
                if not matching_mcap.empty:
                    size_value = matching_mcap.iloc[0]['marketCap']
                    monthly_data_copy.at[idx, 'size'] = np.log(size_value / 1000000)
                    monthly_data_copy.at[idx, 'marketCap'] = size_value

        else:
            pass

        # logBM: Use quarterly equity data
        if balance_sheets:
            equity_data = []
            for bs in balance_sheets:
                if bs.get('totalStockholdersEquity'):
                    equity_data.append(
                        {
                            'date': pd.to_datetime(bs['date']),
                            'equity': bs['totalStockholdersEquity']
                        }
                    )

            if equity_data:
                equity_df = pd.DataFrame(equity_data).sort_values('date')

                for idx, row in monthly_data_copy.iterrows():
                    month_end_date = row['date']

                    # Use the latest quarterly equity before month-end (only data provided by API)
                    recent_equity = None
                    matching_entries = equity_df[equity_df['date'] <= month_end_date]
                    if not matching_entries.empty:
                        recent_equity = matching_entries.iloc[-1]['equity']

                    size_value = monthly_data_copy.at[idx, 'marketCap']

                    if recent_equity and pd.notna(size_value) and size_value > 0:
                        try:
                            log_bm = np.log(recent_equity / size_value)
                        except Exception as e:
                            print(recent_equity, size_value)
                        monthly_data_copy.at[idx, 'log_bm'] = log_bm
                    else:
                        pass
            else:
                pass
        else:
            pass

        return monthly_data_copy

    def get_monthly_factors(self, isin: str, start_date: str, end_date: str) -> Dict:
        """Get all monthly factor data based on ISIN"""
        from datetime import datetime, timedelta

        # 1. Find symbol from ISIN
        symbol = self.get_symbol_from_isin(isin)
        if not symbol:
            return {"error": f"Symbol not found for ISIN: {isin}"}

        # 2. Get data from 1 year before to calculate 12-month returns
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        extended_start = start_dt - timedelta(days=365)
        extended_start_str = extended_start.strftime('%Y-%m-%d')

        # 3. Monthly return data (with extended period)
        monthly_returns = self.get_monthly_returns(symbol, extended_start_str, end_date)
        if monthly_returns.empty:
            return {"error": f"No price data found for symbol: {symbol}"}

        # 4. Calculate monthly Size and logBM
        monthly_data_with_factors = self.calculate_monthly_size_and_logbm(
            symbol, monthly_returns, extended_start_str, end_date
        )

        # 5. Filter only data after start_date from results
        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        filtered_data = monthly_data_with_factors[monthly_data_with_factors['date'].dt.date >= start_date_dt].copy()
        filtered_data = filtered_data.reset_index(drop=True)

        # 6. Combine results
        result = {
            "isin": isin,
            "symbol": symbol,
            "monthly_data": filtered_data
        }

        return result


def test_api():
    """API test function"""
    fmp = FinancialModelingPrepAPI()

    # Test ISIN (Apple Inc.)
    test_isin = "US0378331005"

    # Set data period
    start_date = "2010-01-01"
    end_date = "2025-07-31"

    print(f"Testing with ISIN: {test_isin}")
    print("-" * 50)

    # Get all factor data
    result = fmp.get_monthly_factors(test_isin, start_date, end_date)

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"Symbol: {result['symbol']}")

    print("\nMonthly Factor Data:")
    monthly_data = result['monthly_data']
    if not monthly_data.empty:
        # Display recent 6 months data
        columns_to_show = ['date', 'ret_0', 'ret_minus1', 'ret_12m', 'size', 'log_bm']
        recent_data = monthly_data[columns_to_show]
        print(recent_data.to_string(index=False))
    else:
        print("No monthly data available")


if __name__ == "__main__":
    # test_api()

    api_key = os.getenv("FMP_API_KEY")
    base_url = "https://financialmodelingprep.com/api/v3"
    symbol = "ADT"
    start_date = "2010-01-01"
    end_date = "2025-07-31"
    isin = "US00101J1060"
    # url = f"{base_url}/historical-price-full/{symbol}?from={start_date}&to={end_date}&apikey={api_key}"
    url = f"https://financialmodelingprep.com/stable/search-isin?isin={isin}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        print(data)
    except Exception as e:
        pass
