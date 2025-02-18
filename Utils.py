import numpy as np
import pandas as pd
from arch import arch_model


homepath = "C:/A.PROJECTS/stockprediction/"




def compute_returns(df):
    """
    Computes true weekly, monthly, quarterly, and annual simple and log returns
    from a DataFrame of daily adjusted prices.

    Args:
        df (pd.DataFrame): DataFrame with daily adjusted prices. The index should be datetime.

    Returns:
        dict: A dictionary containing DataFrames for weekly, monthly, quarterly, and annual simple and log returns.
    """

    # Ensure the index is datetime and sorted
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")
    
    df = df.sort_index()

    # Compute Simple Returns
    weekly_simple_returns = df.pct_change(periods=5)  # 5 trading days ≈ 1 week
    monthly_simple_returns = df.pct_change(periods=21)  # 21 trading days ≈ 1 month
    quarterly_simple_returns = df.pct_change(periods=63)  # 63 trading days ≈ 1 quarter
    annual_simple_returns = df.pct_change(periods=252)  # 252 trading days ≈ 1 year

    # Compute Log Returns
    weekly_log_returns = np.log(df / df.shift(5))  # log(P_t / P_{t-5})
    monthly_log_returns = np.log(df / df.shift(21))
    quarterly_log_returns = np.log(df / df.shift(63))
    annual_log_returns = np.log(df / df.shift(252))

    return {
        'weekly_simple': weekly_simple_returns.dropna(),
        'monthly_simple': monthly_simple_returns.dropna(),
        'quarterly_simple': quarterly_simple_returns.dropna(),
        'annual_simple': annual_simple_returns.dropna(),
        'weekly_log': weekly_log_returns.dropna(),
        'monthly_log': monthly_log_returns.dropna(),
        'quarterly_log': quarterly_log_returns.dropna(),
        'annual_log': annual_log_returns.dropna()
    }


def compute_conditional_volatility(log_returns_df, frequency="weekly"):
    """
    Computes the latest conditional volatility using a GARCH(1,1) model for each column in a DataFrame.
    
    Parameters:
    - log_returns_df (pd.DataFrame): A DataFrame where each column is a time series of daily logarithmic returns.
    - frequency (str): Frequency to compute volatility ('weekly', 'monthly', 'quarterly', 'annual').
    
    Returns:
    - pd.Series: Latest estimated conditional volatility for each column.
    """

    # Define resampling rules for different timeframes
    freq_map = {
        "weekly": "W",
        "monthly": "M",
        "quarterly": "Q",
        "annual": "A"
    }

    # Check if the selected frequency is valid
    if frequency not in freq_map:
        raise ValueError("Invalid frequency. Choose from: 'weekly', 'monthly', 'quarterly', 'annual'")

    # Resample daily log returns to the selected frequency
    resample_rule = freq_map[frequency]
    log_returns_resampled = log_returns_df.resample(resample_rule).sum()

    # Function to compute GARCH volatility for a single column
    def compute_volatility(series):
        if series.isnull().all():  # Check if all values are NaN
            return np.nan
        try:
            garch_model = arch_model(series.dropna(), vol='Garch', p=1, q=1)
            garch_fit = garch_model.fit(disp="off")
            return np.sqrt(garch_fit.conditional_volatility.iloc[-1])
        except:
            return np.nan  # Return NaN if GARCH model fails

    # Apply function to each column in DataFrame
    volatilities = log_returns_resampled.apply(compute_volatility, axis=0)

    return volatilities





class WeightingScenario:
    """
    Class to handle weight generation for portfolio scenarios.
    Allows generating multiple scenarios or utilizing multiple custom weight scenarios.
    """
    def __init__(self, num_assets, num_scenarios=1, custom_weights=None):
        """
        Initialize the weighting scenario generator.

        Args:
            num_assets (int): The number of assets to generate weights for.
            num_scenarios (int): The number of scenarios to generate (for random scenarios).
            custom_weights (ndarray or None): A numpy array where each row represents a scenario's weights.
                                              Each row should have the same length as num_assets.
        """
        self.num_assets = num_assets
        self.num_scenarios = num_scenarios
        self.custom_weights = custom_weights

    def generate_weights(self):
        """
        Generate weights for the scenarios. If custom_weights is provided,
        it returns the custom scenarios as they are.

        Returns:
            ndarray: A 2D numpy array where each row is a weight vector for a scenario.
        """
        if self.custom_weights is not None:
            # Validate the shape of the custom_weights ndarray
            if self.custom_weights.shape[1] != self.num_assets:
                raise ValueError(
                    f"Each custom scenario must have {self.num_assets} weights, "
                    f"but got {self.custom_weights.shape[1]}."
                )
            weights = self.custom_weights
        else:
            # Generate num_scenarios rows with random Dirichlet weights
            weights = np.random.dirichlet(alpha=np.ones(self.num_assets), size=self.num_scenarios)

        return np.array(weights)