import pandas as pd
import pmdarima
from prophet import Prophet
from datetime import datetime, timedelta
import logging
import warnings

logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

# Suppress sklearn deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

LOAD_PREDICTORS = {
    "constant": ConstantPredictor,
    "arima": ARIMAPredictor,
    "prophet": ProphetPredictor,
}

class ConstantPredictor:
    """
    Assume load is constant and predict the next load to be the same as most recent load
    """
    def __init__(self, **kwargs):
        self.prev_load = None

    def add_data_point(self, value):
        self.prev_load = value
    
    def predict_next(self):
        return self.prev_load

class ARIMAPredictor:
    def __init__(self, window_size=100, minimum_data_points=5):
        self.window_size = window_size  # How many past points to use
        self.model = None
        self.data_buffer = []
        self.minimum_data_points = minimum_data_points
        
    def add_data_point(self, value):
        """Add new data point to the buffer"""
        self.data_buffer.append(value)
        
        # Keep only the last window_size points
        if len(self.data_buffer) > self.window_size:
            self.data_buffer = self.data_buffer[-self.window_size:]
    
    def predict_next(self):
        """Predict the next value(s)"""
        if len(self.data_buffer) < self.minimum_data_points: 
            # not enough data to make a prediction, simply return the last value
            return self.data_buffer[-1]
            
        # Fit auto ARIMA model
        self.model = pmdarima.auto_arima(
            self.data_buffer,
            suppress_warnings=True,
            error_action='ignore',
        )
        
        # Make prediction
        forecast = self.model.predict(n_periods=1)
        
        return forecast[0]

class ProphetPredictor:
    def __init__(self, window_size=100, step_size=3600, minimum_data_points=5):
        self.window_size = window_size
        self.data_buffer = []
        self.minimum_data_points = minimum_data_points
        self.curr_step = 0
        self.step_size = step_size
        self.start_date = datetime(2024, 1, 1)  # Base date for generating timestamps
        
    def add_data_point(self, value):
        """Add new data point to the buffer"""
        # Use proper datetime for Prophet
        timestamp = self.start_date + timedelta(seconds=self.curr_step)
        self.data_buffer.append({'ds': timestamp, 'y': value})
        self.curr_step += 1
        
        # Keep only the last window_size points
        if len(self.data_buffer) > self.window_size:
            self.data_buffer = self.data_buffer[-self.window_size:]
    
    def predict_next(self):
        """Predict the next value"""
        if len(self.data_buffer) < self.minimum_data_points: 
            # not enough data to make a prediction, simply return the last value
            return self.data_buffer[-1]['y']
            
        # Convert to DataFrame
        df = pd.DataFrame(self.data_buffer)
        
        # Initialize and fit Prophet model
        model = Prophet()
        
        # Fit the model
        model.fit(df)
        
        # Create future dataframe for next timestamp
        next_timestamp = self.start_date + timedelta(seconds=self.curr_step)
        future_df = pd.DataFrame({'ds': [next_timestamp]})
        
        # Make prediction
        forecast = model.predict(future_df)
        
        return forecast['yhat'].iloc[0]