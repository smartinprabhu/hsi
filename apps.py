import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.interpolate import UnivariateSpline
import plotly.graph_objects as go
from prophet import Prophet
from datetime import timedelta

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Increase the container width
st.markdown(
    """
    <style>
    .reportview-container .main .block-container {
        max-width: 1200px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def main():
    # Add a file uploader widget
    uploaded_file = st.file_uploader("Upload a CSV, XLSX, or JSON file", type=["csv", "xlsx", "json"])

    # Create empty placeholders for later use
    uploaded_dataset_placeholder = st.empty()
    uploaded_df_placeholder = st.empty()

    if uploaded_file is not None:
        # Read the uploaded file as a DataFrame
        df = pd.read_csv(uploaded_file) if uploaded_file.type == 'application/vnd.ms-excel' else pd.read_excel(uploaded_file) if uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' else pd.read_json(uploaded_file)

        # Update the placeholders
        uploaded_dataset_placeholder.write("Dataset Uploaded :")

        # Add icons for navigation
        st.markdown("---")
        st.image("https://media.licdn.com/dms/image/C4D0BAQGxJD0PEI8xmg/company-logo_200_200/0/1630548263264/helix_sense_logo?e=2147483647&v=beta&t=xSAnBJHyLfXAD-luInFxh-xA7nGF42EUTvMwcTinsJw",width=30)
        st.write("Click on below to view the summary and description of the dataset uploaded:")

        # Show EDA report on a separate page when the icon is clicked
        if st.button("EDA Report"):
           with st.spinner("Processing..."):
            time.sleep(2)
            display_eda_report(df)
            process_and_display_data(df)
            st.session_state.show_data_preparation_button = True  # Set session state variable to True

    # Check if the button for showing Data Preparation should be displayed
    if st.session_state.get("show_data_preparation_button", False):
        # Add button for navigating to the Data Preparation page
        if st.sidebar.button("Data Preparation"):
            st.session_state.page = "data_preparation"

    # Retrieve the page parameter
    page = st.session_state.get("page", "main")

    # Process and display data based on the selected page
    if page == "main":
        pass
    elif page == "data_preparation":
        if 'df' in locals():
           with st.spinner("Processing..."):
            time.sleep(2)
            data_preparation(df)
            prophet(df)
            give(df)
            got(df)
            gear(df)






def display_eda_report(df):
    st.title("Exploratory Data Analysis (EDA) Report")

    # Add sections for EDA report
    with st.expander("Exploratory Data Analysis (EDA)"):
        # DataFrame Shape
        st.subheader("DataFrame Shape:")
        st.write(df.shape)
        
        # Data Types
        st.subheader("Data Types:")
        st.write(df.dtypes)
        # Columns
        st.subheader("Columns:")
        st.write(df.columns.tolist())

        # Summary Statistics
        st.subheader("Summary Statistics:")
        st.write(df.describe())

        # Missing Values
        st.subheader("Missing Values:")
        st.write(df.isnull().sum())
        # DataFrame Head
        st.subheader("DataFrame Head:")
        st.write(df.head())

        # DataFrame Tail
        st.subheader("DataFrame Tail:")
        st.write(df.tail())

def process_and_display_data(df):
    st.title("Processed Data")
    
    # Add an expander for the processed data section
    with st.expander("Processed Data"):
        # Convert Timestamp to datetime and set as index
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)

        # Resample by hour and calculate energy consumption
        df_hour = df['Answer Value'].resample('H').agg(['first', 'last'])
        df_hour['Energy Consumption (kWh)'] = df_hour['last'] - df_hour['first']
        non_missing_indices = df_hour['Energy Consumption (kWh)'].notnull()
        missing_indices = ~non_missing_indices
        # Display processed data and statistics
        st.header("Transformed Dataset of Intervals")

        # Show basic statistics
        st.subheader("Descriptive Statistics:")
        st.write(df_hour.describe())

        # Show the first few rows of the dataset
        st.subheader("First Few Rows of the Dataset:")
        st.write(df_hour.head())
        st.subheader("Last Few Rows of the Dataset:")
        st.write(df_hour.tail())
        st.subheader("Records having missing values")
        st.write(df_hour[df_hour['Energy Consumption (kWh)'].isnull()])
        st.header("sum of missing and non missing indices")
        st.write(missing_indices.sum(),"missing value records")
        st.write(non_missing_indices.sum(),"non missing value records")
        # Plot distribution of energy consumption over hourly intervals using Plotly
        st.header("Hourly Energy Consumption Distribution")
        fig = go.Figure()

        # Add scatter plot
        fig.add_trace(go.Scatter(x=df_hour.index, y=df_hour['Energy Consumption (kWh)'],
                                 mode='lines', name='Energy Consumption (kWh)',
                                 hovertemplate='<b>Time:</b> %{x}<br><b>Energy Consumption:</b> %{y} kWh<br><extra></extra>'))

        # Update layout with detailed configurations
        fig.update_layout(
            title='Hourly Energy Consumption Distribution',  # Add title
            xaxis=dict(title='Time (Hourly Intervals)'),  # X-axis title
            yaxis=dict(title='Energy Consumption (kWh)'),  # Y-axis title
            hovermode='closest',  # Show hover information for the nearest data point
            hoverlabel=dict(bgcolor='white', font_size=12),  # Hover label styling
            margin=dict(l=50, r=50, t=50, b=50),  # Adjust margin for better layout
            width=1300  # Set width of the plot
        )

        # Show the plot
        st.plotly_chart(fig)
def data_preparation(df):
  st.title("Finding and replacing missing values with synthetic data")
  with st.spinner("Processing..."):
          time.sleep(2)
  with st.expander("Missing value replacement"):

    df.reset_index(inplace=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    st.title("Data Preparation: Filling Missing Values")
    df.set_index('Timestamp', inplace=True)
    df_hour = df['Answer Value'].resample('H').agg(['first', 'last'])
    df_hour['Energy Consumption (kWh)'] = df_hour['last'] - df_hour['first']
    non_missing_indices = df_hour['Energy Consumption (kWh)'].notnull()
    missing_indices = ~non_missing_indices
    x = np.arange(len(df_hour.index[non_missing_indices]))
    spline = UnivariateSpline(x, df_hour.loc[non_missing_indices, 'Energy Consumption (kWh)'], k=3)
    interpolated_values = spline(np.arange(len(df_hour)))
    df.reset_index(inplace=True)
    synthetic_data_points = pd.DataFrame({
        'Timestamp': df_hour.index[missing_indices],
        'Synthetic Energy Consumption (kWh)': interpolated_values[missing_indices]
    })
    
    m=df_hour[df_hour['Energy Consumption (kWh)'].isnull()]
    st.subheader("Records having missing values")
    st.write(m[['Energy Consumption (kWh)']])

    # Display synthetic data points used to replace missing values
    st.subheader("Synthetic Data Points Used to Replace Missing Values:")
    st.write(synthetic_data_points)
    # Display the DataFrame with missing values filled
    st.subheader("DataFrame with Synthetic Data:")
    st.write(df_hour)
    interpolated_values = spline(np.arange(len(df_hour)))
    df_hour['Energy Consumption (kWh)'].loc[missing_indices] = interpolated_values[missing_indices]
    # Plot the original data with missing values indicated in red
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_hour.index, y=df_hour['Energy Consumption (kWh)'], mode='lines', name='Original Data',
                             hovertemplate='<b>Time:</b> %{x}<br><b>Energy Consumption:</b> %{y} kWh<br><extra></extra>'))  # Set hover color for non-missing values to blue
    fig.add_trace(go.Scatter(x=df_hour.index[missing_indices], y=df_hour['Energy Consumption (kWh)'][missing_indices],
                             mode='markers', marker=dict(color='red'), name='synthetic data Values',
                             hovertemplate='<b>Time:</b> %{x}<br><b>Energy Consumption:</b> %{y} kWh <br>(<span style="color:red">Missing</span>)<br><extra></extra>'))

    # Update layout for the plot
    fig.update_layout(title='Energy Consumption (kWh)', xaxis_title='Timestamp', yaxis_title='Energy Consumption (kWh)',
                      hovermode='closest',  # Show hover information for the nearest data point
                      hoverlabel=dict(bgcolor='white', font_size=12),  # Hover label styling
                      margin=dict(l=50, r=50, t=50, b=50),  # Adjust margin for better layout
                      width=1300  # Set width of the plot
                      )

    # Display the plot
    st.subheader("Original Data with Missing Values")
    st.plotly_chart(fig)



def prophet(df):
  st.title("Prophet Model")
  with st.spinner("Processing..."):
          time.sleep(2)
  with st.expander("Seasonality of exisiting data frame (present data)"):

    # Reset index and convert timestamp to datetime
    df.reset_index(inplace=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Resample data by hour
    df_hour = df.resample('H', on='Timestamp').agg({'Answer Value': ['first', 'last']})
    df_hour.columns = ['first', 'last']
    
    # Calculate energy consumption
    df_hour['Energy Consumption (kWh)'] = df_hour['last'] - df_hour['first']
    
    # Interpolate missing values
    non_missing_indices = df_hour['Energy Consumption (kWh)'].notnull()
    missing_indices = ~non_missing_indices
    x = np.arange(len(df_hour.index[non_missing_indices]))
    spline = UnivariateSpline(x, df_hour.loc[non_missing_indices, 'Energy Consumption (kWh)'], k=3)
    interpolated_values = spline(np.arange(len(df_hour)))
    df_hour['Energy Consumption (kWh)'].loc[missing_indices] = interpolated_values[missing_indices]
    
    # Reset index and rename columns for Prophet compatibility
    df_hour.reset_index(inplace=True)
    df_hour.rename(columns={'Timestamp': 'ds', 'Energy Consumption (kWh)': 'y'}, inplace=True)
    
    # Initialize and fit Prophet model
    model = Prophet()
    model.fit(df_hour)

    # Plot components
    st.subheader("Seasonality componnents of the the dataframe  for present hourly dataframe")
    st.markdown(
        """
        <style>
        .reportview-container .main .block-container {
            max-width: 500px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Plot components
    fig = model.plot_components(model.predict(df_hour))
    st.pyplot(fig)
import time
import random
def give(df):
    st.title("Prophet Model")
    with st.spinner("Processing..."):
        time.sleep(2)
        
    with st.expander("Seasonality of forecasted data frame (future data)"):
        with st.spinner("Processing..."):
            time.sleep(2)  # # Simulate processing time

    # Check if 'level_0' column already exists before resetting the index
    if 'level_0' not in df.columns:
        df.reset_index(inplace=True)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    df_hour = df['Answer Value'].resample('H').agg(['first', 'last'])
    df_hour['Energy Consumption (kWh)'] = df_hour['last'] - df_hour['first']
    non_missing_indices = df_hour['Energy Consumption (kWh)'].notnull()
    missing_indices = ~non_missing_indices
    x = np.arange(len(df_hour.index[non_missing_indices]))
    spline = UnivariateSpline(x, df_hour.loc[non_missing_indices, 'Energy Consumption (kWh)'], k=3)
    interpolated_values = spline(np.arange(len(df_hour)))
    synthetic_data_points = pd.DataFrame({
        'Timestamp': df_hour.index[missing_indices],
        'Synthetic Energy Consumption (kWh)': interpolated_values[missing_indices]
    })
    

    interpolated_values = spline(np.arange(len(df_hour)))
    df_hour['Energy Consumption (kWh)'].loc[missing_indices] = interpolated_values[missing_indices]
    
    
    df_hour.reset_index(inplace=True)
    df_hour.rename(columns={'Timestamp': 'ds', 'Energy Consumption (kWh)': 'y'}, inplace=True)

    # Display loading spinner while performing Prophet forecasting
    with st.spinner("Performing Prophet Forecasting..."):
        # Define an expanded parameter grid with finer granularity
        param_grid = {
            'changepoint_prior_scale': [random.uniform(0.0001, 0.002) for _ in range(100)],
            'seasonality_prior_scale': [random.uniform(0.001, 20.0) for _ in range(100)],
            'holidays_prior_scale': [random.uniform(0.001, 2.0) for _ in range(100)],
            'seasonality_mode': ['additive', 'multiplicative'],
        }
        # Initialize best parameters and best variability
        best_params = {}
        best_variability = float('inf')

        # Increase the number of iterations for random search
        num_iterations = 100

        # Iterate over a random selection of parameter combinations 
        for _ in range(num_iterations):
            params = {k: random.choice(v) for k, v in param_grid.items()}
        
        # Initialize and fit Prophet model with current parameters
        model = Prophet(**params)
        model.fit(df_hour)

        # Make forecasts
        future_start_date = df_hour['ds'].max() + pd.Timedelta(hours=1)
        future_end_date = future_start_date + pd.Timedelta(hours=24*30*2)  # Forecasting for the next two months
        future = pd.DataFrame({'ds': pd.date_range(start=future_start_date, end=future_end_date, freq='H')})
        forecast = model.predict(future)

        # Calculate variability
        variability = (forecast['yhat_upper'] - forecast['yhat_lower']).mean()

        # Update best parameters if the variability is lower
        if variability < best_variability:
            best_variability = variability
            best_params = params

        # Use the best parameters to train the final model
        best_model = Prophet(**best_params)
        best_model.fit(df_hour)

        # Make forecasts with the best model
        forecast = best_model.predict(future)

        # Print uncertainty and variability with the best model
        print(forecast['yhat_upper'] - forecast['yhat_lower'])
        print((forecast['yhat_upper'] - forecast['yhat_lower']).mean())
        st.subheader("Future Start Date")
        st.write(future_start_date)
        st.subheader("Future Dataframe")
        st.write(future)
        st.subheader("Forecast")
        st.write(forecast)

        st.subheader("Distribution of the Forecasted Values")
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
        fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
        fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', name='Confidence Interval'))
        fig_forecast.update_layout(title='Prophet Forecast',
                                   xaxis_title='Timestamp',
                                   yaxis_title='Energy Consumption (kWh)',
                                   hovermode='closest',  # Show hover information for the nearest data point
                                   hoverlabel=dict(bgcolor='white', font_size=12),  # Hover label styling
                                   margin=dict(l=50, r=50, t=50, b=50),  # Adjust margin for better layout
                                   width=1300 ) # Set width of the plot)
        st.plotly_chart(fig_forecast)

        fig = go.Figure()

        # Plot actual data
        fig.add_trace(go.Scatter(x=df_hour['ds'], y=df_hour['y'], mode='lines', name='Actual'))

        # Plot forecasted data
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))

        # Plot uncertainty intervals
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', fill=None, line=dict(color='gray'), name='Upper Bound'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', fill='tonexty', line=dict(color='gray'), name='Lower Bound'))

        # Update layout
        fig.update_layout(title="Prophet Forecast",
                          xaxis_title="Date",
                          yaxis_title="Energy Consumption (kWh)",
                          showlegend=True)
        st.plotly_chart(fig)

        # Display forecasted components
        st.subheader("Forecasted Components")
        st.pyplot(best_model.plot_components(forecast))

    
    

def got(df):
    st.title("Prophet Model II")
    with st.spinner("Processing..."):
        time.sleep(2)
        
    with st.expander("Seasonality of forecasted data frame (past data)"):
        with st.spinner("Processing..."):
            time.sleep(2)  # Simulate processing time

    # Check if 'level_0' column already exists before resetting the index
    if 'level_0' not in df.columns:
        df.reset_index(inplace=True)
    df.reset_index(inplace=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    df_hour = df['Answer Value'].resample('H').agg(['first', 'last'])
    df_hour['Energy Consumption (kWh)'] = df_hour['last'] - df_hour['first']
    non_missing_indices = df_hour['Energy Consumption (kWh)'].notnull()
    missing_indices = ~non_missing_indices
    x = np.arange(len(df_hour.index[non_missing_indices]))
    spline = UnivariateSpline(x, df_hour.loc[non_missing_indices, 'Energy Consumption (kWh)'], k=3)
    interpolated_values = spline(np.arange(len(df_hour)))
    synthetic_data_points = pd.DataFrame({
        'Timestamp': df_hour.index[missing_indices],
        'Synthetic Energy Consumption (kWh)': interpolated_values[missing_indices]
    })
    

    interpolated_values = spline(np.arange(len(df_hour)))
    df_hour['Energy Consumption (kWh)'].loc[missing_indices] = interpolated_values[missing_indices]
    
    
    df_hour.reset_index(inplace=True)
    df_hour.rename(columns={'Timestamp': 'ds', 'Energy Consumption (kWh)': 'y'}, inplace=True)

    # Display loading spinner while performing Prophet forecasting
    with st.spinner("Performing Prophet Forecasting..."):
        # Define an expanded parameter grid with finer granularity
        param_grid = {
            'changepoint_prior_scale': [random.uniform(0.0001, 0.002) for _ in range(100)],
            'seasonality_prior_scale': [random.uniform(0.001, 20.0) for _ in range(100)],
            'holidays_prior_scale': [random.uniform(0.001, 2.0) for _ in range(100)],
            'seasonality_mode': ['additive', 'multiplicative'],
        }
        # Initialize best parameters and best variability
        best_params = {}
        best_variability = float('inf')

        # Increase the number of iterations for random search
        num_iterations = 100

        # Iterate over a random selection of parameter combinations 
        for _ in range(num_iterations):
            params = {k: random.choice(v) for k, v in param_grid.items()}
        
        # Initialize and fit Prophet model with current parameters
        model = Prophet(**params)
        model.fit(df_hour)
        # Use the best parameters to train the final model
        best_model = Prophet(**best_params)
        best_model.fit(df_hour)
        # Make forecasts
        future_start_date = df_hour['ds'].max() + pd.Timedelta(hours=1)
        future_end_date = future_start_date + pd.Timedelta(hours=24*30*2)  # Forecasting for the next two months
        future = pd.DataFrame({'ds': pd.date_range(start=future_start_date, end=future_end_date, freq='H')})
        forecast = model.predict(future)

        # Make forecasts
        past_end_date = df_hour['ds'].min()  # End at the hour before the first timestamp
        past = pd.date_range(end=past_end_date, periods=24*30*2, freq='H')  # Forecasting for the past two months
        past = pd.DataFrame({'ds': past})
        past_forecast = model.predict(past)

        # Calculate variability
        variability = (past_forecast['yhat_upper'] - past_forecast['yhat_lower']).mean()

        # Update best parameters if the variability is lower
        if variability < best_variability:
            best_variability = variability
            best_params = params

        # Use the best parameters to train the final model
        best_model = Prophet(**best_params)
        best_model.fit(df_hour)

        # Make forecasts with the best model
        forecast = best_model.predict(past)

        # Print uncertainty and variability with the best model
        print(past_forecast['yhat_upper'] - past_forecast['yhat_lower'])
        print((past_forecast['yhat_upper'] - past_forecast['yhat_lower']).mean())
        st.subheader("past end Date")
        st.write(past_end_date)
        st.subheader("past Dataframe")
        st.write(past)
        st.subheader("past")
        st.write(past_forecast)

        st.subheader("Distribution of the past Values")
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=past_forecast['ds'], y=past_forecast['yhat'], mode='lines', name='Forecast'))
        fig_forecast.add_trace(go.Scatter(x=past_forecast['ds'], y=past_forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
        fig_forecast.add_trace(go.Scatter(x=past_forecast['ds'], y=past_forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', name='Confidence Interval'))
        fig_forecast.update_layout(title='Prophet Forecast',
                                   xaxis_title='Timestamp',
                                   yaxis_title='Energy Consumption (kWh)',
                                   hovermode='closest',  # Show hover information for the nearest data point
                                   hoverlabel=dict(bgcolor='white', font_size=12),  # Hover label styling
                                   margin=dict(l=50, r=50, t=50, b=50),  # Adjust margin for better layout
                                   width=1300 ) # Set width of the plot)
        st.plotly_chart(fig_forecast)

        fig = go.Figure()

        # Plot actual data
        fig.add_trace(go.Scatter(x=df_hour['ds'], y=df_hour['y'], mode='lines', name='Actual'))

        # Plot forecasted data
        fig.add_trace(go.Scatter(x=past_forecast['ds'], y=past_forecast['yhat'], mode='lines', name='Forecast'))

        # Plot uncertainty intervals
        fig.add_trace(go.Scatter(x=past_forecast['ds'], y=past_forecast['yhat_upper'], mode='lines', fill=None, line=dict(color='gray'), name='Upper Bound'))
        fig.add_trace(go.Scatter(x=past_forecast['ds'], y=past_forecast['yhat_lower'], mode='lines', fill='tonexty', line=dict(color='gray'), name='Lower Bound'))

        # Update layout
        fig.update_layout(title="Prophet Forecast",
                          xaxis_title="Date",
                          yaxis_title="Energy Consumption (kWh)",
                          showlegend=True)
        st.plotly_chart(fig)

        # Display forecasted components
        st.subheader("Forecasted Components")
        st.pyplot(best_model.plot_components(past_forecast))
        # Display forecasted components






def gear(df):
    st.title("Prophet Modellling")
    with st.spinner("Processing..."):
        time.sleep(2)
        
    with st.expander("Seasonality of actual and forecasted  data)"):
        with st.spinner("Processing..."):
            time.sleep(2)  # # Simulate processing time

    # Check if 'level_0' column already exists before resetting the index
    if 'level_0' not in df.columns:
        df.reset_index(inplace=True)
    df.reset_index(inplace=True)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    df_hour = df['Answer Value'].resample('H').agg(['first', 'last'])
    df_hour['Energy Consumption (kWh)'] = df_hour['last'] - df_hour['first']
    non_missing_indices = df_hour['Energy Consumption (kWh)'].notnull()
    missing_indices = ~non_missing_indices
    x = np.arange(len(df_hour.index[non_missing_indices]))
    spline = UnivariateSpline(x, df_hour.loc[non_missing_indices, 'Energy Consumption (kWh)'], k=3)
    interpolated_values = spline(np.arange(len(df_hour)))

    interpolated_values = spline(np.arange(len(df_hour)))
    df_hour['Energy Consumption (kWh)'].loc[missing_indices] = interpolated_values[missing_indices]
    
    
    df_hour.reset_index(inplace=True)
    df_hour.rename(columns={'Timestamp': 'ds', 'Energy Consumption (kWh)': 'y'}, inplace=True)

    # Display loading spinner while performing Prophet forecasting
    with st.spinner("Performing Prophet Forecasting..."):
        # Define an expanded parameter grid with finer granularity
        param_grid = {
            'changepoint_prior_scale': [random.uniform(0.0001, 0.002) for _ in range(100)],
            'seasonality_prior_scale': [random.uniform(0.001, 20.0) for _ in range(100)],
            'holidays_prior_scale': [random.uniform(0.001, 2.0) for _ in range(100)],
            'seasonality_mode': ['additive', 'multiplicative'],
        }
        # Initialize best parameters and best variability
        best_params = {}
        best_variability = float('inf')

        # Increase the number of iterations for random search
        num_iterations = 100

        # Iterate over a random selection of parameter combinations 
        for _ in range(num_iterations):
            params = {k: random.choice(v) for k, v in param_grid.items()}
        
        # Initialize and fit Prophet model with current parameters
        model = Prophet(**params)
        model.fit(df_hour)

        # Make forecasts
        future_start_date = df_hour['ds'].max() + pd.Timedelta(hours=1)
        future_end_date = future_start_date + pd.Timedelta(hours=24*30*2)  # Forecasting for the next two months
        future = pd.DataFrame({'ds': pd.date_range(start=future_start_date, end=future_end_date, freq='H')})
        forecast = model.predict(future)

        # Make forecasts
        past_end_date = df_hour['ds'].min()  # End at the hour before the first timestamp
        past = pd.date_range(end=past_end_date, periods=24*30*2, freq='H')  # Forecasting for the past two months
        past = pd.DataFrame({'ds': past})
        past_forecast = model.predict(past)

        # Calculate variability

        # Update best parameters if the variability is lower
        # Use the best parameters to train the final model
        best_model = Prophet(**best_params)
        best_model.fit(df_hour)

        # Make forecasts with the best model
        forecast = best_model.predict(future)

        st.title("distribution of the forecasted and actual values")


# Create a Plotly figure
    figs = go.Figure()

# Add trace for the forecast
    figs.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))

# Add trace for the past forecast
    figs.add_trace(go.Scatter(x=past_forecast['ds'], y=past_forecast['yhat'], mode='lines', name='Past Forecast'))

# Add trace for the original data frame
    figs.add_trace(go.Scatter(x=df_hour['ds'], y=df_hour['y'], mode='lines', name='Original Data'))

# Set layout
    figs.update_layout(title='Forecast vs Past Forecast vs Original Data',
                  xaxis_title='Date',
                  yaxis_title='Energy Consumption (kWh)')
    st.plotly_chart(figs)
    
    # Extract ds and yhat columns from forecast and past_forecast dataframes
    forecast_subset = forecast[['ds', 'yhat']]
    past_forecast_subset = past_forecast[['ds', 'yhat']]

    forecast_subset = forecast[['ds', 'yhat']]
    past_forecast_subset = past_forecast[['ds', 'yhat']]

# Extract ds and y columns from df dataframe
    df_subset = df_hour[['ds', 'y']]

# Concatenate past_forecast_subset and df_subset, setting y equal to yhat
    merged_df = pd.concat([past_forecast_subset, df_subset.rename(columns={'y': 'yhat'})], axis=0)

# Concatenate merged_df with forecast_subset
    merged_df = pd.concat([merged_df, forecast_subset], axis=0)

# Reset index
    merged_df.reset_index(drop=True, inplace=True)






# Hour of the day plot
    fig_hour_of_day = go.Figure()
    fig_hour_of_day.add_bar(x=merged_df['ds'].dt.hour, y=merged_df['yhat'], marker_color='skyblue',
                        hovertext=merged_df['ds'])
    fig_hour_of_day.update_layout(title='Energy Consumption by Hour of the Day',
                               xaxis_title='Hour of the Day',
                               yaxis_title='Energy Consumption')
    st.plotly_chart(fig_hour_of_day)

# Day of the week plot
    fig_day_of_week = go.Figure()
    fig_day_of_week.add_bar(x=merged_df['ds'].dt.weekday, y=merged_df['yhat'], marker_color='blue',
                        hovertext=merged_df['ds'])
    fig_day_of_week.update_layout(title='Energy Consumption by Day of the Week',
                               xaxis_title='Day of the Week',
                               yaxis_title='Energy Consumption',
                               xaxis=dict(tickmode='array', tickvals=[0, 1, 2, 3, 4, 5, 6],
                                          ticktext=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']))
    st.plotly_chart(fig_day_of_week)

# Day of the month plot
    fig_day_of_month = go.Figure()
    fig_day_of_month.add_bar(x=merged_df['ds'].dt.day, y=merged_df['yhat'], marker_color='salmon',
                         hovertext=merged_df['ds'])
    fig_day_of_month.update_layout(title='Energy Consumption by Day of the Month',
                               xaxis_title='Day of the Month',
                               yaxis_title='Energy Consumption')
    st.plotly_chart(fig_day_of_month)

# Month plot
    fig_month = go.Figure()
    fig_month.add_bar(x=merged_df['ds'].dt.month, y=merged_df['yhat'], marker_color='black',
                  hovertext=merged_df['ds'])
    fig_month.update_layout(title='Energy Consumption by Month',
                        xaxis_title='Month',
                        yaxis_title='Energy Consumption')
    st.plotly_chart(fig_month)

# Weekdays plot
    weekdays_df = merged_df[(merged_df['ds'].dt.weekday >= 0) & (merged_df['ds'].dt.weekday <= 4)]
    fig_weekdays = go.Figure()
    fig_weekdays.add_bar(x=weekdays_df['ds'].dt.weekday, y=weekdays_df['yhat'], marker_color='blue',
                     hovertext=weekdays_df['ds'])
    fig_weekdays.update_layout(title='Energy Consumption on Weekdays',
                           xaxis_title='Weekday',
                           yaxis_title='Energy Consumption',
                           xaxis=dict(tickmode='array', tickvals=[0, 1, 2, 3, 4],
                                      ticktext=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']))
    st.plotly_chart(fig_weekdays)

# Weekend plot
    weekend_df = merged_df[(merged_df['ds'].dt.weekday == 5) | (merged_df['ds'].dt.weekday == 6)]
    fig_weekend = go.Figure()
    fig_weekend.add_bar(x=weekend_df['ds'].dt.weekday, y=weekend_df['yhat'], marker_color='purple',
                    hovertext=weekend_df['ds'])
    fig_weekend.update_layout(title='Energy Consumption on Weekends',
                          xaxis_title='Day of the Week',
                          yaxis_title='Energy Consumption',
                          xaxis=dict(tickmode='array', tickvals=[5, 6],
                                     ticktext=['Saturday', 'Sunday']))
    st.plotly_chart(fig_weekend)

# Sundays plot
    sundays_df = merged_df[merged_df['ds'].dt.weekday == 6]
    fig_sundays = go.Figure()
    fig_sundays.add_bar(x=sundays_df['ds'].dt.date, y=sundays_df['yhat'], marker_color='pink',
                    hovertext=sundays_df['ds'])
    fig_sundays.update_layout(title='Energy Consumption on Sundays',
                          xaxis_title='Date',
                          yaxis_title='Energy Consumption')
    st.plotly_chart(fig_sundays)

# Show the plot

if __name__ == "__main__":
    main()

