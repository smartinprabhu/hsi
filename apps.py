import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.interpolate import UnivariateSpline
import plotly.graph_objects as go
from prophet import Prophet
from datetime import timedelta
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import plotly.tools as tls
from transformers import pipeline
import time
import random
import streamlit.components.v1 as components
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
import streamlit as st
import pandas as pd
import time
import seaborn as sns
# Set page configuration


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

# Apply custom CSS to set the theme to light
st.markdown(
    """
    <style>
        /* Define light theme */
        body {
            background-color: #f0f2f6; /* Set background color to light gray */
            color: #000000; /* Set text color to black */
        }
        .stApp {
            color: #000000; /* Set text color to black */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Add your markdown content
st.markdown("""
<div style="background-color: #f9f9f9; border-radius: 10px; padding: 20px; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
    <h3 style="color: #333; font-family: 'Times New Roman', sans-serif; font-size: 24px; margin-bottom: 15px; text-transform: uppercase; font-weight: bold;">Enhancing Energy Management Through Time Series Forecasting</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("""
for details 👉🏻[description](https://predictionhsi.blogspot.com/2024/03/energy-prediction-requirements.html) 
""")
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

        # Add icons for navigation at the bottom of the page
        st.markdown("---")

        st.write("Click on below to view the summary and description of the dataset uploaded:")
        # Add CSS styles for button animations and transitions

        # Configure icons for each button horizontally


# Configure icons for each button horizontally
        col1, col2, col3, col4 = st.columns(4)

        if col1.button("1️⃣ EDA REPORT "):
            st.session_state.show_data_preparation_button = True
            display_eda_report(df)
            process_and_display_data(df)

        if st.session_state.get("show_data_preparation_button", False):
            if col2.button("2️⃣ DATA PREPARATION"):
                st.session_state.page = "data_preparation"
                st.session_state.show_got_button = True

        if st.session_state.get("show_got_button", False):
            if col3.button("3️⃣ MODELING"):
                st.session_state.page = "got"
                st.session_state.show_export_button = True

        if st.session_state.get("show_export_button", False):
            if col4.button("4️⃣"):
                st.session_state.page = "export"

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
        elif page == "got":
            if 'df' in locals():
                with st.spinner("Processing..."):
                    time.sleep(2)
                    got(df)
        elif page == "export":
            if 'df' in locals():
                with st.spinner("Processing..."):
                    time.sleep(2)


def display_eda_report(df):
    st.title("Exploratory Data Analysis (EDA) Report")

    # Add sections for EDA report
    with st.expander("Exploratory Data Analysis (EDA)"):

        # DataFrame Shape
        st.subheader("Dimension of the uploaded file")

        # Write the number of rows with green color
        st.write(f" <b><i>Number of rows:</i> </b><span style='color:green'>{df.shape[0]}</span>", unsafe_allow_html=True)

        # Write the number of columns with green color
        st.write(f"<b><i>Number of columns:</i></b> <span style='color:green'>{df.shape[1]}</span>", unsafe_allow_html=True)

        # Checkbox to toggle the visibility of the explanation
        # Columns
        st.subheader("Attributes present in the file:")
        st.markdown("- <span style='color:blue'>The Attributes (Columns)</span> present in the uploaded file.", unsafe_allow_html=True)
        st.dataframe(df.columns,width=500)

            # Add an expander with a plus icon
        # Data Types
        st.subheader("Here are the data types of each column :")

        # Provide a detailed explanation of the data types
        st.write("- <span style='color:blue'><b>int64</b></span>: Integer values without any decimal points. "
         "- <span style='color:orange'><b>float64</b></span>: Floating-point values with decimal points. "
         "- <span style='color:green'><b>object</b></span>: Text or string values. "
         "- <span style='color:red'><b>datetime64[ns]</b></span>: Date and time values. "
         "- <span style='color:purple'><b>bool</b></span>: Boolean values, either True or False.", unsafe_allow_html=True)
        st.dataframe(df.dtypes, width=500)



        # Summary Statistics
        st.subheader("Summary Statistics: Descriptive summary of the Numerical Attributes")
        st.markdown("-<span style='color:blue'> **count**:</span> The number of non-null values, -<span style='color:green'> **mean**: </span>The average value, -<span style='color:orange'> **std**:</span> The standard deviation, -<span style='color:red'> **min**:</span> The minimum value, -<span style='color:purple'> **25%**:</span> The value below which 25% of the data falls, -<span style='color:brown'> **50%**:</span> The median value, -<span style='color:gray'> **75%**:</span> The value below which 75% of the data falls, -<span style='color:teal'> **max**:</span> The maximum value", unsafe_allow_html=True)

        st.dataframe(df.describe(),width=500)


        # Missing Values
        st.subheader("Missing Values: Empty,null,NaN")
        st.markdown("""
> - Checking for missing values in the dataset helps to identify any <span style='color:blue'>null or NaN</span> values present in the each Attribute . 
> - This  missing values can affect the accuracy and reliability of the results.
""", unsafe_allow_html=True)
        st.dataframe(df.isnull().sum(),width=400)
        # DataFrame Head
        st.subheader("Initial Rows:")
        st.markdown("""
> Here's a preview of the few `initial` rows of the DataFrame. 
""", unsafe_allow_html=True)

        st.write(df.head())

        # DataFrame Tail
        st.subheader("End rows :")
        st.markdown("""
> Here's a preview of the `last` few  rows of the DataFrame. 
""", unsafe_allow_html=True)
        st.write(df.tail())

def process_and_display_data(df):
    st.title(" Data Transformation")
    
    # Add an expander for the processed data section
    with st.expander("Processed Data"):
        # Convert Timestamp to datetime and set as index
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)

        # Resample by hour and calculate energy consumption
# Allow the user to choose the attribute from the DataFrame
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        st.header("Transformed Dataset of Intervals")

        # Allow the user to choose the attribute from the DataFrame
        selected_attribute = st.selectbox("Select Attribute:(target attribute )", numerical_columns)
        st.markdown("""
> - The DataFrame has been transformed into hourly intervals by resampling. Aggregation is performed by taking the `initial` and `final` values for each hour.

>- The `Energy Consumption (kWh)` attribute represents the difference between the `initial` and `final` values for each hour. This calculation provides an estimate of the energy consumed during that hour.
""", unsafe_allow_html=True)

        # Resample by hour and calculate energy consumption
        df_hour = df[selected_attribute].resample('H').agg(['first', 'last'])
        df_hour['Energy Consumption (kWh)'] = df_hour['last'] - df_hour['first']
        non_missing_indices = df_hour['Energy Consumption (kWh)'].notnull()
        missing_indices = ~non_missing_indices

        # Display processed data and statistics after selecting the column
        st.subheader("Few Rows of the transformed Dataset:")
        st.markdown("""
Here's a preview of the few  rows of the Transformed DataFrame. 
""", unsafe_allow_html=True)
        st.dataframe(df_hour,width=500)
        # Show basic statistics
        st.subheader("Descriptive Summary for the Transformed data")
        st.markdown("-<span style='color:blue'> **count**:</span> The number of non-null values, -<span style='color:green'> **mean**: </span>The average value, -<span style='color:orange'> **std**:</span> The standard deviation, -<span style='color:red'> **min**:</span> The minimum value, -<span style='color:purple'> **25%**:</span> The value below which 25% of the data falls, -<span style='color:brown'> **50%**:</span> The median value, -<span style='color:gray'> **75%**:</span> The value below which 75% of the data falls, -<span style='color:teal'> **max**:</span> The maximum value", unsafe_allow_html=True)

        st.write(df_hour.describe())


        # Show the first few rows of the dataset


        
        
        st.subheader("Records having missing values")
        st.markdown("""
       > - The transformed DataFrame may contain records with missing values, denoted as '`NaN`', '`Null`', or '`empty cells`'. These missing values can occur due to various reasons such as data collection errors or equipment malfunction.


        """)
        st.write(df_hour[df_hour['Energy Consumption (kWh)'].isnull()])

        st.header(" Missing indices")
        
        st.markdown("""
> - The sum of missing indices represents the total number of records with missing values in the transformed DataFrame. 
> - missing value indices are records with `missing values`, and non missing indices are the records with `non missing values`

""")
        records_data = {
            'Missing Value Records': [missing_indices.sum()],
            'Non-Missing Value Records': [non_missing_indices.sum()]
        }
        records_df = pd.DataFrame(records_data)

        # Display the DataFrame
        st.write(records_df)
        # Plot distribution of energy consumption over hourly intervals using Plotly
        st.header("Hourly Energy Consumption Distribution")
        st.markdown("""
>>  This interactive plot visualizes the hourly distribution of `Energy consumption (in kWh)` over time.
 Each data point represents the energy consumed for a specific hour interval.

- **X-axis:** Timestamp (Hourly Intervals)
- **Y-axis:** Energy Consumption (kWh)
""")
        fig = go.Figure()

        # Add scatter plot
# Add scatter plot with customized hover template
        fig.add_trace(go.Scatter(x=df_hour.index, y=df_hour['Energy Consumption (kWh)'],
                                mode='lines', name='Energy Consumption (kWh)',
                                hovertemplate='<b>Time:</b> %{x}<br><b>Energy Consumption:</b> %{y} kWh<br><extra></extra>'))

        # Update layout with detailed configurations
        fig.update_layout(
            title='Hourly Energy Consumption Distribution',  # Add title
            xaxis=dict(title='Time (Hourly Intervals)'),  # X-axis title
            yaxis=dict(title='Energy Consumption (kWh)'),  # Y-axis title
            hovermode='closest',  # Show hover information for the nearest data point
            hoverlabel=dict(bgcolor='white', font_size=12, font_color='black'),  # Hover label styling
            margin=dict(l=50, r=50, t=50, b=50),  # Adjust margin for better layout
            width=1700 ,
            height=700 # Set width of the plot
        )

        # Show the plot
        st.plotly_chart(fig)

def data_preparation(df):
  st.title("Handling missing values ")
    
    # Add an expander for the processed data section
  with st.expander("Processed Data"):
        # Convert Timestamp to datetime and set as index
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)

        # Resample by hour and calculate energy consumption
# Allow the user to choose the attribute from the DataFrame


        # Resample by hour and calculate energy consumption
    df_hour = df['Answer Value'].resample('H').agg(['first', 'last'])
    df_hour['Energy Consumption (kWh)'] = df_hour['last'] - df_hour['first']
    non_missing_indices = df_hour['Energy Consumption (kWh)'].notnull()
    missing_indices = ~non_missing_indices
    x = np.arange(len(df_hour.index[non_missing_indices]))
    spline = UnivariateSpline(x, df_hour.loc[non_missing_indices, 'Energy Consumption (kWh)'], k=3)
    interpolated_values = spline(np.arange(len(df_hour)))

    
    m=df_hour[df_hour['Energy Consumption (kWh)'].isnull()]
    st.markdown("""
    > - This involves identifying and replacing missing values within the dataset. Our approach is to generate synthetic data to replace these missing values.
""")

    st.markdown("""
>## Records having missing values
> *The DataFrame already contains records with missing values.And thus extracted records having missing values are given below*.
""")
    st.write(m[['Energy Consumption (kWh)']].style.set_properties(**{'width': '500px'}))

        

    # Display synthetic data points used to replace missing values
    st.markdown("""
>##  Synthetic Data Points Used to Replace Missing Values:  . 
> - It is possible that the missing values in the energy consumption  could mean that the power consumed  during those periods were not recorded or omitted.
>> - As we can see in some records there are consecutive missing values  .
>>-  so in those places I'm going to use `synthetic data` by using [univariate spline interpolation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.InterpolatedUnivariateSpline.html).
>>- synthetic data  is artificial , it is not originally present in the  dataframe
> -  The records where` synthetic data` points replaced missing values are provided below:

""")
    synthetic_data_points = pd.DataFrame({
        'Timestamp': df_hour.index[missing_indices],
        'Synthetic Energy Consumption (kWh)': interpolated_values[missing_indices]
    })
    st.write(synthetic_data_points, width=500)
    # Display the DataFrame with missing values filled
    
    st.subheader("DataFrame with Synthetic Data:")
    
    st.markdown("""
*This processed dataframe  aims to provide a more complete and continuous representation of energy consumption patterns.*
""")

    st.write(df_hour.style.set_properties(**{'width': '500px'}))
    
    
    
    df_hour['Energy Consumption (kWh)'].loc[missing_indices] = interpolated_values[missing_indices]
    # Plot the original data with missing values indicated in red
    
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_hour.index, y=df_hour['Energy Consumption (kWh)'], mode='lines', name='Original Data',
                             hovertemplate='<b>Time:</b> %{x}<br><b>Energy Consumption:</b> %{y} kWh<br><extra></extra>'))  # Set hover color for non-missing values to blue
    fig.add_trace(go.Scatter(x=df_hour.index[missing_indices], y=df_hour['Energy Consumption (kWh)'][missing_indices],
                             mode='markers', marker=dict(color='red'), name='synthetic data Values',
                             hovertemplate='<b>Time:</b> %{x}<br><b>Energy Consumption:</b> %{y} kWh <br>(<span style="color:red">Synthetic</span>)<br><extra></extra>'))

    # Update layout for the plot
    fig.update_layout(title='Energy Consumption (kWh)', xaxis_title='Timestamp', yaxis_title='Energy Consumption (kWh)',
                      hovermode='closest',  # Show hover information for the nearest data point
                      hoverlabel=dict(bgcolor='white', font_size=12,font_color='black'),  # Hover label styling
                      margin=dict(l=50, r=50, t=50, b=50),  # Adjust margin for better layout
                      width=1600,
                      height=700# Set width of the plot
                      )
    st.subheader("Original Data with Missing Values")
    st.plotly_chart(fig)

def got(df):
  st.title("Forecating ( Prophet )")
  with st.spinner("Processing..."):
          time.sleep(2)
  with st.expander("perform Forecasting"):
   with st.spinner("Processing..."):
    time.sleep(2)   # Simulate processing time

    # Check if 'level_0' column already exists before resetting the index
   # if 'level_0' not in df.columns:
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

        # Get user input for future forecasting
        future_months = st.number_input("Enter the number of months for future forecasting:", min_value=1, value=2, step=1)

        # Optionally, offer past forecasting
        past_checkbox = st.checkbox("Perform Past Forecasting")
        if past_checkbox:
            past_months = st.number_input("Enter the number of months for past forecasting:", min_value=1, value=2, step=1)
            # Make forecasts for future
            future_start_date = df_hour['ds'].max() + pd.Timedelta(hours=1)
            future_end_date = future_start_date + pd.Timedelta(hours=24*30*future_months)  # Forecasting for the specified number of months
            future = pd.DataFrame({'ds': pd.date_range(start=future_start_date, end=future_end_date, freq='H')})
            future_forecast = model.predict(future)
            print(future_forecast['yhat_upper'] - future_forecast['yhat_lower'])
            print((future_forecast['yhat_upper'] - future_forecast['yhat_lower']).mean())
            past_end_date = df_hour['ds'].min()  # End at the hour before the first timestamp
            past_start_date = past_end_date - pd.Timedelta(hours=24*30*past_months)  # Forecasting for the specified number of months
            past = pd.date_range(start=past_start_date, end=past_end_date, freq='H')
            past = pd.DataFrame({'ds': past})
            past_forecast = model.predict(past)

            st.write("Forecasted from :", "<span style='color:red'>{}</span> to <span style='color:red'>{}</span>".format(past_forecast['ds'].min().strftime("%Y-%m-%d %H:%M:%S"), future_forecast['ds'].max().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

            st.markdown("""
> # Forecasting I 
                      """  )
            st.markdown("""
    - `ds`: represents the timestamps or dates for which the predictions are made.`trend`: represents the overall direction of the data over time. It captures the long-term behavior of the time series data.`yhat_lower`represents the lower bound of the predicted values (confidence interval lower bound).
`yhat_upper`: represents the upper bound of the predicted values (confidence interval upper bound).
`trend_lower`: represents the lower bound of the trend.
`trend_upper`: represents the upper bound of the trend.`additive_terms`:  additional components added to the trend, such as seasonality and holidays.`additive_terms_lower`: Lower bound of the additive terms.`additive_terms_upper`: Upper bound of the additive terms.`daily`:  the daily seasonality pattern in the data.`daily_lower`: Lower bound of the daily component.`daily_upper`: Upper bound of the daily component.`weekly`: represents the weekly seasonality pattern in the data.
`weekly_lower`: Lower bound of the weekly component.`weekly_upper`: Upper bound of the weekly component.
`multiplicative_terms`: are additional components multiplied with the trend, such as holidays.`multiplicative_terms_lower`: Lower bound of the multiplicative terms.`multiplicative_terms_upper`: Upper bound of the multiplicative terms.
`yhat`: represents the predicted values or forecasted values for the corresponding timestamps.
""")
            st.write("From :", "<span style='color:red'>{}</span> to <span style='color:red'>{}</span>".format(future_forecast['ds'].min().strftime("%Y-%m-%d "), future_forecast['ds'].max().strftime("%Y-%m-%d ")), unsafe_allow_html=True)

            st.write(future_forecast)

            st.subheader("Distribution of the  Forecasted values :")
            st.write("Forecasted from :", "<span style='color:red'>{}</span> to <span style='color:red'>{}</span>".format(future_forecast['ds'].min().strftime("%Y-%m-%d %H:%M:%S"), future_forecast['ds'].max().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

            fig_future = go.Figure()
            fig_future.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'], mode='lines', name='Forecast'))
            fig_future.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
            fig_future.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', name='Confidence Interval'))
            fig_future.update_layout(title=' Distribution of the  Forecasted values',
                                     xaxis_title='Timestamp',
                                     yaxis_title='Energy Consumption (kWh)',
                                     hovermode='closest',  # Show hover information for the nearest data point
                                     hoverlabel=dict(bgcolor='white', font_size=12,font_color='black'),  # Hover label styling
                                     margin=dict(l=50, r=50, t=50, b=50),  # Adjust margin for better layout
                                     width=1600,
                                     height=500) # Set width of the plot)



            st.markdown("""
- **Forecast Line**: Represents the predicted values` (yhat)` for future timestamps `(ds)`. 
- **Confidence Interval**: The shaded area between the `upper` and `lower `bounds of the confidence interval represents the` uncertainty associated with the forecast.` It indicates the range within which the actual values are likely to fall based on the model's confidence level.
""")
        
            st.plotly_chart(fig_future)
            st.subheader("Forecasted Components :")

            st.markdown("""
> The  Prophet model generates a set of subplots showing the trend and various components (e.g., weekly, daily, monthly) of the forecasted data.
- **`Trend`**: Represents the overall direction of the data over time.**`Seasonalities`**: Capture recurring patterns such as weekly, daily, and monthly seasonality.
The plot provides insights into how different components contribute to the overall forecasted values.
""")
            # Set the figure size parameters using Seaborn
            sns.set(rc={'figure.figsize':(6, 6)})  # Set the width to 10 inches and height to 8 inches

            # Generate the plot
            fig_past_comp = model.plot_components(future_forecast)

            # Display the plot
            st.pyplot(fig_past_comp, clear_figure=True)


            st.markdown("""
> # Forecasting II 
                      """  )

            st.write("Forecasting from :", "<span style='color:red'>{}</span> to <span style='color:red'>{}</span>".format(past_forecast['ds'].min().strftime("%Y-%m-%d "), past_forecast['ds'].max().strftime("%Y-%m-%d")), unsafe_allow_html=True)


            st.subheader("Forecasting for Historical(past) Time periods")
            
            st.markdown(""" 
 - This consists of the   data points  which are forecasted for teh historical(past) timeperiods.
 - The forecasted data   has the parameters : `ds` ,`trend`, `yhat_lower` ,`yhat_upper`,
`trend_lower`,`trend_upper`, .`additive_terms` , `additive_terms_lower` ,`additive_terms_upper` ,`daily`  ,`daily_lower` ,`daily_upper` .`weekly`,`weekly_lower` ,`weekly_upper`,`multiplicative_terms`,`multiplicative_terms_lower`,`multiplicative_terms_upper`,`yhat`
""")
            st.write("Forecasted from :", "<span style='color:red'>{}</span> to <span style='color:red'>{}</span>".format(past_forecast['ds'].min().strftime("%Y-%m-%d %H:%M:%S"), past_forecast['ds'].max().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

            st.write(past_forecast)
            # Plot future forecast

            
                # Print uncertainty and variability with the best model
            print(past_forecast['yhat_upper'] - past_forecast['yhat_lower'])
            print((past_forecast['yhat_upper'] - past_forecast['yhat_lower']).mean())

                # Display past forecast

            st.subheader("Distribution of the  Forecasted values :")

            st.markdown("""
- **Forecast Line**: Represents the predicted values (`yhat`) for historical (past) timestamps (`ds`). These values are the model's estimates of the energy consumption at each timestamp.
- **Confidence Interval**: The shaded area between the upper and lower bounds of the confidence interval represents the uncertainty associated with the forecast. It indicates the range within which the actual values are likely to fall based on the model's confidence level. 
In other words, the actual energy consumption values are expected to fall within this range around the predicted values (`yhat`), providing insights into the reliability of the forecast.

""")
            fig_past = go.Figure()
            fig_past.add_trace(go.Scatter(x=past_forecast['ds'], y=past_forecast['yhat'], mode='lines', name='Forecast'))
            fig_past.add_trace(go.Scatter(x=past_forecast['ds'], y=past_forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
            fig_past.add_trace(go.Scatter(x=past_forecast['ds'], y=past_forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', name='Confidence Interval'))
            fig_past.update_layout(title=' Distribution of the  Forecasted values',
                                     xaxis_title='Timestamp',
                                     yaxis_title='Energy Consumption (kWh)',
                                     hovermode='closest',  # Show hover information for the nearest data point
                                     hoverlabel=dict(bgcolor='white', font_size=12,font_color='black'),  # Hover label styling
                                     margin=dict(l=50, r=50, t=50, b=50),  # Adjust margin for better layout
                                     width=1600,
                                     height=500) # Set width of the plot)
            st.write("Forecasted from :", "<span style='color:red'>{}</span> to <span style='color:red'>{}</span>".format(past_forecast['ds'].min().strftime("%Y-%m-%d %H:%M:%S"), past_forecast['ds'].max().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

            st.plotly_chart(fig_past)
            st.subheader("Forecasted components")

            st.markdown("""
> The  Prophet model generates a set of subplots showing the trend and various components (e.g., weekly, daily, monthly) of the forecasted data.
- **`Trend`**: Represents the overall direction of the data over time.**`Seasonalities`**: Capture recurring patterns such as weekly, daily, and monthly seasonality.
The plot provides insights into how different components contribute to the overall forecasted values.
""")
            # Set the figure size parameters using Seaborn
            sns.set(rc={'figure.figsize':(6, 6)})  # Set the width to 10 inches and height to 8 inches

            # Generate the plot
            fig_past_comp = model.plot_components(past_forecast)
            
            
            # Display the plot
            st.pyplot(fig_past_comp, clear_figure=True)

            st.subheader("Distribution of teh Actual and Forecasted Values ")
            st.markdown("""
            - **Forecast**:  These values are predictions made by the forecasting model for future energy consumption.
            - **Past Forecast**: . These values are predictions made by the forecasting model for past energy consumption.
            - **Original Data**:  These are the observed values of energy consumption over time.
            - The plot provides a comparison between the forecasted values, past forecasted values, and the actual observed values. It helps in assessing the accuracy of the forecasting model by comparing its predictions with the actual data.
            """)
# Create a Plotly figure
            figs = go.Figure()

# Add trace for the forecast
            figs.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'], mode='lines', name='Forecast'))

# Add trace for the past forecast
            figs.add_trace(go.Scatter(x=past_forecast['ds'], y=past_forecast['yhat'], mode='lines', name='Past Forecast'))

# Add trace for the original data frame
            figs.add_trace(go.Scatter(x=df_hour['ds'], y=df_hour['y'], mode='lines', name='Original Data'))

# Set layout
            figs.update_traces(hovertemplate='<b>Date:</b> %{x}<br><b>Time:</b> %{x|%H:%M:%S}<br><b>Energy Consumed:</b> %{y} kWh<br><extra></extra>')

                # Adjust layout
            figs.update_layout(title='Actual and Forecasted Values',
                                xaxis_title='Timestamp',
                                yaxis_title='Energy Consumption (kWh)',
                                hovermode='closest',
                                hoverlabel=dict(bgcolor='white', font_size=12, font_color='black'),
                                margin=dict(l=50, r=50, t=50, b=50),
                                width=1700,
                                height=800)
                            
            st.plotly_chart(figs)
            
            df_subset = df_hour[['ds', 'y']]
            forecast_subset = future_forecast[['ds', 'yhat']]
            past_forecast_subset = past_forecast[['ds', 'yhat']]
            # Merge past, future, and original dataframes
            merged_df = pd.concat([past_forecast_subset, df_subset.rename(columns={'y': 'yhat'})], axis=0)
            merged_df = pd.concat([merged_df, forecast_subset], axis=0)
            st.markdown("""
- **Energy Consumption by Hour of the Day (Regressor Component)**: This bar plot shows the variation in energy consumption throughout the day. 
- It represents one of the regressor components used in the forecasting model, capturing the hourly patterns of energy usage.
""")

            # Assuming merged_df['ds'] contains datetime objects

            # Extract date and time separately

            # Extract date and time separately
            merged_df['date'] = merged_df['ds'].dt.strftime('%Y-%m-%d')
            merged_df['time'] = merged_df['ds'].dt.strftime('%H:%M:%S')

            # Create hover text with labels
            hover_text = ('Date: ' + merged_df['date'] + '<br>' +
                        'Time: ' + merged_df['time'] + '<br>' +
                        'Energy: ' + merged_df['yhat'].astype(str))

            # Create the bar chart
            fig_hour_of_day = go.Figure()
            fig_hour_of_day.add_bar(x=merged_df['ds'].dt.hour, 
                                    y=merged_df['yhat'], 
                                    marker_color='skyblue',
                                    hovertext=hover_text)

            # Update layout
            fig_hour_of_day.update_layout(title='Energy Consumption by Hour of the Day',
                                        xaxis_title='Hour of the Day',
                                        yaxis_title='Energy Consumption',
                                        height=600,
                                        width=1500)


            st.plotly_chart(fig_hour_of_day)
            # Hour of the day plot
            st.markdown("""
- **Energy Consumption by Day of the Week (Regressor Component)**: This bar plot illustrates the energy consumption patterns across different days of the week. 
- It serves as another regressor component, providing insights into the weekly variations in energy usage.
""")

            # Create hover text with labels
            hover_text = ('Date: ' + merged_df['date'] + '<br>' +
                        'Time: ' + merged_df['time'] + '<br>' +
                        'Energy: ' + merged_df['yhat'].astype(str) + ' kWh')

            # Create the bar chart
            fig_day_of_week = go.Figure()
            fig_day_of_week.add_bar(x=merged_df['ds'].dt.weekday, 
                                    y=merged_df['yhat'], 
                                    marker_color='blue',
                                    hovertext=hover_text)

            # Update layout
            fig_day_of_week.update_layout(title='Energy Consumption by Day of the Week',
                                        xaxis_title='Day of the Week',
                                        yaxis_title='Energy Consumption',
                                        height=600,
                                        width=1500,
                                        xaxis=dict(tickmode='array', tickvals=[0, 1, 2, 3, 4, 5, 6],
                                                    ticktext=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']))

            # Show the figure
            st.plotly_chart(fig_day_of_week)
            st.markdown("""
            - **Energy Consumption by Day of the Month (Regressor Component)**: This bar plot displays the energy consumption trends over the days of the month. 
            - As a regressor component, it helps in understanding the monthly variations in energy usage.
            """)
            # Day of the month plot
            
            # Create the bar chart
            fig_day_of_month = go.Figure()
            fig_day_of_month.add_bar(x=merged_df['ds'].dt.day, 
                                    y=merged_df['yhat'], 
                                    marker_color='salmon',
                                    hovertext=hover_text)

            # Update layout
            fig_day_of_month.update_layout(title='Energy Consumption by Day of the Month',
                                        xaxis_title='Day of the Month',
                                        yaxis_title='Energy Consumption',
                                        width=1500,
                                        height=600)
            st.plotly_chart(fig_day_of_month)

            st.markdown("""
- **Energy Consumption by Month (Regressor Component)**: This bar plot showcases the energy consumption patterns across different months of the year.
- It serves as a regressor component, capturing the seasonal variations in energy usage.
""")
            # Month plot
            fig_month = go.Figure()
            fig_month.add_bar(x=merged_df['ds'].dt.month, 
                            y=merged_df['yhat'], 
                            marker_color='black',
                            hovertext=hover_text)

            # Update layout
            fig_month.update_layout(title='Energy Consumption by Month',
                                    xaxis_title='Month',
                                    yaxis_title='Energy Consumption',
                                    width=1500,
                                    height=600)

            # Show the figure
            st.plotly_chart(fig_month)
            st.markdown("""
- **Energy Consumption on Weekdays (Regressor Component)**: This bar plot focuses specifically on energy consumption during weekdays (Monday to Friday)
- It serves as a regressor component to capture the differences in energy usage between weekdays and weekends.
            """)

            fig_weekdays = go.Figure()
            weekdays_df = merged_df[(merged_df['ds'].dt.weekday >= 0) & (merged_df['ds'].dt.weekday <= 4)]
            sundays_df = merged_df[merged_df['ds'].dt.weekday == 6]

            weekend_df = merged_df[(merged_df['ds'].dt.weekday == 5) | (merged_df['ds'].dt.weekday == 6)]
            # Extract date and time separately
            weekdays_df['date'] = weekdays_df['ds'].dt.strftime('%Y-%m-%d')
            weekdays_df['time'] = weekdays_df['ds'].dt.strftime('%H:%M:%S')

            # Create hover text with labels
            hover_text = ('Date: ' + weekdays_df['date'] + '<br>' +
                        'Time: ' + weekdays_df['time'] + '<br>' +
                        'Energy: ' + weekdays_df['yhat'].astype(str) + ' kWh')

            # Create the bar chart

            fig_weekdays.add_bar(x=weekdays_df['ds'].dt.weekday, 
                                y=weekdays_df['yhat'], 
                                marker_color='blue',
                                hovertext=hover_text)

            # Update layout
            fig_weekdays.update_layout(title='Energy Consumption on Weekdays',
                                    xaxis_title='Weekday',
                                    yaxis_title='Energy Consumption',
                                    height=600,
                                    width=1500,
                                    xaxis=dict(tickmode='array', tickvals=[0, 1, 2, 3, 4],
                                                ticktext=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']))

            # Show the figure
            st.plotly_chart(fig_weekdays)

            st.markdown("""
- **Energy Consumption on Weekends (Regressor Component)**: This bar plot focuses specifically on energy consumption during weekends (Saturday and Sunday).
- It serves as a regressor component to capture the differences in energy usage between weekdays and weekends.
                        """)

            fig_weekend = go.Figure()
            # Extract date and time separately
            weekend_df['date'] = weekend_df['ds'].dt.strftime('%Y-%m-%d')
            weekend_df['time'] = weekend_df['ds'].dt.strftime('%H:%M:%S')

            # Create hover text with labels
            hover_text = ('Date: ' + weekend_df['date'] + '<br>' +
                        'Time: ' + weekend_df['time'] + '<br>' +
                        'Energy: ' + weekend_df['yhat'].astype(str) + ' kWh')

            # Create the bar chart

            fig_weekend.add_bar(x=weekend_df['ds'].dt.weekday, 
                                y=weekend_df['yhat'], 
                                marker_color='purple',
                                hovertext=hover_text)

            # Update hover template

            # Update layout
            fig_weekend.update_layout(title='Energy Consumption on Weekends',
                                    xaxis_title='Day of the Week',
                                    yaxis_title='Energy Consumption',
                                    height=600,
                                    width=1500,
                                    xaxis=dict(tickmode='array', tickvals=[5, 6],
                                                ticktext=['Saturday', 'Sunday']))

            # Show the figure
            st.plotly_chart(fig_weekend)

            st.markdown("""
- **Energy Consumption on Sundays (Regressor Component)**: This bar plot zooms in on energy consumption specifically on Sundays, providing detailed insights into energy usage patterns on Sundays compared to other days of the week.
- It serves as a regressor component to capture the unique patterns of energy usage on Sundays.
                        """)


            # Extract date and time separately
            sundays_df['date'] = sundays_df['ds'].dt.strftime('%Y-%m-%d')
            sundays_df['time'] = sundays_df['ds'].dt.strftime('%H:%M:%S')

            # Create hover text with labels
            hover_text = ('Date: ' + sundays_df['date'] + '<br>' +
                        'Time: ' + sundays_df['time'] + '<br>' +
                        'Energy: ' + sundays_df['yhat'].astype(str) + ' kWh')

            # Create the bar chart
            fig_sundays = go.Figure()
            fig_sundays.add_bar(x=sundays_df['ds'].dt.date, 
                                y=sundays_df['yhat'], 
                                marker_color='pink',
                                hovertext=hover_text)

            # Update layout
            fig_sundays.update_layout(title='Energy Consumption on Sundays',
                                    xaxis_title='Date',
                                    yaxis_title='Energy Consumption',
                                    height=600,
                                    width=1500)

            # Show the figure
            st.plotly_chart(fig_sundays)



        else:
            # Merge future and original dataframes
            future_start_date = df_hour['ds'].max() + pd.Timedelta(hours=1)
            future_end_date = future_start_date + pd.Timedelta(hours=24*30*future_months)  # Forecasting for the specified number of months
            future = pd.DataFrame({'ds': pd.date_range(start=future_start_date, end=future_end_date, freq='H')})
            future_forecast = model.predict(future)
            forecast_subset = future_forecast[['ds', 'yhat']]
            df_subset = df_hour[['ds', 'y']]
            merged_df = pd.concat([forecast_subset, df_subset.rename(columns={'y': 'yhat'})], axis=0)

            print(future_forecast['yhat_upper'] - future_forecast['yhat_lower'])
            print((future_forecast['yhat_upper'] - future_forecast['yhat_lower']).mean())

            st.write("Forecasted from :", "<span style='color:red'>{}</span> to <span style='color:red'>{}</span>".format(future_forecast['ds'].min().strftime("%Y-%m-%d %H:%M:%S"), future_forecast['ds'].max().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

            st.markdown("""
> # Forecasting 
                      """  )
            st.markdown("""
    - `ds`: represents the timestamps or dates for which the predictions are made.`trend`: represents the overall direction of the data over time. It captures the long-term behavior of the time series data.`yhat_lower`represents the lower bound of the predicted values (confidence interval lower bound).
`yhat_upper`: represents the upper bound of the predicted values (confidence interval upper bound).
`trend_lower`: represents the lower bound of the trend.
`trend_upper`: represents the upper bound of the trend.`additive_terms`:  additional components added to the trend, such as seasonality and holidays.`additive_terms_lower`: Lower bound of the additive terms.`additive_terms_upper`: Upper bound of the additive terms.`daily`:  the daily seasonality pattern in the data.`daily_lower`: Lower bound of the daily component.`daily_upper`: Upper bound of the daily component.`weekly`: represents the weekly seasonality pattern in the data.
`weekly_lower`: Lower bound of the weekly component.`weekly_upper`: Upper bound of the weekly component.
`multiplicative_terms`: are additional components multiplied with the trend, such as holidays.`multiplicative_terms_lower`: Lower bound of the multiplicative terms.`multiplicative_terms_upper`: Upper bound of the multiplicative terms.
`yhat`: represents the predicted values or forecasted values for the corresponding timestamps.
""")
            st.write("From :", "<span style='color:red'>{}</span> to <span style='color:red'>{}</span>".format(future_forecast['ds'].min().strftime("%Y-%m-%d "), future_forecast['ds'].max().strftime("%Y-%m-%d ")), unsafe_allow_html=True)

            st.write(future_forecast)

            st.subheader("Distribution of the  Forecasted values :")
            st.write("Forecasted from :", "<span style='color:red'>{}</span> to <span style='color:red'>{}</span>".format(future_forecast['ds'].min().strftime("%Y-%m-%d %H:%M:%S"), future_forecast['ds'].max().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

            fig_future = go.Figure()
            fig_future.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'], mode='lines', name='Forecast'))
            fig_future.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
            fig_future.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', name='Confidence Interval'))
            fig_future.update_layout(title=' Distribution of the  Forecasted values',
                                     xaxis_title='Timestamp',
                                     yaxis_title='Energy Consumption (kWh)',
                                     hovermode='closest',  # Show hover information for the nearest data point
                                     hoverlabel=dict(bgcolor='white', font_size=12,font_color='black'),  # Hover label styling
                                     margin=dict(l=50, r=50, t=50, b=50),  # Adjust margin for better layout
                                     width=1600,
                                     height=500) # Set width of the plot)



            st.markdown("""
- **Forecast Line**: Represents the predicted values` (yhat)` for future timestamps `(ds)`. 
- **Confidence Interval**: The shaded area between the `upper` and `lower `bounds of the confidence interval represents the` uncertainty associated with the forecast.` It indicates the range within which the actual values are likely to fall based on the model's confidence level.
""")
        
            st.plotly_chart(fig_future)
            st.subheader("Forecasted Components :")

            st.markdown("""
> The  Prophet model generates a set of subplots showing the trend and various components (e.g., weekly, daily, monthly) of the forecasted data.
- **`Trend`**: Represents the overall direction of the data over time.**`Seasonalities`**: Capture recurring patterns such as weekly, daily, and monthly seasonality.
The plot provides insights into how different components contribute to the overall forecasted values.
""")
            # Set the figure size parameters using Seaborn
            sns.set(rc={'figure.figsize':(6, 6)})  # Set the width to 10 inches and height to 8 inches

            # Generate the plot
            fig_past_comp = model.plot_components(future_forecast)

            # Display the plot
            st.pyplot(fig_past_comp, clear_figure=True)
            st.subheader("Distribution of teh Actual and Forecasted Values ")
            st.markdown("""
            - **Forecast**:  These values are predictions made by the forecasting model for future energy consumption.
            - **Past Forecast**: . These values are predictions made by the forecasting model for past energy consumption.
            - **Original Data**:  These are the observed values of energy consumption over time.
            - The plot provides a comparison between the forecasted values, past forecasted values, and the actual observed values. It helps in assessing the accuracy of the forecasting model by comparing its predictions with the actual data.
            """)
# Create a Plotly figure
            figs = go.Figure()

# Add trace for the forecast
            figs.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'], mode='lines', name='Forecast'))

# Add trace for the past forecast

# Add trace for the original data frame
            figs.add_trace(go.Scatter(x=df_hour['ds'], y=df_hour['y'], mode='lines', name='Original Data'))

# Set layout
            figs.update_traces(hovertemplate='<b>Date:</b> %{x}<br><b>Time:</b> %{x|%H:%M:%S}<br><b>Energy Consumed:</b> %{y} kWh<br><extra></extra>')

                # Adjust layout
            figs.update_layout(title='Actual and Forecasted Values',
                                xaxis_title='Timestamp',
                                yaxis_title='Energy Consumption (kWh)',
                                hovermode='closest',
                                hoverlabel=dict(bgcolor='white', font_size=12, font_color='black'),
                                margin=dict(l=50, r=50, t=50, b=50),
                                width=1700,
                                height=800)
                            
            st.plotly_chart(figs)
            
            df_subset = df_hour[['ds', 'y']]
            forecast_subset = future_forecast[['ds', 'yhat']]
            # Merge past, future, and original dataframes
            merged_df = pd.concat([forecast_subset, df_subset.rename(columns={'y': 'yhat'})], axis=0)
            st.markdown("""
- **Energy Consumption by Hour of the Day (Regressor Component)**: This bar plot shows the variation in energy consumption throughout the day. 
- It represents one of the regressor components used in the forecasting model, capturing the hourly patterns of energy usage.
""")

            # Assuming merged_df['ds'] contains datetime objects

            # Extract date and time separately

            # Extract date and time separately
            merged_df['date'] = merged_df['ds'].dt.strftime('%Y-%m-%d')
            merged_df['time'] = merged_df['ds'].dt.strftime('%H:%M:%S')

            # Create hover text with labels
            hover_text = ('Date: ' + merged_df['date'] + '<br>' +
                        'Time: ' + merged_df['time'] + '<br>' +
                        'Energy: ' + merged_df['yhat'].astype(str))

            # Create the bar chart
            fig_hour_of_day = go.Figure()
            fig_hour_of_day.add_bar(x=merged_df['ds'].dt.hour, 
                                    y=merged_df['yhat'], 
                                    marker_color='skyblue',
                                    hovertext=hover_text)

            # Update layout
            fig_hour_of_day.update_layout(title='Energy Consumption by Hour of the Day',
                                        xaxis_title='Hour of the Day',
                                        yaxis_title='Energy Consumption',
                                        height=600,
                                        width=1500)


            st.plotly_chart(fig_hour_of_day)
            # Hour of the day plot
            st.markdown("""
- **Energy Consumption by Day of the Week (Regressor Component)**: This bar plot illustrates the energy consumption patterns across different days of the week. 
- It serves as another regressor component, providing insights into the weekly variations in energy usage.
""")

            # Create hover text with labels
            hover_text = ('Date: ' + merged_df['date'] + '<br>' +
                        'Time: ' + merged_df['time'] + '<br>' +
                        'Energy: ' + merged_df['yhat'].astype(str) + ' kWh')

            # Create the bar chart
            fig_day_of_week = go.Figure()
            fig_day_of_week.add_bar(x=merged_df['ds'].dt.weekday, 
                                    y=merged_df['yhat'], 
                                    marker_color='blue',
                                    hovertext=hover_text)

            # Update layout
            fig_day_of_week.update_layout(title='Energy Consumption by Day of the Week',
                                        xaxis_title='Day of the Week',
                                        yaxis_title='Energy Consumption',
                                        height=600,
                                        width=1500,
                                        xaxis=dict(tickmode='array', tickvals=[0, 1, 2, 3, 4, 5, 6],
                                                    ticktext=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']))

            # Show the figure
            st.plotly_chart(fig_day_of_week)
            st.markdown("""
            - **Energy Consumption by Day of the Month (Regressor Component)**: This bar plot displays the energy consumption trends over the days of the month. 
            - As a regressor component, it helps in understanding the monthly variations in energy usage.
            """)
            # Day of the month plot
            
            # Create the bar chart
            fig_day_of_month = go.Figure()
            fig_day_of_month.add_bar(x=merged_df['ds'].dt.day, 
                                    y=merged_df['yhat'], 
                                    marker_color='salmon',
                                    hovertext=hover_text)

            # Update layout
            fig_day_of_month.update_layout(title='Energy Consumption by Day of the Month',
                                        xaxis_title='Day of the Month',
                                        yaxis_title='Energy Consumption',
                                        width=1500,
                                        height=600)
            st.plotly_chart(fig_day_of_month)

            st.markdown("""
- **Energy Consumption by Month (Regressor Component)**: This bar plot showcases the energy consumption patterns across different months of the year.
- It serves as a regressor component, capturing the seasonal variations in energy usage.
""")
            # Month plot
            fig_month = go.Figure()
            fig_month.add_bar(x=merged_df['ds'].dt.month, 
                            y=merged_df['yhat'], 
                            marker_color='black',
                            hovertext=hover_text)

            # Update layout
            fig_month.update_layout(title='Energy Consumption by Month',
                                    xaxis_title='Month',
                                    yaxis_title='Energy Consumption',
                                    width=1500,
                                    height=600)

            # Show the figure
            st.plotly_chart(fig_month)
            st.markdown("""
- **Energy Consumption on Weekdays (Regressor Component)**: This bar plot focuses specifically on energy consumption during weekdays (Monday to Friday)
- It serves as a regressor component to capture the differences in energy usage between weekdays and weekends.
            """)

            fig_weekdays = go.Figure()
            weekdays_df = merged_df[(merged_df['ds'].dt.weekday >= 0) & (merged_df['ds'].dt.weekday <= 4)]
            sundays_df = merged_df[merged_df['ds'].dt.weekday == 6]

            weekend_df = merged_df[(merged_df['ds'].dt.weekday == 5) | (merged_df['ds'].dt.weekday == 6)]
            # Extract date and time separately
            weekdays_df['date'] = weekdays_df['ds'].dt.strftime('%Y-%m-%d')
            weekdays_df['time'] = weekdays_df['ds'].dt.strftime('%H:%M:%S')

            # Create hover text with labels
            hover_text = ('Date: ' + weekdays_df['date'] + '<br>' +
                        'Time: ' + weekdays_df['time'] + '<br>' +
                        'Energy: ' + weekdays_df['yhat'].astype(str) + ' kWh')

            # Create the bar chart

            fig_weekdays.add_bar(x=weekdays_df['ds'].dt.weekday, 
                                y=weekdays_df['yhat'], 
                                marker_color='blue',
                                hovertext=hover_text)

            # Update layout
            fig_weekdays.update_layout(title='Energy Consumption on Weekdays',
                                    xaxis_title='Weekday',
                                    yaxis_title='Energy Consumption',
                                    height=600,
                                    width=1500,
                                    xaxis=dict(tickmode='array', tickvals=[0, 1, 2, 3, 4],
                                                ticktext=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']))

            # Show the figure
            st.plotly_chart(fig_weekdays)

            st.markdown("""
- **Energy Consumption on Weekends (Regressor Component)**: This bar plot focuses specifically on energy consumption during weekends (Saturday and Sunday).
- It serves as a regressor component to capture the differences in energy usage between weekdays and weekends.
                        """)

            fig_weekend = go.Figure()
            # Extract date and time separately
            weekend_df['date'] = weekend_df['ds'].dt.strftime('%Y-%m-%d')
            weekend_df['time'] = weekend_df['ds'].dt.strftime('%H:%M:%S')

            # Create hover text with labels
            hover_text = ('Date: ' + weekend_df['date'] + '<br>' +
                        'Time: ' + weekend_df['time'] + '<br>' +
                        'Energy: ' + weekend_df['yhat'].astype(str) + ' kWh')

            # Create the bar chart

            fig_weekend.add_bar(x=weekend_df['ds'].dt.weekday, 
                                y=weekend_df['yhat'], 
                                marker_color='purple',
                                hovertext=hover_text)

            # Update hover template

            # Update layout
            fig_weekend.update_layout(title='Energy Consumption on Weekends',
                                    xaxis_title='Day of the Week',
                                    yaxis_title='Energy Consumption',
                                    height=600,
                                    width=1500,
                                    xaxis=dict(tickmode='array', tickvals=[5, 6],
                                                ticktext=['Saturday', 'Sunday']))

            # Show the figure
            st.plotly_chart(fig_weekend)

            st.markdown("""
- **Energy Consumption on Sundays (Regressor Component)**: This bar plot zooms in on energy consumption specifically on Sundays, providing detailed insights into energy usage patterns on Sundays compared to other days of the week.
- It serves as a regressor component to capture the unique patterns of energy usage on Sundays.
                        """)


            # Extract date and time separately
            sundays_df['date'] = sundays_df['ds'].dt.strftime('%Y-%m-%d')
            sundays_df['time'] = sundays_df['ds'].dt.strftime('%H:%M:%S')

            # Create hover text with labels
            hover_text = ('Date: ' + sundays_df['date'] + '<br>' +
                        'Time: ' + sundays_df['time'] + '<br>' +
                        'Energy: ' + sundays_df['yhat'].astype(str) + ' kWh')

            # Create the bar chart
            fig_sundays = go.Figure()
            fig_sundays.add_bar(x=sundays_df['ds'].dt.date, 
                                y=sundays_df['yhat'], 
                                marker_color='pink',
                                hovertext=hover_text)

            # Update layout
            fig_sundays.update_layout(title='Energy Consumption on Sundays',
                                    xaxis_title='Date',
                                    yaxis_title='Energy Consumption',
                                    height=600,
                                    width=1500)

            # Show the figure
            st.plotly_chart(fig_sundays)

        # Show the plot
if __name__ == "__main__":
    main()

