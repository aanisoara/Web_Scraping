import streamlit as st
from plotly import graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import requests
from datetime import datetime , timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import time 
from PIL import Image
import seaborn as sns
sns.set_style("whitegrid")
from  matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
from streamlit_folium import folium_static
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from yellowbrick.regressor.alphas import alphas
from streamlit_folium import folium_static
from folium.plugins import HeatMap
import streamlit as st
import pandas as pd
from prophet import Prophet
#from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import requests
import base64
from matplotlib.pyplot import figure
import plotly.graph_objects as go
from datetime import datetime
from PIL import Image
from io import BytesIO
import plotly.express as px
import requests
import re


pages_name = ['Presentation','Modelisation','Actual Temperature']
# Create a sidebar with a radio button to select the page
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", pages_name)

def file_selector():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_dir, 'Data')
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

    #uploaded_file = st.file_uploader("Choose a file")
uploaded_file = filename = file_selector()
if uploaded_file is not None:
    #st.write('You selected `%s`' % filename)
    dataframe = pd.read_csv(uploaded_file)
    if 'Unnamed: 0' in dataframe.columns:
        dataframe.drop('Unnamed: 0', axis=1, inplace=True)
        
if page == 'Presentation':
    st.markdown("<h1 style='text-align: center; color: black; font-size:25px; border-radius:2%;'>Machine Learning Application for weather forcast</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: ; font-size:20px;'>Members: ABABII Anisoara, LAAUMARI Oussama, ZAHI Camil </h2>", unsafe_allow_html=True)
    st.write("""
    <div style="text-align:justify">
    In this Application for forcasting weather, the *XGBoostRegressor()* function is used  for build a regression model using the **XGBoost** algorithm.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color:red ; font-size:20px;'>Try it, you'll be love! </h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color:blue ; font-size:20px;'>Description of Weather Underground service </h2>", unsafe_allow_html=True)
    st.write("""
    <div style="text-align:justify">
    Weather Underground is a commercial weather service providing real-time weather information over the Internet. Weather Underground provides weather reports for most major cities around the world on its Web site, as well as local weather reports for newspapers and third-party sites. Its information comes from the National Weather Service (NWS), and over 250,000 personal weather stations (PWS). The site is available in many languages, and customers can access an ad-free version of the site with additional features for an annual fee. Weather Underground is owned by The Weather Company, a subsidiary of IBM.
    </div>
    """, unsafe_allow_html=True)

    st.image("weather.jpg", width=700, use_column_width=False)
    if uploaded_file is not None:
        #st.write('You selected `%s`' % filename)
        dataframe = pd.read_csv(uploaded_file)
        #city_name = filename.split(".")[0]
        file_path = filename
        city_name = re.search(r'Data\\(.*?)\.csv', file_path).group(1)
        #st.write(city_name)
        st.write("You have selected data for " + city_name)
        if 'Unnamed: 0' in dataframe.columns:
            dataframe.drop('Unnamed: 0', axis=1, inplace=True)
        if city_name == "Lyon":
            st.image("Lyon.jpg")
        elif city_name == "Madrid":
            st.image("Madrid.jpg")
        elif city_name == "Marseille":
            st.image("Marseille.jpg")
        elif city_name == "Milan":
            st.image("Milan.jpg")
        elif city_name == "New_york":
            st.image("NY.jpg")
        else:
            st.image("Paris.jpg")


elif page == 'Modelisation':

    def df_to_X_y(df, window_size=12):
        df_as_np = df.to_numpy()
        X = []
        y = []
        for i in range(len(df_as_np)-window_size):
            row = [[a] for a in df_as_np[i:i+12]]
            X.append(row)
            label = df_as_np[i+12]
            y.append(label)
        return np.array(X), np.array(y)


    def create_features(df):
        """
        Create time series features based on time series index.
        """
        df = df.copy()
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['dayofyear'] = df.index.dayofyear
        df['dayofmonth'] = df.index.day
        df['weekofyear'] = df.index.isocalendar().week
        return df

    #___________________________________________  Fonction for Modelisation  ______________________________________________#
    # Model building
    def build_model(df):

        df['Date'] =  pd.to_datetime(df['Date'], format='%Y-%m-%d')
        df_1 = df[['Date','Temp_avg']]
        df_1 = df_1.rename(columns={'Date':'ds','Temp_avg':'y'})

        train = df_1[df_1['ds']<'2020-1-1']
        test = df_1[df_1['ds']>='2022-12-1']
        train = df[:int(len(df)*0.8)]
        test = df[int(len(df)*0.8):]

        train = train[['Date','Temp_avg']]
        train = train.rename(columns={'Date':'ds','Temp_avg':'y'})
        test = test[['Date','Temp_avg']]
        test = test.rename(columns={'Date':'ds','Temp_avg':'y'})

        #m = Prophet(interval_width=0.95)
        #model = m.fit(train)

        #future = m.make_future_dataframe(periods=len(test))
        #forecast = m.predict(future)

        test['y'] = test['y'].astype(float)
        st.subheader('2. Model Performance XGBoost')

        #forecast_test = forecast.iloc[-len(test):]['yhat']
        #rmspe = np.sqrt(np.mean((test['y'] - forecast_test) ** 2))
        #st.write('RMSPE')
        #st.info(rmspe)

        #forecast_test = forecast.iloc[-len(test):]['yhat']
        #mae = mean_absolute_error(test['y'], forecast_test)
        #st.write('MAE for forcasting test dataset: Prophet model')
        #st.info(mae)

        # PLOT  prophet
        #plot1 = m.plot(forecast)
        #st.write(plot1)

        train = df[:int(len(df)*0.8)]
        test = df[int(len(df)*0.8):]
        train.index = train['Date']
        test.index = test['Date']

        ###  plot
        figure, axx = plt.subplots(figsize=(15, 5))
        train['Temp_avg'].plot(ax=axx, label='Training Set', title='Data Train/Test Split')
        test['Temp_avg'].plot(ax=axx, label='Test Set')
        axx.axvline('01-01-2020', color='black', ls='--')
        axx.legend(['Training Set', 'Test Set'])
        st.pyplot(figure)

        train_t = create_features(train)
        test_t = create_features(test)

        #st.markdown('1.2. The new features are :')
        FEATURES = ['dayofyear', 'dayofweek', 'quarter', 'month', 'year']
        #st.info(FEATURES)

        TARGET = 'Temp_avg'

        X_train = train_t[FEATURES]
        y_train = train_t[TARGET]

        X_test = test_t[FEATURES]
        y_test = test_t[TARGET]

        reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                            n_estimators=1000,
                            early_stopping_rounds=50,
                            objective='reg:linear',
                            max_depth=3,
                            learning_rate=0.01)

        reg.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=100)


        fi = pd.DataFrame(data=reg.feature_importances_,
                    index=reg.feature_names_in_,
                    columns=['importance'])
        fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
        plt.show()

        st.markdown('Model Péerformance and predicted data plot')
        ypred = reg.predict(X_test)
        fig, x_ax = plt.subplots()
        x_ax =pd.DataFrame(X_test.index)
        plt.plot(x_ax, y_test, label="Test_data")
        plt.plot(x_ax, ypred, label="Predicted_data")
        plt.title("Raw data and Predicted data")
        plt.xlabel('Date')
        plt.ylabel('Y-axis')
        plt.legend(loc='best',fancybox=True, shadow=True)
        plt.grid(True)
        st.pyplot(fig)

        st.markdown('2.1. Prediction on Training set')
        Y_pred_train = reg.predict(X_train)
        st.write('Coefficient of determination ($R^2$):')
        st.info( r2_score(y_train, Y_pred_train) )

        st.markdown('2.2. Prediction on Test set')
        ypred = reg.predict(X_test)
        st.write('Coefficient of determination ($R^2$):')
        st.info( r2_score(y_test, ypred) )

        st.markdown('2.3. Prediction Error')
        st.write('Error: Mean Absolute Error (MAE):')
        st.info( mean_absolute_error(y_test, ypred) )

     ############################ Predicted temperature plot ########################################
        
        def fahrenheit_to_celsius(temp_fahrenheit):
            temp_celsius = (temp_fahrenheit - 32) * 5/9
            return temp_celsius
    
        def predict_with_date():
            today = datetime.now()
            date_array = [today.timetuple().tm_yday, today.weekday(), int((today.month -1) / 4) + 1, today.month, today.year]
            prediction = reg.predict(np.array(date_array).reshape(1,5))
            return fahrenheit_to_celsius(prediction)

        test['prediction'] = ypred
        temp_pred = predict_with_date()

        fig1 = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = temp_pred[0],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Predicted weather (°C)", 'font': {'size': 24}},
            gauge = {'axis': {'range': [-40, 40]},
                    'bar': {'color': "red"},
                    'steps' : [
                        {'range': [-40, -20], 'color': 'blue'},
                        {'range': [-20, 0], 'color': 'green'},
                        {'range': [0, 20], 'color': 'yellow'},
                        {'range': [20, 40], 'color': 'red'}
                    ]}
        ))
        st.plotly_chart(fig1)

    # Sidebar - Collects user input features into dataframe
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your own input CSV file", type=["csv"])

    # Displays the dataset
    st.subheader('1. Dataset')
    st.markdown('1.1. Initial Data set')
    st.write(dataframe.head())
    st.write('The data contains ', dataframe.shape[1], ' columns and ',dataframe.shape[0], ' rows !')
    st.markdown('Description of initial Data set')
    st.write(dataframe.describe())
    build_model(dataframe)  
else:
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        file_path = filename
        city_name = re.search(r'Data\\(.*?)\.csv', file_path).group(1)

    #Gather weather data
        def weather_week(selected_city):
            city = selected_city
            api_key = 'f15d6887ef0449fa9f695040232701'
            url = f"http://api.weatherapi.com/v1/forecast.json?key={api_key}&q={city}&days=7"
            response = requests.get(url)
            data = response.json()
            today = datetime.now()
            predictions = []
            for i in range(7):
                date = today + timedelta(days=i)
                temperature = data['forecast']['forecastday'][i]['day']['avgtemp_c']
                predictions.append(temperature)
            df = pd.DataFrame({'Date': [today + timedelta(days=i) for i in range(7)],
                                'Temperature': predictions})
            df["Advice"] = df["Temperature"].apply(lambda x: "☃️ Il fait plus froid que dans un congélateur à légumes! Température : " + str(x) + "°C" if x < 0 
                else "❄️ N'oubliez pas de vous couvrir chaudement, il fait froid comme dans un frigo en panne! Température : " + str(x) + "°C" if x >= 0 and x < 5 
                else "🌬 Il fait frais, vous pouvez sortir sans risque de fondre comme une glace au soleil! Température : " + str(x) + "°C" if x >= 5 and x < 10 
                else "🌤 Il fait agréable, sortez profiter de cette belle journée! Température : " + str(x) + " °C" if x >= 10 and x < 20 
                else "🔥 Attention aux coups de chaud, n'oubliez pas de vous hydrater régulièrement! Température : " + str(x) + "°C" if x >= 20 and x < 30 
                else "☀️ Mieux vaut rester à l'ombre, c'est plus chaud que dans un sauna! Température : " + str(x) + "°C" if x >= 30 else "☀️ Il fait chaud, n'oubliez pas de vous hydrater et de vous protéger du soleil ! Température : " + str(x) + "°C")
            return df

    fig4 = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = weather_week(city_name)['Temperature'][0],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Actual Temperature (°C) using weatherapi model", 'font': {'size': 24}},
        gauge = {'axis': {'range': [-40, 40]},
                'bar': {'color': "red"},
                'steps' : [
                    {'range': [-40, -20], 'color': 'blue'},
                    {'range': [-20, 0], 'color': 'green'},
                    {'range': [0, 20], 'color': 'yellow'},
                    {'range': [20, 40], 'color': 'red'}
                ]}
    ))
    st.plotly_chart(fig4)
    week_pred_api = weather_week(city_name)
    st.write(week_pred_api["Advice"][0])   
    st.image("thanks.jpg")
