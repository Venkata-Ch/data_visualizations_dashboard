import streamlit as st
import pandas as pd
import configparser
import logging

import plotly.graph_objects as plot
from math import sqrt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger()

config_ =  configparser.ConfigParser()

config_.read('./.config')
api_key = config_.get('alpha_vantage_api','api_key')


st.logo("logos/stock_analyzer.jpg",
    icon_image="logos/stock_analyzer.jpg",
)
st.html('''<html>
              <title></title>
              <h3>Dynamic stock Analysis</h3>
              <div>
              <img src='' , align='center'></img>
              
               <p>The current apps  is visualizer the stock analysis of the data.
               <br><b>Stock Frequency: Monthly<br>
               
               </b> 
               </p>
              </div>
              </html>''')


class Dynamic_analyzer:
    def __init__(self):
        pass

    @staticmethod
    def gather_data():
        try:
            #Enter your stock symbol
            logger.info("Retrieving Data")
            symbol = st.text_input(label="Enter your stock symbol")
            st.write("The stock symbol is", symbol)
            if st.button("Submit"):
                #Data Processing
                url = f"https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol={symbol}&apikey={api_key}&outputsize=full&datatype=csv"
                stock_data = pd.read_csv(url)
                stock  = pd.DataFrame(stock_data)
                st.markdown('Data Retrieved')
                return stock
        except Exception as error:
            logger.error(str(error))

    #Cleaning and  doing exploratory data analysis
    @st.cache_data
    def clean_data(_self,data):
        st.write(data.head())
        data = pd.DataFrame(data)
        #Cleaningdata
        data = data.dropna()

        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['month'] = data['timestamp'].dt.strftime('%B')
        grouped = data.groupby('month')['open'].mean().reset_index()



        #barplot
        bar_plot = plot.Figure()
        bar_plot.add_trace(plot.Bar(
            x=grouped['month'],
            y=grouped['open'],
            marker_color='rgb(158,202,225)',
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5
        ))

        bar_plot.update_layout(
            title='Monthly Values',
            xaxis_title='Month',
            yaxis_title='Value',
            xaxis_tickangle=-45,
            width=1200,
            height=600
        )
        st.plotly_chart(bar_plot)

        #Time Series chart
        price_change = data['close'].iloc[-1] - data['close'].iloc[-2]
        price_change_percent = (price_change / data['close'].iloc[-2]) * 100

        fig = plot.Figure(data=[plot.Candlestick(x=data['timestamp'],
                                             open=data['open'],
                                             high=data['high'],
                                             low=data['low'],
                                             close=data['close'])])

        fig.update_layout(title='Stock Price Over Time',
                          xaxis_title='Date',
                          yaxis_title='Price')

        st.title('Interactive Stock Visualization')

        if price_change >= 0:
            symbol = "▲"
            color = "green"
        else:
            symbol = "▼"
            color = "red"

        html_string = f"""
        <div style="font-size: 24px; color: {color};">
            <span style="cursor: pointer;" title="Price change: ${price_change:.2f} ({price_change_percent:.2f}%)">{symbol}</span>
        </div>
        """

        st.markdown(html_string, unsafe_allow_html=True)

        st.plotly_chart(fig)

        st.subheader('Stock Data')

        return grouped


    def ml_model(self, data):
        try:
            numeric_columns = ['open', 'high', 'low', 'close', 'adjusted close', 'volume', 'dividend amount']
            data[numeric_columns] = data[numeric_columns].astype(float)

            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data = data.sort_values('timestamp')

            # Feature engineering
            data['days_since_start'] = (data['timestamp'] - data['timestamp'].min()).dt.days

            # Preparing features and target
            X = data[['days_since_start', 'high', 'low', 'close', 'volume']]
            y = data['open']

            # Splitting the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Training the model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Making predictions
            y_pred = model.predict(X_test)

            # Model EValuation
            mse = sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            #RandomFOrest Regressor
            forest = RandomForestRegressor(n_estimators=10,random_state=41)
            forest.fit(X_train,y_train)

            #Making predictions
            rf_pred = forest.predict(X_test)

            #Evaluating model
            rf_mse = sqrt(mean_squared_error(y_test, rf_pred))
            rf_r2 = r2_score(y_test, rf_pred)




            # Predicting for 2025
            last_date = data['timestamp'].max()
            days_until_2025 = (pd.Timestamp('2025-01-01') - last_date).days
            X_2025 = pd.DataFrame({
                'days_since_start': [data['days_since_start'].max() + days_until_2025],
                'high': [data['high'].mean()],
                'low': [data['low'].mean()],
                'close': [data['close'].mean()],
                'volume': [data['volume'].mean()]
            })


            st.title('Stock Price Prediction 2025')

            st.subheader('Historical Data')
            st.dataframe(data)
            #Select alignment
            plot_alignment = st.selectbox(
                "Vertical alignment", ["top", "center", "bottom"], index=2
            )
            co1,co2 = st.columns(2,gap="large",vertical_alignment=plot_alignment)
            with co1:
                prediction_2025 = model.predict(X_2025)[0]

                st.subheader('Linear Regression Performance')
                st.write(f'Root Mean Squared Error: {mse:.2f}')
                st.write(f'R-squared Score: {r2:.2f}')
                st.subheader('Prediction for 2025')
                st.write(f'Predicted Opening Price: ${prediction_2025:.2f}')

                fig = plot.Figure()

                fig.add_trace(
                    plot.Scatter(x=data['timestamp'], y=data['open'], mode='lines+markers', name='Historical Open Price'))

                fig.add_trace(
                    plot.Scatter(x=[pd.Timestamp('2025-01-01')], y=[prediction_2025], mode='markers', name='2025',
                               marker=dict(size=10, color='red', symbol='star')))

                fig.update_layout(title='Stock Open Prize prediction for 2025 ',
                                  xaxis_title='Date',
                                  yaxis_title='Opening Price',
                                  hovermode='x unified')

                st.plotly_chart(fig)

                price_change = prediction_2025 - data['open'].iloc[-1]
                price_change_percent = (price_change / data['open'].iloc[-1]) * 100

                if price_change >= 0:
                    symbol = "▲"
                    color = "green"
                else:
                    symbol = "▼"
                    color = "red"

                html_string = f"""
                            <div style="font-size: 24px; color: {color};">
                                <span style="cursor: pointer;" title="Price change: ${price_change:.2f} ({price_change_percent:.2f}%)">{symbol}</span>
                            </div>
                            """

                st.markdown(html_string, unsafe_allow_html=True)
            with co2:
                rf_prediction_2025 = forest.predict(X_2025)[0]

                st.subheader('RandomForest Regression Performance')
                st.write(f'Root Mean Squared Error: {rf_mse:.2f}')
                st.write(f'R-squared Score: {rf_r2:.2f}')

                st.subheader('Prediction for 2025')
                st.write(f'Predicted Opening Price: ${rf_prediction_2025:.2f}')

                fig = plot.Figure()

                fig.add_trace(
                    plot.Scatter(x=data['timestamp'], y=data['open'], mode='lines+markers', name='Historical Open Price'))

                fig.add_trace(
                    plot.Scatter(x=[pd.Timestamp('2025-01-01')], y=[rf_prediction_2025], mode='markers', name='2025',
                               marker=dict(size=10, color='red', symbol='star')))

                fig.update_layout(title='Stock Open Prize prediction for 2025 ',
                                  xaxis_title='Date',
                                  yaxis_title='Opening Price',
                                  hovermode='x unified')

                st.plotly_chart(fig)

                price_change = rf_prediction_2025 - data['open'].iloc[-1]
                price_change_percent = (price_change / data['open'].iloc[-1]) * 100

                if price_change >= 0:
                    symbol = "▲"
                    color = "green"
                else:
                    symbol = "▼"
                    color = "red"

                html_string = f"""
                                            <div style="font-size: 24px; color: {color};">
                                                <span style="cursor: pointer;" title="Price change: ${price_change:.2f} ({price_change_percent:.2f}%)">{symbol}</span>
                                            </div>
                                            """

                st.markdown(html_string, unsafe_allow_html=True)

            return data

        except Exception as e:
            logger.error(str(e))


def main():
    sto = Dynamic_analyzer()
    data = sto.gather_data()
    if data is not None:
        sto.clean_data(data)
        sto.ml_model(data)


if __name__=="__main__":
    main()


