## Project Web scraping

Our plan is to consolidate six databases using web scraping from the website https://www.wunderground.com. We will use this data to create an application that predicts the weather for the user. The database will be a time series data.

The next step is to perform feature engineering by creating new variables from existing ones, such as the day of the week, quarter, and day of the year. We will also conduct a simple analysis using descriptive statistics to explore the data. As a bonus, we will use an XGBoost machine learning model to predict today's forecast on the consolidated data.

It is essential to use PySpark in the feature engineering part. Then, we will use both PySpark and pandas (if PySpark is not installed on your machine) for preprocessing and creating new features.

The final deliverable will be a Python notebook and a Streamlit dashboard. The dashboard will contain a presentation page of the website and data source where the user can choose a curated dataset scraped (e.g. New York, Paris, Milan, etc.), a modeling page for forecasting weather for the chosen dataset, and a page with the actual temperature, which allows us to compare our prediction error.x