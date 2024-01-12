import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import math
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

with st.sidebar:
    selected = option_menu("Main Menu", ["Home","Predict"], 
                icons=['house','cloud'], menu_icon="cast", default_index=0)
         
if selected == "Home":
    st.header("Singapore Resale Flat Prices Predicting",divider='grey')
    st.write("""**The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application that predicts 
             the resale prices of flats in Singapore. This predictive model will be based on historical data of resale flat transactions, and it aims 
             to assist both potential buyers and sellers in estimating the resale value of a flat.**""")
    st.subheader(":orange[Scope]",divider='grey')
    st.write("""**1. Data Collection and Preprocessing: Collect a dataset of resale flat transactions from the Singapore Housing and Development Board (HDB) for
the years 1990 to Till Date. Preprocess the data to clean and structure it for machine learning.**""")
    st.write("""**2. Feature Engineering: Extract relevant features from the dataset, including town, flat type, storey range, floor area, flat model, and lease commence
date. Create any additional features that may enhance prediction accuracy.**""")
    st.write("""**3. Model Selection and Training: Choose an appropriate machine learning model for regression (e.g., linear regression, decision trees, or random
forests). Train the model on the historical data, using a portion of the dataset for training.**""")
    st.write("""**4. Model Evaluation: Evaluate the model's predictive performance using regression metrics such as Mean Absolute Error (MAE), Mean Squared
Error (MSE), or Root Mean Squared Error (RMSE) and R2 Score.**""")
    st.write("""**5. Streamlit Web Application: Develop a user-friendly web application using Streamlit that allows users to input details of a flat (town, flat type, storey
range, etc.). Utilize the trained machine learning model to predict the resale price based on user inputs.**""")
    st.write("""**6. Deployment on Render: Deploy the Streamlit application on the Render platform to make it accessible to users over the internet.**""")
    st.write("""**7. Testing and Validation: Thoroughly test the deployed application to ensure it functions correctly and provides accurate predictions.**""")
    st.subheader(":orange[Skills take away from this Project]",divider='grey')
    st.text("> Data Wrangling,")
    st.text("> EDA,")
    st.text("> Model Building,")
    st.text("> Streamlit")
    st.text("> Model Deployment")

elif selected == "Predict":
    st.subheader(":orange[Enter the values to predict the price]")

    #Regression Model
    flat_model = ['IMPROVED', 'NEW GENERATION', 'MODEL A', 'STANDARD', 'SIMPLIFIED',
       'MODEL A-MAISONETTE', 'APARTMENT', 'MAISONETTE', 'TERRACE',
       '2-ROOM', 'IMPROVED-MAISONETTE', 'MULTI GENERATION',
       'PREMIUM APARTMENT', 'Improved', 'New Generation', 'Model A',
       'Standard', 'Apartment', 'Simplified', 'Model A-Maisonette',
       'Maisonette', 'Multi Generation', 'Adjoined flat',
       'Premium Apartment', 'Terrace', 'Improved-Maisonette',
       'Premium Maisonette', '2-room', 'Model A2', 'DBSS', 'Type S1',
       'Type S2', 'Premium Apartment Loft', '3Gen']

    town = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
        'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI',
        'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
        'KALLANG/WHAMPOA', 'MARINE PARADE', 'QUEENSTOWN', 'SENGKANG',
        'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN',
        'LIM CHU KANG', 'SEMBAWANG', 'BUKIT PANJANG', 'PASIR RIS',
        'PUNGGOL']

    flat_type = ['1 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', '2 ROOM', 'EXECUTIVE',
        'MULTI GENERATION', 'MULTI-GENERATION']

    storey_range = ['10 TO 12', '04 TO 06', '07 TO 09', '01 TO 03', '13 TO 15',
        '19 TO 21', '16 TO 18', '25 TO 27', '22 TO 24', '28 TO 30',
        '31 TO 33', '40 TO 42', '37 TO 39', '34 TO 36', '06 TO 10',
        '01 TO 05', '11 TO 15', '16 TO 20', '21 TO 25', '26 TO 30',
        '36 TO 40', '31 TO 35', '46 TO 48', '43 TO 45', '49 TO 51']

    lease_commence_date = [1977, 1976, 1978, 1979, 1984, 1980, 1985, 1981, 1982, 1986, 1972,
        1983, 1973, 1969, 1975, 1971, 1974, 1967, 1970, 1968, 1988, 1987,
        1989, 1990, 1992, 1993, 1994, 1991, 1995, 1996, 1997, 1998, 1999,
        2000, 2001, 1966, 2002, 2006, 2003, 2005, 2004, 2008, 2007, 2009,
        2010, 2012, 2011, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2022,
        2020]

    M = st.number_input('Enter Month',min_value=1, max_value=12, step=1, value=None,placeholder = 'Month')
    Y = st.number_input('Enter Year',min_value=1990, max_value=2023, step=1, value=None,placeholder = 'Year')
    T = st.selectbox("Select Town",town,index=None,placeholder="Town",key='r_t')
    FM = st.selectbox("Select Flat Model",flat_model,index=None,placeholder="Flat Model",key='r_fm')
    FT = st.selectbox("Select Flat Type",flat_type,index=None,placeholder="Flat Type",key='r_ft')
    SR = st.selectbox("Select Storey Range",storey_range,index=None,placeholder="Storey Range",key='r_sr')
    C = st.selectbox("Select Lease Commence Year",lease_commence_date,index=None,placeholder="Lease Commence Year",key='r_lcy')
    FA = st.number_input('Enter Floor Area in sqm',min_value=28.0, max_value=173.0, value=None,placeholder = 'Floor Area')

    if T != "" and T is not None and FT != "" and FT is not None and SR != "" and SR is not None and FM != "" and FM is not None:
        with open(r'D:\Project\Capstone\pkl_files\resale_label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)

        # Your new input data
        new_data = {
            'town': [T],
            'flat_type': [FT],
            'storey_range': [SR],
            'flat_model': [FM]
        }

        # Create a DataFrame from the new input data
        new_df = pd.DataFrame(new_data)

        # Apply label encoding using the loaded label encoders
        for column, le in label_encoders.items():
            if column in new_df.columns:
                new_df[column] = le.transform(new_df[column])

        town_value = new_df['town'].iloc[0]
        flat_type_value = new_df['flat_type'].iloc[0]
        storey_range_value = new_df['storey_range'].iloc[0]
        flat_model_value = new_df['flat_model'].iloc[0]

    with open(r'D:\Project\Capstone\pkl_files\resaleprice_model_regressor.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    # Now you can use the loaded_model for predictions

    # Example input data (replace this with your own input data)
    #[month,town,flat_type,storey_range,floor_area_sqm,flat_model,lease_commence_date,resale_price,year]
    

    if M is not None and  FA is not None and C is not None and Y is not None and T != "" and T is not None and FT != "" and FT is not None and SR != "" and SR is not None and FM != "" and FM is not None:
        new_input = [[M,town_value,flat_type_value,storey_range_value,FA,flat_model_value,C,Y]]
        # Make predictions
        result = loaded_model.predict(new_input)
        # Print the result
        st.write("**Resale Price:**", *result)






