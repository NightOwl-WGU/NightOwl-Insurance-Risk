# -*- coding: utf-8 -*-

# import needed libraries
from lib2to3.refactor import MultiprocessingUnsupported
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns


def check_password():
    """Returns `True` if the user had the correct password."""
    # creating containers
    login_container = st.container()
    # create the header for the webapp
    with login_container:
        img_col, co_col = st.columns(2)
        img_col.image('owl_logo.jpg')
        co_col.title('NightOwl Insurance Company')

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if check_password():

    # function to get dataset and fill missing values
    #@st.cache
    def get_data(file_info):
        data_retrieved = pd.read_csv(file_info)

        return data_retrieved
    #end get_data



    #   ##########################################################
    #   Preparing the data
    #   ##########################################################

    # get the data from file
    car_df = get_data('Car_Insurance_Claim.csv')

    # fill missing numeric rows with the median
    for label, content in car_df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                car_df[label] = content.fillna(content.median())

    # get copy of df with original values before categorizes values
    ml_df = car_df.copy()

    # remove column not needed for the model or predictions
    ml_df.drop(['ID', 'POSTAL_CODE', 'RACE'], axis=1, inplace=True)

    # convert float to int for columns with int values needed
    ml_df['VEHICLE_OWNERSHIP'] = ml_df['VEHICLE_OWNERSHIP'].astype(int)
    ml_df['MARRIED'] = ml_df['MARRIED'].astype(int)
    ml_df['CHILDREN'] = ml_df['CHILDREN'].astype(int)
    ml_df['OUTCOME'] = ml_df['OUTCOME'].astype(int)

    # converting CREDIT_SCORE into value normally used by populace
    ml_df['CREDIT_SCORE'] = ml_df['CREDIT_SCORE'] * 1000

    # Need to convert objects into numbers so building a view of columns with object datatypes
    unique_data = {
        "AGE": ml_df['AGE'].unique(),
        "GENDER": ml_df['GENDER'].unique(),
        "DRIVING_EXPERIENCE": ml_df['DRIVING_EXPERIENCE'].unique(),
        "EDUCATION": ml_df['EDUCATION'].unique(),
        "INCOME": ml_df['INCOME'].unique(),
        "VEHICLE_YEAR": ml_df['VEHICLE_YEAR'].unique(),
        "VEHICLE_TYPE": ml_df['VEHICLE_TYPE'].unique()
    }

    # create a new array/dictionary of values numerical representation
    obj_dict = {
        "AGE": {'16-25': 0, '26-39': 1, '40-64': 2, '65+':3},
        "GENDER": {"male": 1, "female": 0, "Female": 0, "Male": 1},
        "DRIVING_EXPERIENCE": {'0-9y': 0, '10-19y': 1, '20-29y': 2, '30y+': 3},
        "EDUCATION": {'none': 0, 'high school': 1, 'university': 2},
        "INCOME": {'poverty': 0, 'working class': 1, 'middle class': 2, 'upper class': 3},
        "VEHICLE_YEAR": {'after 2015': 0, 'before 2015': 1},
        "VEHICLE_TYPE": {'sedan': 0, 'sports car': 1}
    }

    # converting the objects to the corresponding replacements created in the obj_dict
    ml_df.replace(to_replace=['16-25', 'female', '0-9y','none', 'poverty', 'after 2015', 'sedan'], value=0, inplace=True)
    ml_df.replace(to_replace=['26-39', 'male', '10-19y','high school', 'working class', 'before 2015', 'sports car'], value=1, inplace=True)
    ml_df.replace(to_replace=['40-64', '20-29y', 'university', 'middle class'], value=2, inplace=True)
    ml_df.replace(to_replace=['65+', '30y+', 'upper class'], value=3, inplace=True)



    #   ##########################################################
    #   Building a Random Forest Classifier Model
    #   ##########################################################

    # split data into variables and result
    X = ml_df.drop('OUTCOME', axis=1)
    y = ml_df['OUTCOME']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Instantiate model
    clf = RandomForestClassifier()

    # Fit the Model
    clf.fit(X_train, y_train)



    #   ##########################################################
    #   Creating a Header
    #   ##########################################################

    # creating containers
    header_container = st.container()
    diag1_container = st.container()
    diag2_container = st.container()
    diag3_container = st.container()
    ml_container = st.container()


    # create the header for the webapp
    with header_container:

        img_col, co_col = st.columns(2)
        img_col.image('owl_logo.jpg')
        co_col.title('NightOwl Insurance Company')


    #   ##########################################################
    #   Bar graph of percentage of accidents by age
    #   ##########################################################

    # creating a dict of percentage of accidents within an age group
    age_percent = {
        '16-25': ((car_df.loc[car_df['AGE']=='16-25', 'OUTCOME'].sum() / car_df.loc[car_df['AGE']=='16-25', 'OUTCOME'].count())*100),
        '26-39': ((car_df.loc[car_df['AGE']=='26-39', 'OUTCOME'].sum() / car_df.loc[car_df['AGE']=='26-39', 'OUTCOME'].count())*100),
        '40-64': ((car_df.loc[car_df['AGE']=='40-64', 'OUTCOME'].sum() / car_df.loc[car_df['AGE']=='40-64', 'OUTCOME'].count())*100),
        '65+': ((car_df.loc[car_df['AGE']=='65+', 'OUTCOME'].sum() / car_df.loc[car_df['AGE']=='65+', 'OUTCOME'].count())*100)
    }


    with diag1_container:

        #adding line to create space
        st.text('                            ')
        st.text('                            ')

        # graph title
        st.header('Percentage of Accidents within Age Group')

        # generating bar plot of age_percent data
        fig = plt.figure(figsize = (6,7))
        plt.style.use('seaborn')
        plt.ylim(bottom=0, top=80)
        plt.bar(list(age_percent.keys()), list(age_percent.values()), width = 0.6)

        plt.grid(axis='x')
        plt.xlabel("Age Group", fontsize=15)
        plt.ylabel("Percentage of Accidents", fontsize=15)
        plt.yticks()
        #plt.title("Percentage of Accidents within age Group", fontsize=22)
        
        st.pyplot(fig)

    #   ##########################################################
    #   Creating pie chart for gender percentage of accidents
    #   ##########################################################

    gender_percent = np.array([
        ((car_df.loc[car_df['GENDER']=='female', 'OUTCOME'].sum() / car_df['OUTCOME'].sum())*100),
        ((car_df.loc[car_df['GENDER']=='male', 'OUTCOME'].sum() / car_df['OUTCOME'].sum())*100)
    ])
    gender_labels = ["Female", "Male"]
    gender_colors = ['m', 'b']



    with diag2_container:

        #adding line to create space
        st.text('                            ')
        st.text('                            ')

        # graph title
        st.header('Accidents by Gender')
    
        fig1, ax1 = plt.subplots()
        ax1.pie(gender_percent, labels=gender_labels, colors=gender_colors, autopct='%1.1f%%', startangle=90, 
            textprops={'fontsize': 14, 'fontweight': 'bold'})
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        #plt.setp(autotexts, size = 8, weight ="bold")
        st.pyplot(fig1)




    #   ##########################################################
    #   Feature importance
    #   ##########################################################

        #adding line to create space
        st.text('                            ')
        st.text('                            ')

        # graph title
        st.header('Feature Importance leading to an Accident')

        
        #Create arrays from feature importance and feature names
        feature_importance = np.array(clf.feature_importances_)
        feature_names = np.array(X_train.columns)

        #Create a DataFrame using a Dictionary
        data={'feature_names':feature_names,'feature_importance':feature_importance}
        fi_df = pd.DataFrame(data)

        #Sort the DataFrame in order decreasing feature importance
        fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

        #Define size of bar plot
        fig2 = plt.figure(figsize=(10,8))
        #Plot Searborn bar chart
        sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
        #Add chart labels
        plt.xlabel('FEATURE IMPORTANCE')
        plt.ylabel('FEATURE NAMES')
        st.pyplot(fig2)


    #   ##########################################################
    #   Predictions
    #   ##########################################################

    with ml_container:

        #adding line to create space
        st.text('                            ')
        st.text('                            ')
        st.text('                            ')
        st.text('                            ')

        # title
        st.header('Vehicle Accident Prediction')

        # create columns for display
        input_col, space_col, predict_col = st.columns(3)

        # yes/no dict to convert input to number
        yes_no = {"Yes": 1, "No": 0}

        input_col.text("Enter Insured's Info:")
        age = input_col.selectbox('Age Group', options=['16-25', '26-39', '40-64', '65+'], index=0)
        age = obj_dict["AGE"][age]
        
        sex = input_col.radio("Sex", ('Female', 'Male'))
        sex = obj_dict["GENDER"][sex]
        
        exp = input_col.selectbox('Driving Experience', options=['0-9y', '10-19y', '20-29y', '30y+'], index=0)
        exp = obj_dict["DRIVING_EXPERIENCE"][exp]
        
        edu = input_col.selectbox('Education', options=['none', 'high school', 'university'], index=0)
        edu = obj_dict["EDUCATION"][edu]
        
        income = input_col.selectbox('Income Range', options=['poverty', 'working class', 'middle class', 'upper class'], index=0)
        income = obj_dict["INCOME"][income]
        
        credit = input_col.slider('Credit Score', min_value=50, max_value=950, value=500, step=10 )
        
        owner = input_col.radio('Owns Vehicle', ('Yes', 'No'))
        owner = yes_no[owner]

        car_year = input_col.radio("Car Year", ('before 2015', 'after 2015'))
        car_year = obj_dict["VEHICLE_YEAR"][car_year]

        marriage = input_col.radio('Married?', ('Yes', 'No'))
        marriage = yes_no[marriage]   
        
        children = input_col.radio('Children?', ('Yes', 'No'))
        children = yes_no[children]
        
        mileage = input_col.slider('Annual Mileage', min_value=0, max_value=100000, step=5000)

        car_type = input_col.radio("Car Type", ('sedan', 'sports car'))
        car_type = obj_dict["VEHICLE_TYPE"][car_type]
        
        speeding = input_col.radio('Has Speeding Violations?', ('Yes', 'No'))
        speeding = yes_no[speeding]

        dui = input_col.radio("Has any DUI's?", ('Yes', 'No'))
        dui = yes_no[dui]

        accidents = input_col.slider('How Many Past Accidents?', min_value=0, max_value=30, step=1)




        pred_df = pd.DataFrame(
            {'AGE': [age],
            'GENDER': [sex],
            'DRIVING_EXPERIENCE': [exp],
            'EDUCATION': [edu],
            'INCOME': [income],
            'CREDIT_SCORE': [credit],
            'VEHICLE_OWNERSHIP': [owner],
            'VEHICLE_YEAR': [car_year],
            'MARRIED': [marriage],
            'CHILDREN': [children],
            'ANNUAL_MILEAGE': [mileage],
            'VEHICLE_TYPE': [car_type],
            'SPEEDING_VIOLATIONS': [speeding],
            'DUIS': [dui],
            'PAST_ACCIDENTS': [accidents]
            }
        )


        predict_col.header('Accident Prediction')
        predict_col.text('-------------------')

        prediction = clf.predict(pred_df)[0]

        if prediction == 0:
            acc_string = "not being in a car accident."
        elif prediction == 1:
            acc_string = "being in a car accident."

        probability = (clf.predict_proba(pred_df)[0][prediction])
        

        predict_col.header('The insured has a ' + "{:.2%}".format(probability) + ' probability of ' + acc_string)


    
