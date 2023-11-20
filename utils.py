'''
This file contains all the helper functions that we need to use in the app
'''
import random
import streamlit as st
import numpy as np
import pandas as pd
from google_big_query import GoogleBigQuery, TransformationGoogleBigQuery
from parameters import *
from ai_classifier import ArtificialWalla

random.seed(42)

empty= ['', 'nan', ' ', np.nan]
value_map = {
    5: 10,
    4: 8,
    3: 5,
    2: 1,
    1: 1,
    0: 10
    }

def prepare_data(data):
    ''' Adding the necessary columns to the dataframe'''
    data = data.rename(columns={'venue': 'Reservation: Venue'})
    data = data.rename(columns={'reservation_date': 'Reservation: Date'})
    data['Date Submitted'] = data['Reservation: Date']
    data['Reservation: Time'] = ''
    data = data.rename(columns={'ambience': 'Feedback: Ambience Rating'})
    data = data.rename(columns={'service': 'Feedback: Service Rating'})
    data = data.rename(columns={'food': 'Feedback: Food Rating'})
    data = data.rename(columns={'drinks': 'Feedback: Drink Rating'})
    data = data.rename(columns={'overall': 'Overall Rating'})
    data['Reservation: Overall Rating'] = data['Overall Rating']
    data['Reservation: Food Rating'] = data['Feedback: Food Rating']
    data['Reservation: Drinks Rating'] = data['Feedback: Drink Rating']
    data['Reservation: Service Rating'] = data['Feedback: Service Rating']
    data['Reservation: Ambience Rating'] = data['Feedback: Ambience Rating']
    data['Reservation: Feedback Notes'] = ''
    data['Title'] = ''
    data['Feedback: Recommend to Friend'] = data['recommend_to_friend'].apply(
        lambda x: 'Yes' if x == 'True' else 'No')
    data['Reservation: Updated Date'] = ''
    data = data.rename(columns={'notes': 'Details'})
    data['Platform'] = 'SevenRooms'
    data = data.drop(columns=['order_id']    )
    data = data.fillna('')
    if 'Feedback: Drink Rating' in data.columns.tolist():
        data['Feedback: Drink Rating'] = data['Feedback: Drink Rating'].apply(
            lambda x: x.replace('"', ''))
        # if empty then 0
        data['Feedback: Drink Rating'] = data['Feedback: Drink Rating'].apply(
            lambda x: 0 if x == '' else x)
    return data

def process_direct_f(df_direct):
    '''
    They only contain 5 columns
    0 : 'CAFE' 
    1 : 'DATE RECEIVED' 
    3 : 'FEEDBACK'
    4 : 'SOURCE'
    5 : 'DONE'

    # need to make it this form

    0:"Feedback: Ambience Rating"
    1:"Feedback: Drink Rating"
    2:"feedback_type"
    3:"Feedback: Food Rating"
    4:"Details"
    5:"Overall Rating"
    6:"received_date"
    7:"recommend_to_friend"
    8:"Reservation: Date"
    9:"reservation_id"
    10:"Feedback: Service Rating"
    11:"Reservation: Venue"
    12:"Date Submitted"
    13:"Reservation: Time"
    14:"Reservation: Overall Rating"
    15:"Reservation: Food Rating"
    16:"Reservation: Drinks Rating"
    17:"Reservation: Service Rating"
    18:"Reservation: Ambience Rating"
    19:"Reservation: Recommend to Friend"
    20:"Reservation: Feedback Notes"
    21:"Title"
    22:"Feedback: Recommend to Friend"
    23:"Reservation: Updated Date"
    24:"Platform"
    '''
    # make it with this columns
    # get initial columns
    # drop empty rowas
    df_initial_columns = df_direct.columns.tolist()
    # create all the columns
    df_direct['Feedback: Ambience Rating'] = ''
    df_direct['Feedback: Drink Rating'] = ''
    df_direct['feedback_type'] = ''
    df_direct['Feedback: Food Rating'] = ''
    df_direct['Details'] = df_direct['FEEDBACK']
    df_direct['Overall Rating'] = ''
    df_direct['received_date'] = df_direct['DATE RECEIVED']
    df_direct['recommend_to_friend'] = ''
    df_direct['Reservation: Date'] = ''
    df_direct['reservation_id'] = ''
    df_direct['Feedback: Service Rating'] = ''
    df_direct['Reservation: Venue'] = df_direct['CAFE']
    df_direct['Date Submitted'] = df_direct['DATE RECEIVED']
    df_direct['Reservation: Time'] = ''
    df_direct['Reservation: Overall Rating'] = ''
    df_direct['Reservation: Food Rating'] = ''
    df_direct['Reservation: Drinks Rating'] = ''
    df_direct['Reservation: Service Rating'] = ''
    df_direct['Reservation: Ambience Rating'] = ''
    df_direct['Reservation: Recommend to Friend'] = ''
    df_direct['Reservation: Feedback Notes'] = ''
    df_direct['Title'] = ''
    df_direct['Feedback: Recommend to Friend'] = ''
    df_direct['Reservation: Updated Date'] = ''
    df_direct['Platform'] = df_direct['Source']
    # frop all the initial columns
    df_direct = df_direct.drop(columns=df_initial_columns)
    # # from '' to none
    df_direct = df_direct.replace('', np.nan)
    # # drop all the rows that are empty
    df_direct = df_direct.dropna(how='all')
    # transform the nan into ''
    df_direct = df_direct.fillna('')
    return df_direct

def lambda_for_month(x: pd.Series):
    '''
    This function is used to get the month from the date,
    if the date is empty, it will return "Not Specified"

    It favors the Reservation: Date over the Date Submitted
    
    ---

    params:
        x: the row of the dataframe

    return:
        the month of the date

    '''
    if x['Reservation: Date'] in empty and x['Date Submitted'] not in empty:
        return str(pd.to_datetime(x['Date Submitted']).month)
    if x['Reservation: Date'] not in empty and x['Date Submitted'] in empty:
        return str(pd.to_datetime(x['Reservation: Date']).month)
    if x['Reservation: Date'] not in empty and x['Date Submitted'] not in empty:
        return str(pd.to_datetime(x['Reservation: Date'],dayfirst = True).month)
    return "Not Specified"

def lambda_for_week(x: pd.Series):
    '''
    This function is used to get the week from the date,
    if the date is empty, it will return "Not Specified"

    It favors the Reservation: Date over the Date Submitted

    ---

    params:
        x: the row of the dataframe

    return:
        the week of the date

    '''
    if x['Reservation: Date'] in empty and x['Date Submitted'] not in empty:
        return str(pd.to_datetime(x['Date Submitted']).week)
    if x['Reservation: Date'] not in empty and x['Date Submitted'] in empty:
        return str(pd.to_datetime(x['Reservation: Date']).week)
    if x['Reservation: Date'] not in empty and x['Date Submitted'] not in empty:
        return str(pd.to_datetime(x['Reservation: Date'],dayfirst = True).week)
    return 'Not Specified'

def lambda_for_day_name(x: pd.Series):
    '''
    This function is used to get the day name from the date,
    if the date is empty, it will return "Not Specified"

    It favors the Reservation: Date over the Date Submitted

    ---

    params:
        x: the row of the dataframe
    return:
        the day name of the date
    '''
    if x['Reservation: Date'] in empty and x['Date Submitted'] not in empty:
        return str(pd.to_datetime(x['Date Submitted']).day_name())
    if x['Reservation: Date'] not in empty and x['Date Submitted'] in empty:
        return str(pd.to_datetime(x['Reservation: Date']).day_name())
    if x['Reservation: Date'] not in empty and x['Date Submitted'] not in empty:
        return str(pd.to_datetime(x['Reservation: Date'],dayfirst = True).day_name())
    return 'Not Specified'

def lambda_for_day_part(x: pd.Series):
    '''
    This function is used to get the day part from the time,
    '''
    if x['Reservation: Time'] == "":
        return ""
    return _get_day_part(str(pd.to_datetime(x['Reservation: Time']).hour))

def _get_day_part(hour: str):
    '''
    This function takes the hour as a `str` and returns the day part

    ---
    Parameters:
        hour: str
        the hour of the day

    Returns:
        the day part (Breakfast, Lunch, Dinner, Late Night)
    '''

    if hour in ['Not Specified', '', 'nan']:
        return 'Not Specified'
    if 7 <= int(hour) < 12:
        time_part = 'Breakfast'
    if 12 <= int(hour) < 16:
        time_part = 'Lunch'
    if 16 <= int(hour) < 20:
        time_part = 'Dinner'
    time_part = 'Late Night'
    return time_part

def get_day_part(hour: str):
    '''
    This function takes the hour as a `str` and returns the day part

    ---
      Parameters:
         hour: str
            the hour of the day

      Returns:
         the day part (Breakfast, Lunch, Dinner, Late Night)
    '''
    if hour in ['Not Specified', '', 'nan']:
        return 'Not Specified'
    if 7 <= int(hour) < 12:
        time_part = 'Breakfast'
    if 12 <= int(hour) < 16:
        time_part = 'Lunch'
    if 16 <= int(hour) < 20:
        time_part = 'Dinner'
    time_part = 'Late Night'
    return time_part

def rescoring(df):
    '''
    This function takes the dataframe and rescoring the values of 
    the columns that are related to the ratings

    ---

    df: dataframe
    '''
    columns_to_rescore = ['Feedback: Food Rating', 'Feedback: Drink Rating',
                          'Feedback: Service Rating', 'Feedback: Ambience Rating',
                          'Overall Rating']
    df.loc[:, columns_to_rescore] = df[columns_to_rescore].replace('', 0)
    df.loc[:, columns_to_rescore] = df[columns_to_rescore].astype(float)
    #df.loc[:, columns_to_rescore] = df[columns_to_rescore].replace(value_map)
    return df

def rescoring_empty(df, new = False):
    '''
    This function takes the dataframe and rescoring the values of 
    the columns that are related to the ratings

    Parameters:
        df: dataframe
            the dataframe that we want to apply the rescoring
        new: bool
            if True, it means that the dataframe is the new one, so the 
            columns are different
    '''
    #st.write(df)
    if not new:
        columns_to_rescore = [
            'Feedback: Food Rating', 'Feedback: Drink Rating',
            'Feedback: Service Rating', 'Feedback: Ambience Rating', 'Overall Rating']
    else:
        columns_to_rescore = [
            'Feedback_Food_Rating', 'Feedback_Drink_Rating',
            'Feedback_Service_Rating', 'Feedback_Ambience_Rating', 'Overall_Rating']
    df = df.copy()
    df.loc[:, columns_to_rescore] = df[columns_to_rescore].replace('', 0)
    df.loc[:, columns_to_rescore] = df[columns_to_rescore].astype(float)
    df.loc[:, columns_to_rescore] = df[columns_to_rescore].replace(value_map)
    return df

def clean_label(labels_: str):
    '''This function will clean the labels'''
    if labels_ not in ['', 'nan']:
        labels_ = labels_.split('-')
        labels_ = [l.replace("'", "") for l in labels_]
        labels_ = [l.strip() for l in labels_ if l != '']
        return labels_
    return []

def preprocess_single_df(df):
    '''This function will preprocess the single dataframe'''
    columns_to_keep = [
    'Date Submitted',
    'Title',
    'Details',
    'Overall Rating',
    'Feedback: Food Rating',
    'Feedback: Drink Rating',
    'Feedback: Service Rating',
    'Feedback: Ambience Rating',
    'Feedback: Recommend to Friend',
    'Reservation: Date',
    'Reservation: Venue',
    'Reservation: Time',
    'Reservation: Overall Rating',
    'Reservation: Food Rating',
    'Reservation: Drinks Rating',
    'Reservation: Service Rating',
    'Reservation: Ambience Rating',
    'Reservation: Recommend to Friend',
    'Reservation: Feedback Notes',
    'Reservation: Updated Date',
    'Label: Dishoom',
    '👍',
    '👎',
    '💡',
    'Source',
    'Week',
    'Month',
    'Day_Name',
    'Day_Part',
    'Year',
    'Week_Year',
    'Month_Year',
    'date_for_filter',
    'Suggested to Friend',
    'New Overall Rating',
    'New Food Rating',
    'New Drink Rating',
    'New Service Rating',
    'New Ambience Rating'
    ]
    venue = df["Reservation: Venue"].unique().tolist()
    venue = [v for v in venue if str(v) != 'nan'][0]
    venue = str(venue).replace("'", "")
    df["Reservation: Venue"] = venue
    # add all the columns that we are going to use
    df["Label: Dishoom"] = ["" for i in range(len(df))]
    df['👍'] = False
    df['👎'] = False
    df['💡'] = False
    df['Source'] = df['Platform']
    # ADD: Week, Month, Day_Name, Day_Part, Year, Week_Year, Month_Year, date_for_filter
    # there is this sign / and the opposite \ in the date, so we need to check for both
    df["Week"] = df.apply(lambda_for_week, axis=1)
    df["Month"] = df.apply(lambda_for_month, axis=1)
    df["Day_Name"] = df.apply(lambda_for_day_name, axis=1)
    df['Day_Part'] = df.apply(lambda_for_day_part, axis=1)
    df['Year'] = df.apply(
        lambda x: str(pd.to_datetime(
            x['Date Submitted'],dayfirst = True).year) \
            if x['Reservation: Date'] in empty          \
            else str(pd.to_datetime(x['Reservation: Date'],dayfirst = True).year), axis=1)
    df['Week_Year'] = df.apply(lambda x: x['Week'] + 'W' + x['Year'], axis=1)
    df['Month_Year'] = df.apply(lambda x: x['Month'] + 'M' + x['Year'], axis=1)
    df['date_for_filter'] = df.apply(lambda x: str(
        pd.to_datetime(x['Date Submitted'], dayfirst = True).date()) \
        if x['Reservation: Date'] in empty\
        else str(pd.to_datetime(x['Reservation: Date'],dayfirst = True).date()), axis=1)
    df['Suggested to Friend'] = df['Feedback: Recommend to Friend'].apply(lambda x:
                                                x if x == 'Yes' or x == 'No' else 'Not Specified')
    # initialize the new scoring columns
    df['New Overall Rating'] = 1
    df['New Food Rating'] = 1
    df['New Drink Rating'] = 1
    df['New Service Rating'] = 1
    df['New Ambience Rating'] = 1
    # set all scores to 0
    df = df[columns_to_keep]
    return df

def get_sales_date(store_id, date, time = None):
    '''This function access the google big query and gets the sales data 
    for a specific date and time'''

    try:
        googleconnection = GoogleBigQuery()
        query_for_only_a_date = f'''
        SELECT *,
            EXTRACT(MONTH FROM DateOfBusiness) AS Month
            FROM `sql_server_on_rds.Dishoom_dbo_dpvHstCheckSummary`
            WHERE DateOfBusiness = '{date}'
                AND FKStoreID IN ({','.join([str(i) for i in store_id])})
        '''
        df = googleconnection.query(query = query_for_only_a_date, as_dataframe = True)
        fig, df = TransformationGoogleBigQuery(df, plot = True).transform()
        # add vertical line on time
        if time is not None and time!= 'nan':
            fig.add_vline(x=time, line_width=10, line_color="red", opacity=0.3)
        st.plotly_chart(fig, use_container_width = True)
    except Exception as e:
        st.info('We cant display the data')

def _preprocessing(data):
    '''
    Here we will do the cleaning of the data
    
    - Just filling na with empty string
    ---
    Parameters:
    
        data: pandas dataframe

    Returns:
        data: pandas dataframe
    ---
    '''
    data = data.fillna('nan')
    return data

def _classifing(data):
    '''
    Here we will do the classification of the data
    - Sentiment
    - Confidence
    - Menu Item
    - Keywords
    - Drink Item
    '''
    walla = ArtificialWalla()
    for index, row in data.iterrows():
        sentiment, confidence, menu_items, keywords_, drinks_items = walla.classify_review(
            review = row['Details'])

        values = [
            row['Overall Rating'], row['Feedback: Food Rating'],
            row['Feedback: Drink Rating'], row['Feedback: Service Rating'],
            row['Feedback: Ambience Rating']
            ]
        # replace 5.0 with 5
        values = [str(v) for v in values]
        # replace 5.0 with 5
        values = [v.replace('.0', '') for v in values]
        # if all 5 or 0, then the sentiment is positive
        not_positive_values = ['1', '2', '3', '4']
        if all(v not in not_positive_values for v in values):
            sentiment = 'POSITIVE'
            confidence = 1
        else:
            sentiment = 'NEGATIVE'
            confidence = 1
        data.loc[index, 'Sentiment'] = sentiment
        data.loc[index, 'Confidence'] = confidence
        data.loc[index, 'Menu Item'] = ' '.join(menu_items)
        data.loc[index, 'Keywords'] = ' '.join(keywords_)
        data.loc[index, 'Drink Item'] = ' '.join(drinks_items)

    return data

def process_data(df):
    '''
    Here we run the actual transformation of the data
    Cleaning, Classifying, Rescoring
    '''
    df = _preprocessing(df)
    df = _classifing(df)
    df = rescoring(df)
    return df

def clean_column_entries(review, col_name):
    '''
    This function will clean the column entries
    example:
    'Menu_Item': 'Chicken Ruby - House Black Daal - Dishoom Calamaris'

    will become:
    ['Chicken Ruby', 'House Black Daal', 'Dishoom Calamari']
    '''
    col_values = review[col_name]
    col_values = col_values.split('-') if '-' in col_values else [col_values]
    col_values = [l.strip() for l in col_values if l != '']
    return col_values

def clean_rating_number(rating_n):
    '''This function will clean the rating number'''
    if isinstance(rating_n, str):
        if rating_n == 'nan':
            return 5
        if '.0' in rating_n:
            return int(rating_n.split('.')[0])
        return int(rating_n)
    if isinstance(rating_n, float):
        return int(rating_n)
    return rating_n

def get_descr(value):
    '''This function will get the description of the rating'''
    if value in ['0.0', 0.0, '0', 0]:
        return 'Nan'
    return value
    