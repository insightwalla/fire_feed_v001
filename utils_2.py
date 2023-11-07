from utils import rescoring, lambda_for_day_name, lambda_for_day_part, lambda_for_week, lambda_for_month    , empty
from ai_classifier import ArtificialWalla
import pandas as pd

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
         sentiment, confidence, menu_items, keywords_, drinks_items = walla.classify_review(review = row['Details'])
         columns_for_rating = ['Overall Rating','Feedback: Food Rating', 'Feedback: Drink Rating','Feedback: Service Rating', 'Feedback: Ambience Rating']
         values = [row['Overall Rating'], row['Feedback: Food Rating'], row['Feedback: Drink Rating'], row['Feedback: Service Rating'], row['Feedback: Ambience Rating']]
         # replace 5.0 with 5
         #as strings
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
         '''
         df = _preprocessing(df)
         df = _classifing(df)
         df = rescoring(df)
         return df
       
def preprocess_single_df(df):
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
   # 3. Prepare the dataframes: 
   # add Reservation: Venue when empty (name of the restaurant)
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
   df['Year'] = df.apply(lambda x: str(pd.to_datetime(x['Date Submitted']).year) if x['Reservation: Date'] in empty else str(pd.to_datetime(x['Reservation: Date']).year), axis=1)
   df['Week_Year'] = df.apply(lambda x: x['Week'] + 'W' + x['Year'], axis=1)
   df['Month_Year'] = df.apply(lambda x: x['Month'] + 'M' + x['Year'], axis=1)
   df['date_for_filter'] = df.apply(lambda x: str(pd.to_datetime(x['Date Submitted']).date()) if x['Reservation: Date'] in empty else str(pd.to_datetime(x['Reservation: Date']).date()), axis=1)
   df['Suggested to Friend'] = df['Feedback: Recommend to Friend'].apply(lambda x: x if x == 'Yes' or x == 'No' else 'Not Specified')
   # initialize the new scoring columns
   df['New Overall Rating'] = 1
   df['New Food Rating'] = 1
   df['New Drink Rating'] = 1
   df['New Service Rating'] = 1
   df['New Ambience Rating'] = 1
   # set all scores to 0
   df = df[columns_to_keep]
   # rename all the columns taking off columns spacesand using _ instead
   return df
      

value_map = {
            5: 10,
               4: 8,
                  3: 5,
                   2: 1,
                      1: 1,
                        0: 10
            }
