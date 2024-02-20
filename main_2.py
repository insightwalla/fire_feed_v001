'''
To run this app, run the following command in your terminal:

streamlit run main_2.py
'''
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
import base64
import streamlit_antd_components as sac
from google_big_query import get_direct_feedback, get_google_sheet_data
from templates.ai_mod import ai_template
from graphs import  create_chart_totals_labels, create_container_for_each_sentiment, \
                    create_timeseries_graph, create_graph_keywords_as_a_whole, \
                    create_pie_chart, create_graph_for_source_analysis, \
                    create_graph_for_day_analysis, create_graph_for_hour_analysis,\
                    create_graph_for_week_analysis, create_graph_for_month_analysis
from utils import *

st.set_page_config(layout = 'wide')

@st.cache_resource()
def init_firebase_db():
    '''We read the credentials from secrets.toml file
       and we initialise the connection to the database'''
    cred = credentials.Certificate(dict(st.secrets['firebase']))
    try:
        firebase_admin.initialize_app(cred)
    except ValueError:
        pass
    db = firestore.client()
    return db

db = init_firebase_db()

def add_data(collection : str, data : dict):
    '''
    params:
        collection: name of the collection
        data: dictionary with data

    returns:
        None

    --- 
    
    Assuming a granted connection to the database, this function adds data to a collection.

    example:
    ```
    add_data('conversations', {'type': 'AI', 'content': 'How are you?', 'conversation_id': '1'})
    ```
    '''
    doc_ref = db.collection(collection).document()
    doc_ref.set(data)

def get_data(collection: str, as_dict=False):
    '''
    params:
        collection: name of the collection
        as_dict: if True, returns a list of dictionaries, else returns a list of documents
    returns:
        docs: list of documents or list of dictionaries
    '''
    docs = db.collection(collection).stream()
    if as_dict:
        docs = [doc.to_dict() for doc in docs]
    return docs

def clear_all_collection(collection: str):
    '''
    params:
        collection: name of the collection
    returns:
        None

    ---

    Assuming a granted connection to the database, this function deletes 
    all documents in a collection.

    example:
    ```
    clear_all_collection('conversations')
    ```
    '''
    docs = db.collection(collection).stream()
    for doc in docs:
        doc.reference.delete()

def clear_collection_venue(collection, venue_name):
    '''
    This function deletes all the documents in a collection with a specific venue name
    '''
    docs = db.collection(collection).stream()
    for doc in docs:
        if doc.to_dict()['Reservation_Venue'] == venue_name:
            doc.reference.delete()

def modify_entry(collection: str, idx: str, data: dict):
    '''
    params:
        collection: name of the collection
        idx: id of the feedback
        data: dictionary with data

    returns:
        None

    ---
    
    Assuming a granted connection to the database, this function modifies an entry in a collection.

    '''
    docs = db.collection(collection).stream()
    doc = [doc for doc in docs if doc.to_dict()['idx'] == idx][0]
    old_data = doc.to_dict()
    doc.reference.update(data)
    new_data = db.collection(collection).document(doc.id).get().to_dict()
    # get hte 
    # return old and new data
    return old_data, new_data

def modify_from_details_and_venue(collection: str, details: str, venue: str, data: dict):
    '''
    params:
        collection: name of the collection
        details: details of the feedback
        venue: name of the venue
        data: dictionary with data

    returns:
        None

    ---
    
    Assuming a granted connection to the database, this function modifies an entry in a collection.

    '''
    docs = db.collection(collection).stream()
    doc = [doc for doc in docs if doc.to_dict()['Details'] == details and doc.to_dict()['Reservation_Venue'] == venue][0]
    old_data = doc.to_dict()
    doc.reference.update(data)
    new_data = db.collection(collection).document(doc.id).get().to_dict()
    # get hte 
    # return old and new data
    return old_data, new_data

def clear_agent_from_name(collection: str, name: str):
    '''
    params:
        collection: name of the collection
        name: name of the agent

    returns:
        None

    ---

    Assuming a granted connection to the database, 
    this function deletes all documents in a collection.

    example:
    ```
    clear_all_collection('conversations')
    ```
    '''
    docs = db.collection(collection).stream()
    for doc in docs:
        if doc.to_dict()['name'] == name:
            doc.reference.delete()

def main():
    '''This is the main function of the app'''
    collection_name = 'testing_feedback_v01'

    def create_week_year_dict():
        dates = pd.date_range(start='2022-12-01', end=pd.to_datetime('today').date(), freq='D')
        dates = pd.DataFrame(dates, columns=['date'])
        dates['week_year'] = dates['date'].apply(lambda x: 'W' + str(x.week) + 'Y' + str(x.year))
        dates = dates.groupby('week_year').agg({'date': ['min', 'max']})
        dates = dates.sort_values(ascending=False, by=[('date', 'min')])
        week_year_dict = {}
        for index, row in dates.iterrows():
            week_year_dict[index] = {
            'start_date': row['date']['min'].date(),
            'end_date': row['date']['max'].date()
            }
        return week_year_dict

    def get_selected_dates(week_year_dict, c1):
        week_years = list(week_year_dict.keys())
        week_year = c1.selectbox('Choose the week year', week_years, index=0)
        start_date = week_year_dict[week_year]['start_date']
        end_date = week_year_dict[week_year]['end_date']
        return start_date, end_date

    def get_selected_venue(c2, venues):
        venues_list = list(venues.keys())
        venue = c2.selectbox('Choose the venue', venues_list, index=0)
        return venue

    def adding_data():
        collection_name = 'testing_feedback_v01'
        week_year_dict = create_week_year_dict()
        c1,c2 = st.columns(2)
        start_date, end_date = get_selected_dates(week_year_dict, c1)
        st.write(start_date, end_date)
        start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
        end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')
        start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
        end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')
        # select the venues

        venues = {
        'Dishoom Covent Garden': '`jp-gs-379412.sevenrooms_covent_garden.reservation_feedback`',
        'Dishoom Shoreditch': '`jp-gs-379412.sevenrooms_shoreditch.reservation_feedback`',
        'Dishoom Kings Cross': '`jp-gs-379412.sevenrooms_kings_cross.reservation_feedback`',
        'Dishoom Carnaby':    '`jp-gs-379412.sevenrooms_carnaby.reservation_feedback`',
        'Dishoom Edinburgh': '`jp-gs-379412.sevenrooms_edinburgh.reservation_feedback`',
        'Dishoom Kensington': '`jp-gs-379412.sevenrooms_kensington.reservation_feedback`',
        'Dishoom Manchester': '`jp-gs-379412.sevenrooms_manchester.reservation_feedback`',
        'Dishoom Birmingham': '`jp-gs-379412.sevenrooms_birmingham.reservation_feedback`',
        'Dishoom Canary Wharf': '`jp-gs-379412.sevenrooms_canary_wharf.reservation_feedback`',
        'Dishoom Permit Room Brighton': '`jp-gs-379412.sevenrooms_permit_room_brighton.reservation_feedback`'
        }
        # use it to query the data

        with st.form('query'):
            venue = get_selected_venue(c2, venues)
            query = f'''
            select
               *
            from
               {venues[venue]}
            where
                parse_date('%d/%m/%Y', received_date) >= '{start_date}'
                and parse_date('%d/%m/%Y', received_date) <= '{end_date}'
            '''
            submit = st.form_submit_button(
                f'Prepare **{venue}** (**{start_date}** - **{end_date}**)', 
                use_container_width=True,
                type='primary')

            if submit:
                data = GoogleBigQuery().query(query)
                data_direct_feedback = get_direct_feedback(
                    venue, start_date=start_date, end_date=end_date
                    )
                data_direct_feedback = process_direct_f(
                    data_direct_feedback
                    )
                if len(data) == 0:
                    st.info('No data found for the selected dates')
                    st.stop()
                # add to the data the direct feedback
                # reset index
                data = prepare_data(data)
                data = pd.concat([data, data_direct_feedback], axis=0)
                # reset index
                data = data.reset_index(drop=True)
                data_processed = preprocess_single_df(data)
                st.session_state.data = data_processed

            if 'data' in st.session_state:
                st.write(st.session_state.data)

            if st.form_submit_button(
                    label = 'Add the data',
                    use_container_width =True,
                    type = 'primary'
                    ) and 'data' in st.session_state:

                df = st.session_state.data
                df['idx'] = list(range(len(df)))
                df = process_data(df)
                df = df.rename(columns=lambda x: x.replace(':', '').replace('(', '').replace(')', '').replace(' ', '_'))

                # transform all in strings
                for col in df.columns.tolist():
                    df[col] = df[col].astype(str)

                # check if the doc exists
                # get all reviews from db with the same name
                data = get_data(collection_name)
                unique_venues = list(set([doc.to_dict()['Reservation_Venue'] for doc in data]))
                if venue in unique_venues:
                    with st.spinner('Clearing data...'):
                        clear_collection_venue(collection_name, venue)

                with st.spinner('Adding data...'):
                    # get totla number of rows
                    len_df = len(df)
                    my_small_bar = st.progress(0, text=f'Uploading 0/{len_df}')
                    for i, row in df.iterrows():
                        add_data(collection_name, row.to_dict())
                        my_small_bar.progress((i+1)/len_df, text=f'Uploading {i+1}/{len_df}')
                    my_small_bar.progress(100, text='Upload Completed')
                    st.balloons()

    def adding_data_from_google_sheet():
        '''
        Almost ready to be used

        # double entry issue -> check same venue name
        # need to add the possibility to add ALL the venue in the same moment
        # need to add the the direct feedback
        '''
        data = get_google_sheet_data()
        # no empty rows
        data = data.dropna(how='all')
        venues = data["Reservation: Venue"].unique().tolist()
        #add all and remove nan
        venues = ['All'] + [venue for venue in venues if str(venue) != 'nan']
        st.write(venues)

        # choose the venue
        with st.form('query'):
            venue = st.selectbox('Choose the venue', options = venues, index = 0)
            submit = st.form_submit_button(
                f'Prepare **{venue}**', 
                use_container_width=True,
                type='primary')
            
            if submit:
                data = prepare_data_from_gsheets(data)
                data = preprocess_single_df(data)
                st.session_state.data = data
                st.write(st.session_state.data)

            if st.form_submit_button(
                    label = 'Add the data',
                    use_container_width =True,
                    type = 'primary'
                    ) and 'data' in st.session_state:

                df = st.session_state.data
                df['idx'] = list(range(len(df)))
                df = process_data(df)
                df = df.rename(columns=lambda x: x.replace(':', '').replace('(', '').replace(')', '').replace(' ', '_'))

                # transform all in strings
                for col in df.columns.tolist():
                    df[col] = df[col].astype(str)

                # check if the doc exists
                data = get_data(collection_name)
                unique_venues = list(set([doc.to_dict()['Reservation_Venue'] for doc in data]))
                if venue in unique_venues:
                    with st.spinner('Clearing data...'):
                        clear_collection_venue(collection_name, venue)

                with st.spinner('Adding data...'):
                    # get totla number of rows
                    len_df = len(df)
                    my_small_bar = st.progress(0, text=f'Uploading 0/{len_df}')
                    for i, row in df.iterrows():
                        add_data(collection_name, row.to_dict())
                        my_small_bar.progress((i+1)/len_df, text=f'Uploading {i+1}/{len_df}')
                    my_small_bar.progress(100, text='Upload Completed')
                    st.balloons()
    
    def adding_data_from_google_sheet_all_option():
        '''
        Almost ready to be used

        # double entry issue -> check same venue name
        # need to add the possibility to add ALL the venue in the same moment
        # need to add the the direct feedback
        '''
        data = get_google_sheet_data()
        data = data.dropna(how='all')
        # take off the '
        data['Reservation: Venue'] = data['Reservation: Venue'].str.replace("'", "")
        venues = data["Reservation: Venue"].unique().tolist()
        venues = ['All'] + [venue for venue in venues if str(venue) != 'nan']
        #st.write(venues)

        # choose the venue
        with st.form('query'):
            venue = st.selectbox('Choose the venue', options = venues, index = 0)
            submit = st.form_submit_button(
                f'Prepare **{venue}**', 
                use_container_width=True,
                type='primary')
            
            if submit:
                if venue == 'All':
                    # create a tab for each venue
                    tabs = st.tabs(venues[1:] + ['Direct Feedback'])
                    # direct feedback
                    df_dir = get_direct_feedback()
                    df_dir = process_direct_f(df_dir)
                    st.session_state['Direct Feedback'] = df_dir
                    with tabs[-1]:
                        st.write(st.session_state['Direct Feedback'])

                    for i, venue in enumerate(venues[1:]):
                        # select the data for that venue
                        data_for_venue = st.session_state['Direct Feedback']
                        df = data[data['Reservation: Venue'] == venue]

                        if len(data_for_venue) > 0:
                            data_for_venue = data_for_venue[data_for_venue['Reservation: Venue'] == venue]
                            # append to the data
                        df = pd.concat([df, data_for_venue], axis=0)
                            # need to reset the index
                            #df = df.reset_index(drop=True)
                        df = prepare_data_from_gsheets(df)
                        df = preprocess_single_df(df)
                        # get direct feedback
 
                        st.session_state[venue] = df
                        # write in tab
                        with tabs[i]:
                            st.write(f'{venue} : {len(df)} reviews')
                            st.write(st.session_state[venue])


                    # now add the data from direct feedback to venue filtering from the venue
                else:
                    df_dir = get_direct_feedback()
                    df_dir = process_direct_f(df_dir)
                    df = data[data['Reservation: Venue'] == venue]
                    st.session_state['Direct Feedback'] = df_dir
                    # need to put together the data
                    df_dir_for_venue = df_dir[df_dir['Reservation: Venue'] == venue]
                    if len(df_dir_for_venue) > 0:
                        # combine the data
                        df = pd.concat([df, df_dir_for_venue], axis=0)

                    df = prepare_data_from_gsheets(df)
                    df = preprocess_single_df(df)
                    st.session_state[venue] = df
                    st.write(st.session_state[venue])

            if st.form_submit_button(
                    label = 'Add the data',
                    use_container_width =True,
                    type = 'primary'
                    ):
                
                if venue == 'All':
                    # for each venue
                    final_df = pd.DataFrame()
                    for v in venues[1:]:
                        # select the data for that venue
                        df = st.session_state[v]
                        df['idx'] = list(range(len(df)))
                        df = process_data(df)
                        df = df.rename(columns=lambda x: x.replace(':', '').replace('(', '').replace(')', '').replace(' ', '_'))
                        # transform all in strings
                        for col in df.columns.tolist():
                            df[col] = df[col].astype(str)

                        # check if the doc exists
                        data = get_data(collection_name)
                        unique_venues = list(set([doc.to_dict()['Reservation_Venue'] for doc in data]))
                        if v in unique_venues:
                            with st.spinner('Clearing data...'):
                                clear_collection_venue(collection_name, v)

                        # add to the final df
                        final_df = pd.concat([final_df, df], axis=0)

                        with st.spinner('Adding data...'):
                            # get totla number of rows
                            len_df = len(final_df)
                            my_small_bar = st.progress(0, text=f'Uploading 0/{len_df}')
                            for i, row in df.iterrows():
                                add_data(collection_name, row.to_dict())
                                my_small_bar.progress(min((i+1)/len_df, 1.0), text=f'Uploading {i+1}/{len_df}')
                            my_small_bar.progress(100, text=f'{v} - Upload Completed')
                    st.balloons()

                else:
                    v = venue
                    df = st.session_state[v]
                    df['idx'] = list(range(len(df)))
                    df = process_data(df)
                    df = df.rename(columns=lambda x: x.replace(':', '').replace('(', '').replace(')', '').replace(' ', '_'))

                    # transform all in strings
                    for col in df.columns.tolist():
                        df[col] = df[col].astype(str)

                    # check if the doc exists
                    data = get_data(collection_name)
                    unique_venues = list(set([doc.to_dict()['Reservation_Venue'] for doc in data]))
                    if v in unique_venues:
                        with st.spinner('Clearing data...'):
                            clear_collection_venue(collection_name, venue)

                    with st.spinner('Adding data...'):
                        # get totla number of rows
                        len_df = len(df)
                        my_small_bar = st.progress(0, text=f'Uploading 0/{len_df}')
                        for i, row in df.iterrows():
                            add_data(collection_name, row.to_dict())
                            my_small_bar.progress(min((i+1)/len_df, 1.0), text=f'Uploading {i+1}/{len_df}')
                        my_small_bar.progress(100, text=f'{v} - Upload Completed')
                        st.balloons()

    def filter_data_by_venue(data, venue):
        return [doc for doc in data if doc['Reservation_Venue'] == venue]

    def filter_data_by_sentiment(data, sentiment):
        if sentiment == 'Negative':
            return [doc for doc in data if doc['Sentiment'] == 'NEGATIVE']
        if sentiment == 'Positive':
            return [doc for doc in data if doc['Sentiment'] == 'POSITIVE']
        return data

    def filter_data_by_search(data, search):
        if search != '':
            return [doc for doc in data if search.lower() in doc['Details'].lower()]
        return data

    def get_review_id(c2, data, map_id):
        '''
        Handle error more precisely:
        # case 1: review_id is not a number
        # case 2: review_id is not in the range
        # case 3: review_id is not a number and not in the range
        '''
        try:
            review_id = c2.number_input(
                f'Review: {st.session_state.get("review_id", 1)}/{len(data)}', 
                min_value = 1,
                max_value = len(data),
                value= st.session_state.get('review_id', 1)
                )
        except:
            st.stop()
        return map_id[review_id-1]

    def get_review_data(data, review_id):
        return [doc for doc in data if doc['idx'] == review_id][0]

    def edit_data():
        data = get_data(collection_name, as_dict=True)
        if len(data) == 0:
            st.info('Go to the Upload section and select some!')
            st.stop()
        else:
            pass

        search_container = st.container()
        graph_container = st.container()
        c1,c2 = st.columns(2)
        venue = c1.selectbox('Select venue', list(set([doc['Reservation_Venue'] for doc in data])))
        data = filter_data_by_venue(data, venue)
        # as a dataframe
        df_full_ = pd.DataFrame(data)
        best_rev = df_full_[df_full_['👍'] == '1']
        worst_rev = df_full_[df_full_['👎'] == '1']
        suggestions_rev = df_full_[df_full_['💡'] == '1']

        if len(data) == 0:
            st.info('No data available, need to upload some reviews!')
            st.stop()

        with st.sidebar.expander('Filtering'):
            only_to_label = st.toggle('Filter by empty label', value=False)
            if only_to_label:
                data = [doc for doc in data if doc['Label_Dishoom'] == '']
                data = [doc for doc in data if doc['Sentiment'] == 'NEGATIVE']
                st.session_state.review_id = 1
            sent =  st.selectbox(
                label= 'Filter by sentiment',
                options = ['All', 'Negative', 'Positive'],
                key='sentiment')
            data = filter_data_by_sentiment(data, sent)

        search = search_container.text_input(
            label = 'Search',
            key = 'search',
            placeholder = 'Search for a word in the review'
            )
        data = filter_data_by_search(data, search)

        df = pd.DataFrame(data)
        data = [doc for doc in data if doc['Details'] != '']
        df_empty = df[df['Details'] == '']
        df_full = df[df['Details'] != '']
        df_empty = rescoring_empty(df_empty, new = True)
        with graph_container:
            try:
                create_container_for_each_sentiment(df_full,df_empty)
            except Exception as e:
                st.write(e)

        map_id = {i: doc['idx'] for i, doc in enumerate(data)}
        id_map = {doc['idx']: i for i, doc in enumerate(data)}
        # initialise the review id
        if 'review_id' not in st.session_state:
            st.session_state.review_id = 1


        map_id = {i: doc['idx'] for i, doc in enumerate(data)}
        review_id = get_review_id(c2, data, map_id)
        review = get_review_data(data, review_id)
        #st.write(review)
        #st.write(review)
        edit_tab, venue_tab = st.tabs(['Edit', 'Venue Details'])
        with edit_tab.form(key='my_editing_form', clear_on_submit=False):
            _, space_for_update_button = st.columns([0.7,0.3])
            # get all the informations


            is_this_best = review['👍'] == '1'
            is_this_worst = review['👎'] == '1'
            is_this_suggestion = review['💡'] == '1'
            col_best, col_worst, col_sugg = st.columns(3)

            if len(best_rev) >= 3 and not is_this_best:
                is_best = False
                col_best.info('Already selected 3 👍')
            else:
                is_best = col_best.toggle(
                    label ='👍',
                    value = is_this_best,
                    key = f'is_good{review_id}={venue}'
                    )
            if len(worst_rev) >= 3 and not is_this_worst:
                is_worst = False
                col_worst.info('Already selected 3 👎')
            else:
                is_worst = col_worst.toggle(
                    label = '👎',
                    value = is_this_worst,
                    key= f'is_bad{review_id}={venue}'
                    )
            if len(suggestions_rev) >= 3 and not is_this_suggestion:
                is_suggestion = False
                col_sugg.write('Already selected 3 💡')
            else:
                is_suggestion = col_sugg.toggle(
                    label = '💡',
                    value = is_this_worst,
                    key= f'is_suggestion{review_id}={venue}'
                    )
            c1,c2,c3,c4,c5 = st.columns(5)
            st.markdown('**Review**')
            st.markdown(review['Details'])

            # try transform from str to float '1.0' -> 1.0
            review['New_Overall_Rating'] = float(review['New_Overall_Rating'])
            review['New_Food_Rating'] = float(review['New_Food_Rating'])
            review['New_Drink_Rating'] = float(review['New_Drink_Rating'])
            review['New_Service_Rating'] = float(review['New_Service_Rating'])
            review['New_Ambience_Rating'] = float(review['New_Ambience_Rating'])
            

            with c1:
                overall = sac.rate(
                    label=f'Overall Rating: **{get_descr(review["Overall_Rating"])}**',
                    value=int(review['New_Overall_Rating']),
                    count=value_map[clean_rating_number(review['Overall_Rating'])],
                    key=f'{review["idx"]}_overall'
                    )
            with c2:
                food = sac.rate(
                    label=f'Food Rating: **{get_descr(review["Feedback_Food_Rating"])}**',
                    value=int(review['New_Food_Rating']),
                    count=value_map[clean_rating_number(review['Feedback_Food_Rating'])],
                    key=f'{review["idx"]}_food'
                    )
            with c3:
                drink = sac.rate(
                    label=f'Drink Rating: **{get_descr(review["Feedback_Drink_Rating"])}**',
                    value=int(review['New_Drink_Rating']),
                    count=value_map[clean_rating_number(review['Feedback_Drink_Rating'])],
                    key=f'{review["idx"]}_drink'
                    )
            with c4:
                service = sac.rate(
                    label=f'Service Rating: **{get_descr(review["Feedback_Service_Rating"])}**',
                    value=int(review['New_Service_Rating']),
                    count=value_map[clean_rating_number(review['Feedback_Service_Rating'])],
                    key=f'{review["idx"]}_service'
                    )
            with c5:
                ambience = sac.rate(
                    label=f'Ambience Rating: **{get_descr(review["Feedback_Ambience_Rating"])}**',
                    value=int(review['New_Ambience_Rating']),
                    count=value_map[clean_rating_number(review['Feedback_Ambience_Rating'])],
                    key = f'{review["idx"]}_ambience'
                    )

            col1,col2 = st.columns(2)
            new_food_items = col1.multiselect(
                'Menu Items', 
                default=clean_column_entries(review, 'Menu_Item'),
                options=menu_items_lookup,
                key=f'{review["idx"]}_menu_items'
                )
            new_drinks_items = col2.multiselect(
                'Drinks Items', 
                default=clean_column_entries(review, 'Drink_Item'),
                options=drink_items_lookup,
                key=f'{review["idx"]}_drink_items'
                )
            new_keywords = st.multiselect(
                'Keywords', 
                default=clean_column_entries(review, 'Label_Dishoom'),
                options=options_for_classification,
                key=f'{review["idx"]}_keywords'
                )

            data = {
                    'New_Overall_Rating': str(overall),
                    'New_Food_Rating': str(food),
                    'New_Drink_Rating': str(drink),
                    'New_Service_Rating': str(service),
                    'New_Ambience_Rating': str(ambience),
                    'Reservation_Venue': str(venue),
                    'Menu_Item': ' - '.join(new_food_items),
                    'Drink_Item':  ' - '.join(new_drinks_items),
                    'Label_Dishoom':' - '.join(new_keywords),
                    '👍': '1' if is_best else '0',
                    '👎': '1' if is_worst else '0',
                    '💡': '1' if is_suggestion else '0',
                    }
            if space_for_update_button.form_submit_button(
                            #label = f'Edit {review_id} - {venue} : {collection_name}',
                            label = 'Save',
                            type = 'primary',
                            use_container_width=True
                            ):
                with st.spinner(f'Updating {review_id} - {collection_name}'):
                    # map review_id to idx
                    #old_value, new_v = modify_entry(collection_name, review_id, data)
                    old_value, new_v = modify_from_details_and_venue(collection_name, review['Details'], venue, data)
                    with st.expander(f'Old Value {old_value["Details"] if old_value["Details"] != "" else "nan"}'):
                        for k, v in old_value.items():
                            st.write(f'{k} : {v}')
                    with st.expander(f'New Value {new_v["Details"] if new_v["Details"] != "" else "nan"}'):
                        for k, v in new_v.items():
                            st.write(f'{k} : {v}')
                st.toast('Data edited', icon='✅')
                df_to_label = df_full[df_full['Label_Dishoom'] == '']
                df_to_label = df_to_label[df_to_label['Sentiment'] == 'NEGATIVE']
                if len(df_to_label) == 0 and only_to_label:
                    st.balloons()
                    st.stop()
                else:
                    pass

        with venue_tab:
            venue_map = {
               'Dishoom Covent Garden': 1,
               'Dishoom Shoreditch': 2,
               'Dishoom Kings Cross': 3,
               'Dishoom Carnaby': 4,
               'Dishoom Edinburgh': 5,
               'Dishoom Kensington': 6,
               'Dishoom Manchester': 7,
               'Dishoom Birmingham': 8,
               'Dishoom Canary Wharf': 9,
               'Dishoom Battersea': 10,
               'Permit Room Brighton': 11
            }

            # get the id from the name
            store_id = venue_map[venue]
            date = review['Reservation_Date']
            time = review['Reservation_Time']
            #get_sales_date(store_id= [store_id], date = date, time = time)
            st.stop()

    def clear_data():
        with st.form(key='my_clearing_form'):
            st.subheader('Clear data')
            # clear all data
            if st.form_submit_button('Clear all', use_container_width = True, type = 'primary'):
                with st.spinner('Deleting All...'):
                    clear_all_collection(collection_name)
                st.success('All data cleared')
                st.balloons()

    def create_sidebar_menu():
        with st.sidebar:
            menu = sac.menu([
                sac.MenuItem('Feedback', icon='database', children=[
                    sac.MenuItem('Scoring', icon='brush'),
                    sac.MenuItem('Upload', icon='upload'),
                    sac.MenuItem('Download', icon='download'),
                    sac.MenuItem('Clear', icon='trash'),
                ]),
                sac.MenuItem('AI Assistant', icon='robot'),
                sac.MenuItem('Reporting', icon='share'),
                #sac.MenuItem('Settings', icon='gear'),
            ], open_all=False)
            with st.expander('Rating Guide'):
                st.write('**5** : **10**')
                st.write('**4** : **8**')
                st.write('**3** : **5**')
                st.write('**2** : **1**')
                st.write('**1** : **1**')
            return menu

    def plot(df):
        final = df[df['Details'] != 'nan']
        container_keywords = st.sidebar.container()
        with st.expander('Graphs 📉', expanded=False): # graph emoji 📈 or 📊 or 📉
            tabs = st.tabs(['Graphs', 'Keywords', 'Pie Chart',
                            'Source Analysis', 'Day Analysis',
                            'Hour Analysis', 'Week Analysis',
                            'Month Analysis', 'Totals'])
            with tabs[0]:
                create_timeseries_graph(final)
            with tabs[1]:
                create_graph_keywords_as_a_whole(final)
            with tabs[2]:
                create_pie_chart(final)
            with tabs[3]:
                create_graph_for_source_analysis(final)
            with tabs[4]:
                create_graph_for_day_analysis(final)
            with tabs[5]:
                create_graph_for_hour_analysis(final)
            with tabs[6]:
                create_graph_for_week_analysis(final)
            with tabs[7]:
                create_graph_for_month_analysis(final)
            with tabs[8]:
                create_chart_totals_labels(final)

    def get_venue_data(data, venue):
        venue_data = data[data['Reservation_Venue'] == venue]
        venue_data_to_lab = venue_data[venue_data['Details'] != '']
        venue_data_to_lab = venue_data_to_lab[venue_data_to_lab['Details'] != 'nan']
        return venue_data_to_lab, venue_data

    def get_totals(venue_data_to_lab, venue_data):
        negative_to_lab = venue_data_to_lab[venue_data_to_lab['Sentiment'] == 'NEGATIVE']
        tot_ = len(negative_to_lab) + 6 
        tot_done = len(negative_to_lab[negative_to_lab['Label_Dishoom'] != ''])
        tot_not_done = len(negative_to_lab[negative_to_lab['Label_Dishoom'] == ''])
        thumbs_up = venue_data[venue_data['👍'] == '1']
        thumbs_down = venue_data[venue_data['👎'] == '1']
        tot_done += len(thumbs_up) + len(thumbs_down) 
        suggestions = venue_data[venue_data['💡'] == '1']
        number_of_thumbs_up = len(thumbs_up)
        number_of_thumbs_down = len(thumbs_down)
        return tot_, tot_done, tot_not_done, thumbs_up, thumbs_down, suggestions, number_of_thumbs_up, number_of_thumbs_down

    def reporting():
        '''
        1. Read the data from the database
        2. Create a dataframe
        3. Sort the dataframe by idx
        4. Create a container for each sentiment
        5. Create a delete button
        '''
        data = get_data(collection_name, as_dict=True)
        df = pd.DataFrame(data)
        if len(df) ==  0:
            st.info('Update some reviews, you can do it in the Upload Section')
            st.stop()
        data = df
        with st.spinner('Processing data...'):
            plot(data)

        # get unique venues
        list_of_venue = data['Reservation_Venue'].unique()
        for _, venue in enumerate(list_of_venue):
            venue_data_to_lab, venue_data = get_venue_data(data, venue)
            tot_, tot_done, tot_not_done, thumbs_up, thumbs_down, suggestions, number_of_thumbs_up, number_of_thumbs_down = get_totals(venue_data_to_lab, venue_data)

            # get suggestions
            suggestions = venue_data_to_lab[venue_data_to_lab['💡'] == '1']
            number_of_thumbs_up = len(thumbs_up)
            number_of_thumbs_down = len(thumbs_down)

            try:
                percentage_completion = round(tot_done/tot_*100, 0 )
                # if greater than 100, set to 100
                if percentage_completion > 100:
                    percentage_completion = 100
                message = venue + f' **{round(percentage_completion, 0)}%**'
            except ZeroDivisionError:
                message = venue

            with st.expander(message):
                negative_reviews = venue_data_to_lab[venue_data_to_lab['Sentiment'] == 'NEGATIVE']
                tot_already_done = len(negative_reviews[negative_reviews['Label_Dishoom'] != ''])
                tab_pie, tab_good, tab_bad, tab_sugg, tab_g = st.tabs(
                    [f'Reviews {len(negative_reviews)}/{tot_already_done}',
                    f'Good {number_of_thumbs_up}/3',
                    f'Bad {number_of_thumbs_down}/3',
                    f'Suggestions {len(suggestions)}',
                    'Labelling Graphs',
                    ])


                with tab_pie:
                    # now create a pie chart
                    fig = go.Figure()
                    fig.add_trace(go.Pie(
                        labels=['Done', 'Not Done'],
                        values=[tot_done, tot_not_done]))
                    fig.update_layout(title_text=venue)
                    # green for done, red for not done
                    fig.update_traces(marker_colors=['green', 'red'])
                    # set opacity
                    fig.update_traces(opacity=0.6, textinfo='percent+label')
                    # set size 200x200
                    st.plotly_chart(fig, use_container_width=True)


                with tab_good:
                    for good in thumbs_up['Details'].tolist():
                        st.write(good)
                        st.write('---')

                with tab_bad:
                    for bad in thumbs_down['Details'].tolist():
                        st.write(bad)
                        st.write('---')

                with tab_sugg:
                    for sugg in suggestions['Details'].tolist():
                        st.write(sugg)
                        st.write('---')

                with tab_g:
                    create_chart_totals_labels(venue_data_to_lab)

        # add a complete df
        with st.expander('View all data'):
            data = data.rename(
                columns={
                    'Overall Rating': 'Overall', 
                    'Feedback: Food Rating': 'Food', 
                    'Feedback: Service Rating': 'Service', 
                    'Feedback: Ambience Rating': 'Ambience',
                    'Feedback: Drink Rating': 'Drink'})
            st.write(data)

        # create a download link
        def get_table_download_link(data):
            # rename the columns that have emoji
            data = data.rename(columns={'👍': 'thumbs_up', '👎': 'thumbs_down', '💡': 'suggestions'})
            # create a link to download the dataframe
            csv = data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download csv file</a>'
            return href
        st.markdown(get_table_download_link(data), unsafe_allow_html=True)

    def download():
        c1,c2,c3 = st.columns([0.3,0.5, 0.2])

        c1.subheader('Download')
        data = get_data(collection_name, as_dict=True)
        df = pd.DataFrame(data)
        if len(df) == 0:
            st.info('No data found - Please select Upload to upload the data')
            st.stop()
        df = df.sort_values(by=['idx'])
        all_venues = df['Reservation_Venue'].unique().tolist()
        all_venues =  ['All'] + all_venues

        if len(df) == 0:
            st.info('No data found - Please select Upload to upload the data')
            st.stop()
        df = df.sort_values(by=['idx'])

        venue = st.selectbox('Select the Venue', options = all_venues, index = 0)
        if venue != 'All':
            df = df[df['Reservation_Venue'] == venue]

        name_file = c2.text_input(
            label = 'File Name',
            value ='labelled_reviews' if venue == 'All' else f'labelled_rev_{venue}')

        @st.cache_data
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df)

        c3.download_button(
            label="Download data as CSV",
            data=csv,
            file_name=name_file,
            mime='text/csv',
            type = 'primary'
        )
        st.write(f'{len(df)} Reviews')
        st.write(df)
        plot(df)

        st.stop()

    menu = create_sidebar_menu()
    if menu == 'Upload':
        adding_data_from_google_sheet_all_option()

    elif menu == 'Scoring':
        edit_data()

    elif menu == 'Clear':
        clear_data()

    elif menu == 'Reporting':
        reporting()

    elif menu == 'AI Assistant':
        # get data
        #st.info('OpenAI release a new version of their API last week - Need some time to change the logic!')
        #st.stop()
        data = get_data(collection_name, as_dict=True)
        if len(data) == 0:
            st.info('Upload some data first!')
            st.stop()
        df = pd.DataFrame(data)
        df = df[df['Details'] != '']
        df = df[df['Details'] != 'nan']
        df = df[df['Label_Dishoom'] == '']
        df = df[df['Sentiment'] == 'NEGATIVE']
        # keep only venue and details columns
        df = df[['Reservation_Venue', 'Details']]
        st.session_state.data_for_ai = df
        ai_template(st.session_state.data_for_ai)

    elif menu == 'Download':
        download()

if __name__ == '__main__':
    from login_light import login
    login(render_func=main)
