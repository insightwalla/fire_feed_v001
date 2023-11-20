'''
This file contains all the graphs that are used in the app
'''
from collections import Counter
import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from utils import clean_label

def create_graph_keywords_as_a_whole(df):
    '''This function creates a graph with the keywords and the sentiment'''
    feat_for_key = ['Keywords', 'Sentiment']
    keywords_columns_sentiment = df[feat_for_key].copy()
    keywords_columns_sentiment.loc[:, 'Keywords'] = keywords_columns_sentiment['Keywords'].apply(
        lambda x: x.split('-'))
    keywords_columns_sentiment = keywords_columns_sentiment.explode('Keywords')
    keywords_columns_sentiment = keywords_columns_sentiment.groupby(
       ['Keywords', 'Sentiment']).size().reset_index(name='Count')
	# order by total
    keywords_columns_sentiment = keywords_columns_sentiment.sort_values(
        by=['Count'], ascending=False)
    fig = go.Figure()
    color = {'POSITIVE': 'green', 'NEGATIVE': 'red', 'neutral': 'lightblue'}
    for sentiment in keywords_columns_sentiment['Sentiment'].unique():
        sentiment_df = keywords_columns_sentiment[
            keywords_columns_sentiment['Sentiment'] == sentiment
            ]
        fig.add_trace(go.Bar(
            x=sentiment_df['Keywords'],
            y=sentiment_df['Count'], name=sentiment,
            opacity=0.5, marker_color=color[sentiment])
            )
    fig.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'})
    fig.update_layout(xaxis_tickangle=-45)
    # no legend and title
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def create_timeseries_graph(df):
    '''This function creates a graph with the keywords and the sentiment'''
    # Convert dates to datetime and extract date
    for col in ['Date_Submitted', 'Reservation_Date']:
        df[col] = pd.to_datetime(df[col], dayfirst=True).dt.date

    # Create a new column called date_to_plot, is the reservation date if it is not empty,
    # otherwise is the date submitted
    df['Date_to_plot'] = df['Reservation_Date'].fillna(df['Date_Submitted'])

    # Remove empty details
    df = df[df['Details'] != '']

    df = df.groupby(['Date_to_plot', 'Sentiment']).size().reset_index(name='Count')

    fig = go.Figure()
    color = {'POSITIVE': 'green', 'NEGATIVE': 'red', 'neutral': 'lightblue'}
    for sentiment in df['Sentiment'].unique():
        sentiment_df = df[df['Sentiment'] == sentiment]
        fig.add_trace(go.Bar(
            x=sentiment_df['Date_to_plot'],
            y=sentiment_df['Count'],
            name=sentiment, opacity=0.5,
            marker_color=color[sentiment])
            )

    fig.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'})
    fig.update_layout(xaxis_tickangle=-45)
    # title is "Distribution by date"
    fig.update_layout(title_text='Distribution by Date')
    st.plotly_chart(fig, use_container_width=True)

def create_container_for_each_sentiment(df, df_empty = None):
    '''This function creates a container for each sentiment and inside each container
    it creates a table with the reviews for that sentiment'''
    # plot total negative, neutral, positive
    sentiments = df['Sentiment'].unique()
    number_of_total_reviews = len(df)
    with st.expander(f'Reviews **{number_of_total_reviews + len(df_empty)}**'):
        tabs = st.tabs([
            f'{sentiment} **{len(df[df["Sentiment"]==sentiment])}**' for sentiment in sentiments]+\
            [f'Empty **{len(df_empty)}**'])

        columns_to_keep = [
            'Details', 'Sentiment', 'Feedback_Food_Rating', 
            'Feedback_Drink_Rating', 'Feedback_Service_Rating', 
            'Feedback_Ambience_Rating', 'Overall_Rating', 
            'Date_Submitted', 'Reservation_Date', 'Day_Name', 
            'Day_Part', 'Source'
            ]
        for sentiment, tab in zip(sentiments, tabs[:-1]):
            with tab:
                st.write(df[df['Sentiment'] == sentiment][columns_to_keep])
        with tabs[-1]:
            df = df_empty
            if len(df) > 0:
                scores = []
                columns_to_rescore = [
                    'Feedback_Food_Rating', 'Feedback_Drink_Rating', 'Feedback_Service_Rating',
                    'Feedback_Ambience_Rating', 'Overall_Rating']
                # transform in int if possible
                # if value in column to rescore is empty then change to 0
                for column in columns_to_rescore:
                    if isinstance(df[column].values[0], str):
                        # if is empty then change to 0
                        df[column] = df[column].apply(lambda x: 0 if x == '' else x)
                        try:
                            df[column] = df[column].astype(int)
                        except:
                            df[column] = df[column].astype(float)
                    df[column] = df[column].fillna(0)

                for _, row in df.iterrows():
                    all_scores = row[columns_to_rescore].values
                    all_scores = all_scores[all_scores != 0]
                    avg_score = np.mean(all_scores)
                    scores.append(avg_score)

                # create a dataframe with the scores
                df_scores = pd.DataFrame(scores, columns=['Score'])
                avg_score = np.mean(df_scores['Score'])

                # create a average for each row
                df = df.copy()
                df['Average Score'] = scores

                col1, col2 = st.columns(2)
                col1.metric(label = "Average Score", value = f'{round(avg_score*10, 1)}%')
                col2.metric("Total Reviews", len(df))

                # take an average of each column in the list
                averages = []
                for column in columns_to_rescore:
                    # take off the nan values
                    val = df[column].values
                    val = val[~np.isnan(val)]
                    # take off the zeros
                    val = val[val != 0]
                    averages.append(np.mean(val))

                columns_scores = st.columns(len(columns_to_rescore))
                for i,avg in enumerate(averages):
                    columns_scores[i].metric(label = columns_to_rescore[i],
                                             value = f'{round(avg*10, 1)}%')

                fig = px.histogram(
                    df_scores, x='Score', nbins=20,
                    title='Distribution of scores', opacity=0.5, text_auto=True)
                fig.update_layout(
                    xaxis_title_text='Score',
                    yaxis_title_text='Count',
                    bargap=0.2,
                    bargroupgap=0.1
                )


                c1,c2 = st.columns(2)
                c1.plotly_chart(fig, use_container_width=True)

                # group by day and get the average score
                df.loc[:,'date_for_filter'] = pd.to_datetime(df.loc[:,'date_for_filter'])
                df_day = df.groupby(['date_for_filter']).agg({'Average Score':'mean'}).reset_index()

                fig = go.Figure()

                fig.add_trace(go.Bar(
                    x=df_day['date_for_filter'],
                    y=df_day['Average Score'],
                    name='Average Score',
                    text=df_day['Average Score'], opacity=0.5))
                fig.update_layout(
                    xaxis_title_text='Date',
                    yaxis_title_text='Average Score',
                    bargap=0.2,
                    bargroupgap=0.1
                )
                # change color of bar if the value is less than the average
                fig.update_traces(marker_color='green')
                fig.add_trace(go.Scatter(
                    x=df_day['date_for_filter'],
                    y=[avg_score]*len(df_day),
                    name='Average Score', mode='lines',
                    marker_color='red', text = [avg_score]*len(df_day),
                    opacity=0.5)
                    )

                c2.plotly_chart(fig, use_container_width=True)
                st.write(df_empty)

def create_pie_chart(df):
    '''This graph is used inside the section: Overall'''
    # Get counts of each sentiment
    sentiment_counts = df[df['Details'] != '']['Sentiment'].value_counts()

    # Create a pie chart
    labels = sentiment_counts.index.tolist()
    values = sentiment_counts.values.tolist()

    # Transform to percent
    values = [round((v / sum(values)) * 100, 1) for v in values]

    # Create the figure
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_layout(showlegend=False)
    fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20, opacity=0.5,
                    marker={'colors': ['lightblue', 'green', 'red']})
    fig.update_traces(textposition='inside', textinfo='percent')

    # Add title
    fig.update_layout(title_text='Overall sentiment')

    st.plotly_chart(fig, use_container_width=True)

def create_graph_for_week_analysis(df):
    '''This graph is used inside the section: Week'''
    features_time = ['Week_Year', 'Sentiment']
    df = df[features_time].groupby(features_time).size().reset_index(name='Count')

    fig = go.Figure()

    colors = {'POSITIVE': 'green', 'NEGATIVE': 'red', 'neutral': 'lightblue'}
    for sentiment in df['Sentiment'].unique():
        sentiment_df = df[df['Sentiment'] == sentiment]
        fig.add_trace(go.Bar(
            x=sentiment_df['Week_Year'],
            y=sentiment_df['Count'],
            name=sentiment, opacity=0.5,
            marker_color=colors.get(sentiment, 'gray'))
            )

    fig.update_layout(xaxis_tickangle=-45)
    fig.update_layout(barmode='stack')
    fig.update_layout(title_text='Distribution by Week')
    st.plotly_chart(fig, use_container_width=True)

def create_graph_for_day_analysis(df):
    '''This graph is used inside the section: Day'''
    features_time = ['Day_Name', 'Sentiment']
    df = df[features_time].groupby(features_time).size().reset_index(name='Count')

    # same that above
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df[df['Sentiment'] == 'POSITIVE']['Day_Name'],
        y=df[df['Sentiment'] == 'POSITIVE']['Count'],
        name='POSITIVE', opacity=0.5, marker_color = 'green'))

    fig.add_trace(go.Bar(
        x=df[df['Sentiment'] == 'NEGATIVE']['Day_Name'],
        y=df[df['Sentiment'] == 'NEGATIVE']['Count'],
        name='NEGATIVE', opacity=0.5, marker_color = 'red'))

    fig.add_trace(go.Bar(
        x=df[df['Sentiment'] == 'neutral']['Day_Name'],
        y=df[df['Sentiment'] == 'neutral']['Count'],
        name='NEUTRAL', opacity=0.5, marker_color = 'lightblue'))

    fig.update_layout(xaxis_tickangle=-45)

    # stack the bars
    fig.update_layout(barmode='stack')
    # set title to "Distribution by day"
    fig.update_layout(title_text='Distribution by Day')

    # order the day of the week from monday to sunday
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fig.update_layout(xaxis={'categoryorder':'array', 'categoryarray':order})
    st.plotly_chart(fig, use_container_width=True)

def create_graph_for_hour_analysis(df):
    '''This graph is used inside the section: Hour'''
    features_time = ['Day_Part', 'Sentiment']
    df = df[features_time].groupby(features_time).size().reset_index(name='Count')

    # same that above
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df[df['Sentiment'] == 'POSITIVE']['Day_Part'],
        y=df[df['Sentiment'] == 'POSITIVE']['Count'],
        name='POSITIVE', opacity=0.5, marker_color = 'green'))

    fig.add_trace(go.Bar(
        x=df[df['Sentiment'] == 'NEGATIVE']['Day_Part'],
        y=df[df['Sentiment'] == 'NEGATIVE']['Count'],
        name='NEGATIVE', opacity=0.5, marker_color = 'red'))

    fig.add_trace(go.Bar(
        x=df[df['Sentiment'] == 'neutral']['Day_Part'],
        y=df[df['Sentiment'] == 'neutral']['Count'],
        name='NEUTRAL', opacity=0.5, marker_color = 'lightblue'))

    fig.update_layout(xaxis_tickangle=-45)

    # stack the bars
    fig.update_layout(barmode='stack')

    # order the day of the week from breakfast, lunch, dinner, night
    order = ['Breakfast', 'Lunch', 'Dinner', 'Late Night', 'Not Specified']
    fig.update_layout(xaxis={'categoryorder':'array', 'categoryarray':order})
    # set title to "Distribution by day"
    fig.update_layout(title_text='Distribution by Day Part')
    st.plotly_chart(fig, use_container_width=True)

def create_graph_for_month_analysis(df):
    '''This graph is used inside the section: Month'''
    features_time = ['Month_Year', 'Sentiment']
    df = df[features_time].groupby(features_time).size().reset_index(name='Count')

    # same that above
    fig = go.Figure()
    fig.update_layout(title_text='Distribution by Month')
    fig.add_trace(go.Bar(
        x=df[df['Sentiment'] == 'POSITIVE']['Month_Year'],
        y=df[df['Sentiment'] == 'POSITIVE']['Count'],
        name='POSITIVE', opacity=0.5, marker_color = 'green'))

    fig.add_trace(go.Bar(
        x=df[df['Sentiment'] == 'NEGATIVE']['Month_Year'],
        y=df[df['Sentiment'] == 'NEGATIVE']['Count'],
        name='NEGATIVE', opacity=0.5, marker_color = 'red'))

    fig.add_trace(go.Bar(
        x=df[df['Sentiment'] == 'neutral']['Month_Year'],
        y=df[df['Sentiment'] == 'neutral']['Count'],
        name='NEUTRAL', opacity=0.5, marker_color = 'lightblue'))

    fig.update_layout(xaxis_tickangle=-45)

    # stack the bars

    fig.update_layout(barmode='stack')

    st.plotly_chart(fig, use_container_width=True)

def create_graph_for_source_analysis(df):
    '''
    This graph is used inside the section: Source
    '''
    # count the source considering the sentiment as well
    features_time = ['Source', 'Sentiment']
    df = df[features_time].groupby(features_time).size().reset_index(name='Count')

    # same that above
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df[df['Sentiment'] == 'POSITIVE']['Source'],
        y=df[df['Sentiment'] == 'POSITIVE']['Count'],
        name='POSITIVE', opacity=0.5, marker_color = 'green'))

    fig.add_trace(go.Bar(
        x=df[df['Sentiment'] == 'NEGATIVE']['Source'],
        y=df[df['Sentiment'] == 'NEGATIVE']['Count'],
        name='NEGATIVE', opacity=0.5, marker_color = 'red'))

    fig.add_trace(go.Bar(
        x=df[df['Sentiment'] == 'neutral']['Source'],
        y=df[df['Sentiment'] == 'neutral']['Count'],
        name='NEUTRAL', opacity=0.5, marker_color = 'lightblue'))

    fig.update_layout(xaxis_tickangle=-45)

    # stack the bars
    fig.update_layout(barmode='stack')

    # set title to "Distribution by source"
    fig.update_layout(title_text='Distribution by Source')

    st.plotly_chart(fig, use_container_width=True)

def create_chart_totals_labels(data_frame):
    '''
    This function creates a chart with the total of labels
    '''
    # 7. create a new graph with all the labels and their counts
    labels = data_frame['Label_Dishoom'].tolist()
    #st.write('This is the labels: {}'.format(labels))
    labels = [clean_label(l) for l in labels]
    #st.write('This is the labels after clean: {}'.format(labels))
    labels = [item for sublist in labels for item in sublist]
    labels = [l for l in labels if l != '']
    labels = Counter(labels)
    labels = pd.DataFrame(labels.items(), columns=['Label_Dishoom', 'count'])

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels['Label_Dishoom'], y=labels['count'], opacity=0.8))
    fig.update_layout(xaxis={'categoryorder':'total descending'})
    # set color to index
    fig.update_traces(marker_color= '#01cc96')
    fig.update_xaxes(tickangle=45)
    fig.update_layout(title_text='Labels')
    fig.update_layout(showlegend=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_traces(text=labels['count'])
    st.plotly_chart(fig, use_container_width=True)
