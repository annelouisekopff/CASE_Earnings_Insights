import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from helpers.forge_helpers import *
import streamlit as st

creds_al = {"username": "anne-louise@vgimfs.com", "password" : "Vanguard.2023"}
cs = get_cursor(profile='Transcripts', creds=creds_al)

# App displayed in landscape format
st.set_page_config(
    page_title="Earnings Call Transcripts Overview",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

############################################################################################

### List of helper functions to switch from uri to company's name and vice versa

# Extract the name of a company from its URL
def get_name(uri):
    uri_split = uri.split('/')
    company_index = uri_split.index('company') + 1
    name_split = uri_split[company_index].split("_")
    name = ''
    for i in name_split:
        name += f'{i[0].upper()}{i[1:]} ' if i else ''
    return name



# From a company name get its URL
def get_company_uri(company):
    
    sql = """SELECT LOWER(e.label) AS label, a.uri, COUNT(*) AS cnt
        FROM Entities e
        JOIN EntityEquivalentTo a ON (a.docId = e.docId AND a.entityId = e.entityId)
        WHERE e.label ILIKE '{}%'
        AND a.uri LIKE 'http://forge.ai/company/%'
        GROUP BY 1, 2
        ORDER BY 3 DESC""".format(company)
        
    df = forge_request(sql, cs)
    uri = df['uri'].value_counts().idxmax()
    return uri

# Create a dictionary where each key is the name of the company and its value is the uri of this company 
def dict_uri_names(uri_list):
    result = {}
    for x in uri_list:
        name = get_name(x)
        result[name] = x
    return result

    
def get_list_uris(company_names, dict):
    uris = [dict.get(name) for name in company_names]
    return uris

""" Returns the list of the sector's uris"""
@st.cache_data
def get_list_sector_uris(sector):
    filtered_df = data_uris[data_uris['vgi_sector_3'] == sector]    
    result = filtered_df['uri'].tolist()
    return result

""" Returns the list of the uris of the companies in the same sector as our target company"""
def find_peers_uris(company):
    sector = data_uris[data_uris['uri'] == company]['vgi_sector_3'].iloc[0]  # Get the sector of the target company
    filtered_df = data_uris[data_uris['vgi_sector_3'] == sector]
    result = filtered_df['uri'].tolist()
    return result

############################################################################################

# Plots for company view

""" Returns the graph of the difference in topics importance between Q&A and MD sections """
@st.cache_data
def difference_topics(company_uri):
    
    # Retrieve the data using SQL and Snowflake
    sql_qa = """with co as (
            select distinct r.docid, e.uri
            from Relationships r
            left join EntityEquivalentTo e on e.docid = r.docid and e.entityid = r.objectentityid
            where model = 'FACTSQUARED'
            and predicatelabel = 'isCorporateRepresentative')

    select COUNT (dt.name) as topic_freq, dt.name, to_date(docdatetime) as date
    from documents d join co on d.docid = co.docid
    join DocumentScores ds on ds.docid = d.docid
    join documenttopics dt on dt.docid = d.docid
    join SourceTexts st on st.docid = d.docid
    where uri = '{}'
    and st.title = 'Q&A'
    group by date, dt.name
    order by date desc; """.format(company_uri)

    sql_md = """with co as (
            select distinct r.docid, e.uri
            from Relationships r
            left join EntityEquivalentTo e on e.docid = r.docid and e.entityid = r.objectentityid
            where model = 'FACTSQUARED'
                and predicatelabel = 'isCorporateRepresentative')

        select COUNT (dt.name) as topic_freq, dt.name, to_date(docdatetime) as date
        from documents d join co on d.docid = co.docid
        join DocumentScores ds on ds.docid = d.docid
        join documenttopics dt on dt.docid = d.docid
        join SourceTexts st on st.docid = d.docid
        where uri = '{}'
        and st.title = 'Management Discussion'
        group by date, dt.name
        order by date desc; """.format(company_uri)
    

    topics_qa = forge_request(sql_qa, cs)
    topics_md = forge_request(sql_md, cs)
    
    qa_mean_scores = topics_qa.groupby('name')['topic_freq'].mean()
    common_names = list(set(topics_qa['name']).intersection(set(topics_md['name'])))
    score_differences = qa_mean_scores.loc[common_names] - topics_md.groupby('name')['topic_freq'].mean().loc[common_names]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=score_differences[score_differences > 0].index,
            y=score_differences[score_differences > 0].values,
            name='Topics more important in Q&A',
            marker_color='red'
        )
    )
    fig.add_trace(
        go.Bar(
            x=score_differences[score_differences < 0].index,
            y=score_differences[score_differences < 0].values,
            name='Topics more important in MD',
            marker_color='blue'
        )
    )

    fig.update_layout(
        title="Differences of topics importance between Q&A session and management discussion section",
        xaxis_title="Name",
        yaxis_title="Score Difference"
    )
    
    return fig


"""Get the list of topics mentioned for one company and their confidence score"""
def get_events(company):
    
    sql = """with co as (
            select distinct r.docid, e.uri
            from Relationships r
            left join EntityEquivalentTo e on e.docid = r.docid and e.entityid = r.objectentityid
            where model = 'FACTSQUARED'
                and predicatelabel = 'isCorporateRepresentative')

        select d.title, to_date(docdatetime) as date, ie.label, ie.confidence
        from documents d join co on d.docid = co.docid
        join IntrinsicEvents ie on ie.docid = d.docid
        where uri = '{}'
        order by date desc; """.format(company)
    events = forge_request(sql, cs)
    avg_confidence = events.groupby('label')['confidence'].mean().reset_index()

    top_20_labels = avg_confidence.nlargest(30, 'confidence')
    fig = go.Figure(data=[go.Bar(x=top_20_labels['label'], y=top_20_labels['confidence'])])
    fig.update_layout(
        title='Histogram of Labels with Highest Confidence Scores',
        xaxis_title='Label',
        yaxis_title='Average Confidence',
        yaxis_range = [0.8,1],      
    )

    return fig.show()

    
""" For a specific topic and the chosen company, plots the importance of the topic over time"""    
def get_topic_over_time(company, topic):
    sql_1 = """ with co as (
          select distinct r.docid, e.uri
          from Relationships r
          left join EntityEquivalentTo e on e.docid = r.docid and e.entityid = r.objectentityid
          where model = 'FACTSQUARED'
            and predicatelabel = 'isCorporateRepresentative')

    select d.title, to_date(docdatetime) as date, dt.name, AVG(dt.score) as score
    from documents d join co on d.docid = co.docid
    join DocumentTopics dt on dt.docid = d.docid
    where uri = '{}'
    and docdatetime >= DATEADD(DAYS, -1600, CURRENT_TIMESTAMP)
    and dt.name = '{}'
    group by d.title, date, dt.name
    order by date desc""".format(company, topic)
    
    data_topics = forge_request(sql_1, cs)
    
    fig = go.Figure(data=[go.Bar(
    x=data_topics['title'],
    y=data_topics['score'],
    )])

    fig.update_layout(
        title='Topic importance across documents',
        xaxis_title='Document',
        yaxis_title='Importance',
        # yaxis_range=[0.1, 0.3]
    )
    return fig
         

data_uris = pd.read_csv('casee_company_uris_metadata.csv')


""" Plot the evolution of sentiment over time for the chosen peers of the company """
def sentiment_peers(peers_uris):
    
    fig = px.line(title='Sentiment Analysis - Peers Comparison') 
    colors = px.colors.qualitative.Plotly[:len(peers_uris)]
    
    for comp, color in zip(peers_uris, colors):
        name_peer = get_name(comp)
        sql_peer = """with co as (
        select distinct r.docid, e.uri
        from Relationships r
        left join EntityEquivalentTo e on e.docid = r.docid and e.entityid = r.objectentityid
        where model = 'FACTSQUARED'
            and predicatelabel = 'isCorporateRepresentative')

        select d.title, to_date(docdatetime) as date, ds.sentiment
        from documents d join co on d.docid = co.docid
        join DocumentSentiments ds on ds.docid = d.docid
        where uri = '{}'
        and docdatetime >= DATEADD(DAYS, -1700, CURRENT_TIMESTAMP)
        order by date asc; """.format(comp)
        
        df_peer = forge_request(sql_peer, cs)
        fig.add_scatter(x=df_peer['date'], y=df_peer['sentiment'], name=name_peer, line_color = color)
        
        fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Sentiment',
        legend_title='Company Peers',
    )
            
    return fig

#######################################################################################################

# Plots for the sector view

@st.cache_data  
def sector_sentiment_over_time(sector, from_date):
   
    sector_uris = get_list_sector_uris(sector)

    sql_sent_time = """with co as (
                select distinct r.docid, e.uri
                from Relationships r
                left join EntityEquivalentTo e on e.docid = r.docid and e.entityid = r.objectentityid
                where model = 'FACTSQUARED'
                    and predicatelabel = 'isCorporateRepresentative')



            SELECT TO_DATE(docdatetime) AS date, AVG(ds.sentiment) as sentiment
            FROM Documents d
            JOIN co ON d.docid = co.docid
            JOIN DocumentSentiments ds ON ds.docid = d.docid
            WHERE uri IN ({})
            AND date >= DATEADD(DAYS, -{}, CURRENT_TIMESTAMP)
            GROUP BY date
            ORDER BY date ASC""".format("'" + "', '".join(sector_uris) + "'", from_date)
     
    sql_all = """with co as (
                select distinct r.docid, e.uri
                from Relationships r
                left join EntityEquivalentTo e on e.docid = r.docid and e.entityid = r.objectentityid
                where model = 'FACTSQUARED'
                    and predicatelabel = 'isCorporateRepresentative')



            SELECT TO_DATE(docdatetime) AS date, AVG(ds.sentiment) as sentiment
            FROM Documents d
            JOIN co ON d.docid = co.docid
            JOIN DocumentSentiments ds ON ds.docid = d.docid
            WHERE date >= DATEADD(DAYS, -{}, CURRENT_TIMESTAMP)
            AND date <= DATEADD(DAYS, -2, CURRENT_TIMESTAMP)
            GROUP BY date
            ORDER BY date ASC""".format(from_date)     
            
    df_sent_time = forge_request(sql_sent_time, cs)
    df_all = forge_request(sql_all, cs)

    df_sent_time['date'] = pd.to_datetime(df_sent_time['date'])
    df_sent_time['year_month'] = df_sent_time['date'].dt.to_period('M').astype(str)
    df_avg_sentiment = df_sent_time.groupby('year_month')['sentiment'].mean().reset_index()
    
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all['year_month'] = df_all['date'].dt.to_period('M').astype(str)
    df_avg_all = df_all.groupby('year_month')['sentiment'].mean().reset_index()
    
    fig = go.Figure(data=[go.Bar(x=df_avg_sentiment['year_month'], y=df_avg_sentiment['sentiment'], name='Average sentiment for the sector')])
    # Set the title and axis labels
    
    fig.add_scatter(x=df_avg_all['year_month'], y=df_avg_all['sentiment'], name='Average sentiment for all companies')
    
    fig.update_layout(title='Average sentiment for the sector',
                xaxis_title='Month and Year',
                yaxis_title='Average Sentiment',
                yaxis_range = [0.15, 0.3])

    return fig



""" Returns a plot of the average sentiment for each company of the sector"""
@st.cache_data
def sector_sentiment(sector, selected_sector_uris):
   
    sector_uris = get_list_sector_uris(sector)

    sql_sector = """with co as (
            select distinct r.docid, e.uri
            from Relationships r
            left join EntityEquivalentTo e on e.docid = r.docid and e.entityid = r.objectentityid
            where model = 'FACTSQUARED'
                and predicatelabel = 'isCorporateRepresentative')



        SELECT d.title, TO_DATE(docdatetime) AS date, ds.sentiment, dt.name, dt.score, co.uri
        FROM Documents d
        JOIN co ON d.docid = co.docid
        JOIN DocumentSentiments ds ON ds.docid = d.docid
        JOIN DocumentTopics dt ON dt.docid = d.docid
        WHERE uri IN ({})
        ORDER BY date DESC; """.format("'" + "', '".join(sector_uris) + "'")
        
    sector_df = forge_request(sql_sector, cs)
    sentiment_sector = sector_df.groupby('uri')['sentiment'].mean().reset_index()
    sentiment_sector_top20 = sentiment_sector.sort_values('sentiment', ascending=True)
    mean_sentiment = sentiment_sector_top20['sentiment'].mean()
    std_sentiment = sentiment_sector_top20['sentiment'].std()
    
    sentiment_sector_top20 = sentiment_sector_top20[sentiment_sector_top20['uri'].isin(selected_sector_uris)]
    
    sentiment_sector_top20['company_name'] = sentiment_sector_top20['uri'].apply(lambda x: x.split('/')[-1].replace('_', ' ').title())
    
    # Rescale the sentiment 
    sentiment_sector_top20['rescaled_sentiment'] = (sentiment_sector_top20['sentiment'] - mean_sentiment)/std_sentiment

    colors = ['red' if sentiment < 0 else 'green' for sentiment in sentiment_sector_top20['rescaled_sentiment']]

    fig = go.Figure(data=[go.Bar(
        x=sentiment_sector_top20['company_name'],
        y=sentiment_sector_top20['rescaled_sentiment'],
        marker=dict(color=colors)
    )])

    fig.update_layout(
        title='Average sentiment by company for the chosen sector',
        xaxis_title='Company',
        yaxis_title='Normalized Sentiment',
        # yaxis_range=[-1, 1]
    )
    return fig

""" Returns a plot of the average readability score for each company of the sector"""
@st.cache_data       
def sector_readability(sector, selected_sector_uris):
    
    sector_uris = get_list_sector_uris(sector)
    # uri_list = "', '".join(sector_uris)  # Convert the list to a comma-separated string

    sql_evasiveness = """with co as (
            select distinct r.docid, e.uri
            from Relationships r
            left join EntityEquivalentTo e on e.docid = r.docid and e.entityid = r.objectentityid
            where model = 'FACTSQUARED'
                and predicatelabel = 'isCorporateRepresentative')



        SELECT d.title, TO_DATE(docdatetime) AS date, dt.scoretype, dt.score, co.uri
        FROM Documents d
        JOIN co ON d.docid = co.docid
        JOIN DocumentScores dt on dt.docid = d.docid
        WHERE uri IN ({})
        AND date >= DATEADD(DAYS, -700, CURRENT_TIMESTAMP)
        AND dt.scoretype = 'readability'
        ORDER BY date DESC; """.format("'" + "', '".join(sector_uris) + "'")
        
    sector_evasiveness = forge_request(sql_evasiveness, cs)
    
    
    sector_evasiveness['score'] = pd.to_numeric(sector_evasiveness['score'], errors='coerce')
    sector_evasiveness = sector_evasiveness.dropna(subset=['score'])
    sector_evasiveness = sector_evasiveness.groupby('uri')['score'].mean().reset_index()
    sector_evasiveness_top = sector_evasiveness.sort_values('score', ascending=True)
    
    mean_score = sector_evasiveness_top['score'].mean()
    std_score = sector_evasiveness_top['score'].std()
    sector_evasiveness_top = sector_evasiveness_top[sector_evasiveness_top['uri'].isin(selected_sector_uris)]
    sector_evasiveness_top['company_name'] = sector_evasiveness_top['uri'].apply(lambda x: x.split('/')[-1].replace('_', ' ').title())


    # Rescale the evasiveness scores relative to the mean
    sector_evasiveness_top['rescaled_score'] = (sector_evasiveness_top['score'] - mean_score)/std_score

    colors = ['red' if score < 0 else 'green' for score in sector_evasiveness_top['rescaled_score']]

    fig = go.Figure(data=[go.Bar(
        x=sector_evasiveness_top['company_name'],
        y=sector_evasiveness_top['rescaled_score'],
        marker=dict(color=colors)
    )])

    fig.update_layout(
        title='Evasiveness score of companies from the chosen sector',
        xaxis_title='Company',
        yaxis_title='Normalized Evasiveness',
        # yaxis_range=[mean_score - 50, mean_score + 50]
    )
    return fig



# Get average sentiment by topic
# def get_topic_sentiment(topic):
    
#     sql_topic = """ SELECT dt.startsentenceid, AVG(ss.score) as sentencescore, d.docdatetime
#         FROM Documents d
#         JOIN DocumentTopics dt on dt.docid = d.docid
#         JOIN SentenceScores ss on ss.sentenceid = dt.startsentenceid
#         WHERE ss.scoretype = 'sentence-sentiment'
#         AND dt.name = '{}'
#         GROUP BY d.docdatetime, (dt.startsentenceid)    
#         ORDER BY d.docdatetime ASC """.format(topic)
        
#     df = forge_request(sql_topic, cs)
#     return df


"""Evolution of the average sentiment relative to a sector over time"""    
