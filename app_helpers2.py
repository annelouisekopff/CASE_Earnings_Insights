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

# # App displayed in landscape format
# st.set_page_config(
#     page_title="Earnings Call Transcripts Overview",
#     page_icon="🧊",
#     layout="wide",
#     initial_sidebar_state="expanded",
#     menu_items={
#         'Get Help': 'https://www.extremelycoolapp.com/help',
#         'Report a bug': "https://www.extremelycoolapp.com/bug",
#         'About': "# This is a header. This is an *extremely* cool app!"
#     }
# )

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

data_uris = pd.read_csv('casee_company_uris_metadata.csv')

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

""" Plots the topics importance since a specified date for the chosen sector """
def trending_topics(sector, days_before):
    sector_uris = get_list_sector_uris(sector)
    sql_req = """with co as (
                select distinct r.docid, e.uri
                from Relationships r
                left join EntityEquivalentTo e on e.docid = r.docid and e.entityid = r.objectentityid
                where model = 'FACTSQUARED'
                    and predicatelabel = 'isCorporateRepresentative')



                SELECT TO_DATE(docdatetime) AS date, COUNT (dt.name) as topic_frequency, dt.name
                FROM Documents d
                JOIN co ON d.docid = co.docid
                JOIN DocumentTopics dt ON dt.docid = d.docid
                WHERE uri IN ({})
                AND date >= DATEADD(DAYS, -{}, CURRENT_TIMESTAMP)
                GROUP BY dt.name, date
                ORDER BY date ASC""".format("'" + "', '".join(sector_uris) + "'", days_before)
                
    data_topics = forge_request(sql_req, cs)
    cache = data_topics
    
    data_topics['date'] = pd.to_datetime(data_topics['date'])
    data_topics['year_month'] = data_topics['date'].dt.to_period('M').astype(str)
    df_avg_frequency = data_topics.groupby('name')['topic_frequency'].mean().reset_index()
    
    fig = go.Figure(data=[go.Bar(x=df_avg_frequency['name'], y=df_avg_frequency['topic_frequency'])])
    

    # Set the title and axis labels
    fig.update_layout(title='Trending topics in the last {} days'.format(days_before),
                    xaxis_title='Topic names',
                    yaxis_title='Frequency across the {} sector'.format(sector))

    return fig

# trending_topics('Technology', 360)

""" Plots the difference of topics importance between two specified dates"""
def topic_change(sector, date_1):
    sector_uris = get_list_sector_uris(sector)
    sql_to_date = """with co as (
                select distinct r.docid, e.uri
                from Relationships r
                left join EntityEquivalentTo e on e.docid = r.docid and e.entityid = r.objectentityid
                where model = 'FACTSQUARED'
                    and predicatelabel = 'isCorporateRepresentative')



                SELECT TO_DATE(docdatetime) AS date, COUNT (dt.name) as topic_frequency, dt.name
                FROM Documents d
                JOIN co ON d.docid = co.docid
                JOIN DocumentTopics dt ON dt.docid = d.docid
                WHERE uri IN ({})
                AND date >= DATEADD(DAYS, -100, CURRENT_TIMESTAMP)
                GROUP BY dt.name, date
                ORDER BY date ASC""".format("'" + "', '".join(sector_uris) + "'")
                
    sql_date_1 = """with co as (
                select distinct r.docid, e.uri
                from Relationships r
                left join EntityEquivalentTo e on e.docid = r.docid and e.entityid = r.objectentityid
                where model = 'FACTSQUARED'
                    and predicatelabel = 'isCorporateRepresentative')



                SELECT TO_DATE(docdatetime) AS date, COUNT (dt.name) as topic_frequency, dt.name
                FROM Documents d
                JOIN co ON d.docid = co.docid
                JOIN DocumentTopics dt ON dt.docid = d.docid
                WHERE uri IN ({})
                AND date >= DATEADD(DAYS, -{}, CURRENT_TIMESTAMP)
                GROUP BY dt.name, date
                ORDER BY date ASC""".format("'" + "', '".join(sector_uris) + "'", date_1)
                
    df_to_date = forge_request(sql_to_date, cs)
    df_date_1 = forge_request(sql_date_1, cs)
    
    df_to_date['date'] = pd.to_datetime(df_to_date['date'])
    df_to_date['year_month'] = df_to_date['date'].dt.to_period('M').astype(str)
    df_avg_to_date = df_to_date.groupby('name')['topic_frequency'].mean().reset_index()
    
    df_date_1['date'] = pd.to_datetime(df_date_1['date'])
    df_date_1['year_month'] = df_date_1['date'].dt.to_period('M').astype(str)
    df_avg_date_1 = df_date_1.groupby('name')['topic_frequency'].mean().reset_index()


    # Merge the two dataframes on the 'name' column
    merged_df = df_avg_to_date.merge(df_avg_date_1, on='name', how='inner')

    colors = ['blue', 'green']
    fig = go.Figure()

    # Add bars for relative frequency in df_to_date and df_date_1
    for i, col_name in enumerate(['topic_frequency_x', 'topic_frequency_y']):
        names = ['last quarter', 'last {} days'.format(date_1)]
        fig.add_trace(go.Bar(
            x=merged_df['name'],
            y=merged_df[col_name],
            name=names[i],
            marker_color=colors[i]
        ))

    fig.update_layout(
        title='Topic frequency for the {} sector in the last quarter vs. in the last {} days'.format(sector, date_1),
        xaxis_title='Topics',
        yaxis_title='Topic frequency'
    )

    return fig

# topic_change('Technology', 1500)

def company_mentions_topic(topic, sector, date_1):
    sector_uris = get_list_sector_uris(sector)
    sql_freq_sector = """with co as (
            select distinct r.docid, e.uri
            from Relationships r
            left join EntityEquivalentTo e on e.docid = r.docid and e.entityid = r.objectentityid
            where model = 'FACTSQUARED'
                and predicatelabel = 'isCorporateRepresentative')



            SELECT TO_DATE(docdatetime) AS date, COUNT (dt.name) as topic_frequency, co.uri
            FROM Documents d
            JOIN co ON d.docid = co.docid
            JOIN DocumentTopics dt ON dt.docid = d.docid
            WHERE uri IN ({})
            AND dt.name = '{}'
            AND date >= DATEADD(DAYS, -{}, CURRENT_TIMESTAMP)
            GROUP BY uri, date
            ORDER BY date ASC""".format("'" + "', '".join(sector_uris) + "'", topic, date_1)
    
    topic_freq_sector = forge_request(sql_freq_sector, cs)
    
    topic_freq_sector['date'] = pd.to_datetime(topic_freq_sector['date'])
    topic_freq_sector['year_month'] = topic_freq_sector['date'].dt.to_period('M').astype(str)
    topic_freq_avg = topic_freq_sector.groupby('year_month')['topic_frequency'].mean().reset_index()
    
    top_comp = topic_freq_sector.groupby('uri')['topic_frequency'].mean().reset_index()
    top_comp = top_comp.sort_values('topic_frequency', ascending=False).head(3)
    top_uris = top_comp['uri'].tolist()[:3]
    

    
    fig = go.Figure(data=[go.Bar(x=topic_freq_avg['year_month'], y=topic_freq_avg['topic_frequency'], name='Frequency of topic for the sector')])
    # Set the title and axis labels
    
    for uri in top_uris:
        nameComp = get_name(uri)
        filtered_topic_freq = topic_freq_sector[topic_freq_sector['uri']==uri]
        topic_freq_avg_top = filtered_topic_freq.groupby('year_month')['topic_frequency'].mean().reset_index()
        fig.add_scatter(x=topic_freq_avg_top['year_month'], y=topic_freq_avg_top['topic_frequency'], name='topic frequency for {}'.format(nameComp))
    
    fig.update_layout(title='Frequency of the topic {} over time for the sector {}'.format(topic, sector),
                xaxis_title='Month and Year',
                yaxis_title='Average number of topic mentions',
                # yaxis_range = [0.15, 0.3])
    )

    return fig
    
    