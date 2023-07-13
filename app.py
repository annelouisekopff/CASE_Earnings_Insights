
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from app_helpers import *
from app_helpers2 import *


# Define CSS styles for better aesthetics
st.markdown(
    """
    <style>
    .st-eb {
        padding: 15px;
        border: 1px solid lightgray;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .st-f3 {
        margin-top: 30px;
    }
    .st-f6 {
        margin-top: 60px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Earnings Call Transcripts Overview')

st.sidebar.title("View Options")
view_option = st.sidebar.radio("Select View", ("Company View", "Sector View"))

# left_column, right_column = st.columns(2)
# Company view
if view_option == "Company View":
    st.subheader('Company View')
    company_name = st.text_input("Company name", key="comp_name")
    company_uri = get_company_uri(company_name)
    peers_uris = find_peers_uris(company_uri)
    comp_dict = dict_uri_names(peers_uris)
    peers_names = comp_dict.keys()
    
    st.plotly_chart(difference_topics(company_uri), use_container_width=True)
    
    topic = st.text_input("Topic", key="topic")
    topic.title()
    st.plotly_chart(get_topic_over_time(company_uri, topic))
    
    companies = st.multiselect('Peers', peers_names)
    selected_companies_uris = get_list_uris(companies, comp_dict)
    st.plotly_chart(sentiment_peers(selected_companies_uris))
    
# Sector view
if view_option == "Sector View":
    st.subheader('Sector View')
    
    col1, col2 = st.columns(2)
    
    with col1:
        sector = st.text_input("Sector", key="sector_sent")
        from_date = st.number_input("Number of days", key="date")
        
        sector_uris = get_list_sector_uris(sector)
        names_dict = dict_uri_names(sector_uris)
        sector_companies = names_dict.keys()

        
        # Plot sentiment evolution for the whole sector
        st.plotly_chart(sector_sentiment_over_time(sector, from_date), use_container_width=True)
        st.plotly_chart(topic_change(sector, from_date))
    
        topic = st.text_input("Topic", key = "topic_")
        st.plotly_chart(company_mentions_topic(topic, sector, from_date))
        
        footnote_text_1 = """1. Topic frequency over time for the sector and for the top 3 companies that mention this topic the most"""
        footnote_html_1 = f'<sub><sup>{footnote_text_1}</sup></sub>'
        
        st.markdown(footnote_html_1, unsafe_allow_html=True)
    
    with col2:
        # Plot evasiveness and sentiment for specific companies
        selected_comp = st.multiselect("Select companies", sector_companies)
        selected_uris = get_list_uris(selected_comp, names_dict)
        
        st.plotly_chart(sector_readability(sector, selected_uris), use_container_width=True)
        st.plotly_chart(sector_sentiment(sector, selected_uris), use_container_width=True)
        
        footnote_text = """2. Normalized score = (score - mean score of the sector) / (std_deviation of the sector).\n The y-axis represents the number of standard deviation away from the mean of the sector."""
        footnote_html = f'<sub><sup>{footnote_text}</sup></sub>'
        
        st.markdown(footnote_html, unsafe_allow_html=True)
    
    
    
    
    

