import snowflake.connector
import pandas as pd
import numpy as np
from datetime import date
import re
#from helpers.document_annotation import DocAnnotation
from helpers import get_user
#from fastapi import HTTPException
#from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
#import os
#import boto3



def get_cursor(profile='Articles', role=None, database=None, schema=None, creds=None):
    if profile == 'Transcripts':
        role = 'IMFS'
        database = "FORGEAI_FACTSQUARED_V0670"
        schema = 'FACTSQUARED_V0670'
    elif profile == 'Articles':
        role = 'IMFS'
        database = "FORGEAI_ARTICLES_V0660"
        schema = 'ARTICLES_V0660'
    # Verify connection

    # if role is not None:
    #     role = role
    # if database is not None:
    #     database = database
    # if schema is not None:
    #     schema = schema

    ctx = snowflake.connector.connect(
        user=creds["username"],
        password=creds["password"],
        account='rk09149.us-east-1',
        role=role,
        warehouse='IMFS_WH',
        database=database,
        schema=schema
    )
    cs = ctx.cursor()
    cs.execute("SELECT current_version()")
    one_row = cs.fetchone()
    print('Connected to Snowflake version {}'.format(one_row[0]))

    return cs


def forge_request(sql, cursor):
    data = cursor.execute(sql).fetchall()

    col_names = []
    for elt in cursor.description:
        col_names.append(elt[0].lower())
    df = pd.DataFrame(data, columns=col_names)

    return pd.DataFrame(df)

# Functions for Document View

def get_company_name(row, uri: str = 'uri'):
    uri_split = row[uri].split('/')
    company_index = uri_split.index('company') + 1
    name_split = uri_split[company_index].split("_")
    name = ''
    for i in name_split:
        name += f'{i[0].upper()}{i[1:]} ' if i else ''
    return name


def get_available_earnings(company_uri, cursor):
    """Dates with available earnings"""

    query = """with co as (
          select distinct r.docid, e.uri
          from Relationships r
          left join EntityEquivalentTo e on e.docid = r.docid and e.entityid = r.objectentityid
          where model = 'FACTSQUARED'
            and predicatelabel = 'isCorporateRepresentative')

    select d.docid, to_date(docdatetime) as date
    from documents d join co on d.docid = co.docid
    where uri = '{}'
    order by date desc; """.format(company_uri)

    df_ = forge_request(query, cursor)
    return df_


def get_doc_view_data(earning_call_id, user_creds, cursor):
    """return dataframe with sentence by sentence sentiment/topic breakdown
    for earnings transcript with docid: @earning_call_id"""
    print(earning_call_id)
    # define importand and unimportant keywords for highlighting
    fixed_income_keywords = ['balance sheet', 'cash', 'rating', 'interest', 'debt', 'distribution', 'dividends', 'buyback', 'capital', 'covenant',
                             'liquidity', 'equity', 'leverage', 'funding', 'spending', 'investment grade', 'resilience', 'ambitious', 'transform', 'diversify', 'gearing']
    fixed_income_ignores = ['good day', 'operator instructions', 'Safe Harbor provisions', 'forward-looking statements', 'potential risks and uncertainties',
                            'ability to protect patents', 'good afternoon', 'good morning', 'redistribution, retransmission or rebroadcast', 'call instructions']
    include_pattern = '|'.join(fixed_income_keywords)
    exclude_pattern = '|'.join(fixed_income_ignores)

    # form query, merging salience scores and VADER sentiment scores
    q = """select distinct s.sentenceId, s.sectionid, s.text, sc.score as vader, sub.score as salience, t.name as topic
            from sentences s
            join sentencescores sc on (s.docid = sc.docid and s.sentenceId = sc.sentenceId and sc.modelName = 'vader' and sc.scoreType = 'sentence-sentiment')
            left join documenttopics t on s.docid = t.docid and s.sentenceId between t.startSentenceID and t.endSentenceId
            left join (
                select k.sentenceid, k.sectionid, k.score from 
                sentencescores k 
                where k.modelname = 'factset-salience' and k.scoretype = 'document_salience_score' and k.docid = {0}
            ) sub on sub.sectionid = s.sectionid and sub.sentenceid = s.sentenceid
            where s.docid = {1}
            order by s.sectionid, s.sentenceid;""".format(earning_call_id, earning_call_id)
    sentences = forge_request(q, cursor)
    for index, row in sentences.iterrows():
        decision = classifier(row['text'])[0]['label']
        prediction = 'negative' if decision == 'positive' else ('neutral' if decision == 'negative' else 'positive')
        sentences.at[index, 'vader'] = prediction
    sentences['included'] = sentences['vader'] != 'neutral'
    
    # Enrich data with sentence reviews from DocumentDB
    docObj = DocAnnotation()
    #user_name, user_email, doc_id
    sentence_reviews = docObj.get_sentence_review({
        'user_name': user_creds['name'],
        'user_email': user_creds['email'],
        'doc_id': earning_call_id
    })
    sent_review_df = pd.DataFrame.from_dict(sentence_reviews)
    
    if not sent_review_df.empty:
      enriched_sentences = sentences.merge(sent_review_df, how="left", left_on=["sectionid","sentenceid"], right_on=["section_id","sentence_id"])
      enriched_sentences = enriched_sentences.drop(["_id", "section_id", "sentence_id"], axis=1)
      enriched_sentences.is_correct.replace({np.nan: None}, inplace=True)
      enriched_sentences.correction.replace({np.nan: None}, inplace=True)

      return enriched_sentences
    else:
      sentences['is_correct'] = None
      sentences['correction'] = None

    return sentences



# Functions for Company Theme View


def get_company_view_data(company_URI, cursor):
    # hacky method - to change code to pull from URI directly
    company_URI_path = company_URI[len(
        'http://forge.ai/company/'):].split('/')[0]
    # print(company_URI_path)
    topic_change = """with present as (
        SELECT t.name as name, sum(sentiment) as sentiment
        FROM EntitySubsidiaryOf e
        JOIN Documents d ON (d.docId = e.docId)
        join documenttopics t on (e.docid = t.docid)
        join documentsentiments s on (s.docid = e.docid)
        WHERE e.uriPath = '{}'
            AND d.collectDatetime >= DATEADD(DAYS, -90, CURRENT_TIMESTAMP)
        group by 1)
        , past as(
        SELECT t.name as name, avg(sentiment) as sentiment
        FROM EntitySubsidiaryOf e
        JOIN Documents d ON (d.docId = e.docId)
        join documenttopics t on (e.docid = t.docid)
        join documentsentiments s on (s.docid = e.docid)
        WHERE e.uriPath = '{}'
            AND d.collectDatetime <= DATEADD(DAYS, -90, CURRENT_TIMESTAMP)
            AND d.collectDatetime >= DATEADD(DAYS, -180, CURRENT_TIMESTAMP)
        group by 1)
        select past.name as past, present.name as present, past.sentiment as past_sentiment, present.sentiment as present_sentiment, (NVL(present_sentiment,0) - NVL(past_sentiment,0)) / abs(NVL(past_sentiment,.1)) as perc_sentiment
        from past
        Full join present on (past.name = present.name)
        where past.name is not null or present.name is not null
        order by perc_sentiment;"""

    df_ = forge_request(topic_change.format(
        company_URI_path, company_URI_path), cursor)
    return df_

# Functions for Keywords


def query_keyword1(keyword, cursor):
    query = """select NVL(to_date(docdatetime), to_date(collectdatetime)) as date, count(*) as mentions 
    from entities e
    join documents d
    on d.docid = e.docid
    where label like '{}'
    and date >= ('2016-01-01')
    group by date""".format(keyword)

    df_query = forge_request(query, cursor)

    # process data
    df = df_query.copy()
    df.date = pd.to_datetime(df.date)
    # remove erronous post-dated data
    today = date.today()
    today = today.strftime("%Y-%m-%d")
    df = df[df.date < today]
    df = df.sort_values(by='date')
    df = df.set_index('date')
    return df


def query_keyword(keyword: str, cursor):
    """Args:
        keyword: #TODO protect against potential injection of invalid argument.
        cursor: Should call to forge TRANSCRIPTS database.
    Currently, data truncated at the monthly view.
    Querying data from the transcripts requires a degree of normalization
    (unlike news articles, which can be assumed to continuously be released over the month/weeks)
    Here, mentions aggregated by the month and normalized by the # of transcripts over the month"""

    cutoff_start = '2016-01-01'

    query = f"""select count(text) as n_mentions,
                count(distinct d.docid) as n_doc,
                LEFT(DOCDATETIME, 7) as date
                from sentences s join documents d on d.docid = s.docid 
                where text ilike '%{keyword}%'
                and date >= '{cutoff_start}'
                group by date
                order by date desc;
                """

    df_query = forge_request(query, cursor)

    # process data
    df = df_query.copy()
    df.date = pd.to_datetime(df.date)
    # remove erronous post-dated data
    today = date.today()
    today = today.strftime("%Y-%m-%d")
    df = df[df.date < today]

    # normalize monthly mentions by monthly transcripts release
    df['mentions'] = df['n_mentions'] / df['n_doc']
    df = df.drop(['n_mentions', 'n_doc'], axis=1)

    df = df.sort_values(by='date')
    df = df.set_index('date')
    return df

# Functions for Heat map


def get_bbg_info():
    # download the ratings info
    processed_ratings = pd.read_csv('./app/files/casee_company_uris_metadata.csv')

    return processed_ratings

# TODO Clean up once we confirm other functions working as expected


def get_heatmap_data(cursor):
    q_a_spr_query = """with russell_and_SP as
       ( select distinct uri, cast(indices.value as varchar) as index_member, split_part(cast(tickers.value as varchar), ':',1) as country, split_part(cast(tickers.value as varchar), ':',2) as exchange,split_part(cast(tickers.value as varchar), ':',3) as ticker
       from "FORGEAI_KG_V0660"."KG_V0660"."COMPANIES" c, lateral flatten(input=>indices) indices, lateral flatten(input=>tickers) tickers
       where index_member in  ('SPX', 'RUA')
       and country = 'US'
       and Exchange in ('NASDAQ', 'NYSE', 'AMEX')
       ),
       co as (
              select distinct r.docid, e.uri, russell_and_SP.index_member, russell_and_SP.ticker
              from Relationships r
              left join EntityEquivalentTo e on e.docid = r.docid and e.entityid = r.objectentityid
              right join russell_and_SP on e.uri = russell_and_SP.uri
              where model = 'FACTSQUARED'
              and predicatelabel = 'isCorporateRepresentative'
              ), vader as (
              select co.docid, p.sectionid,
                     avg(vader.score::float) as score
              from co
              join DialogueProperties p on p.docid = co.docid
              left join SentenceScores vader on vader.docid = p.docid and vader.sectionid = p.sectionid
                                          and vader.modelName = 'vader'
                                          and vader.scoreType = 'sentence-sentiment'
              group by 1,2
              )
       select co.uri, co.ticker, co.index_member,
                     any_value(d.title) as "call title",
                     any_value(to_date(d.docdatetime)) as "date",
                     iff(p.type in ('Q', 'A'), p.type, 'MD') as "category",
                     avg(vader.score) as vader
                     
       from co
       join Documents d on d.docid = co.docid
       join DialogueProperties p on p.docid = co.docid
       join SourceTexts t on t.docid = p.docid and t.sectionid = p.sectionid
       left join Vader on vader.docid = p.docid and vader.sectionid = p.sectionid
       where (   p.type in ('Q', 'A')
              or t.title in ('Q&A', 'Management Discussion'))
       group by d.docid, co.uri, "category", co.ticker, co.index_member
       order by co.uri, "date" desc, "category";"""

    # Uses transcripts
    q_a_spr_request = forge_request(q_a_spr_query, cursor)
    return q_a_spr_request


def process_heatmap_data(heatmap_data):
    q_a_spr_df = heatmap_data.copy(deep=True)

    q_a_spr_df['quarter'] = pd.to_datetime(
        q_a_spr_df.date).dt.to_period('Q')

    q_a_spr_df = q_a_spr_df.drop(
        ['ticker', 'index_member'], axis=1).drop_duplicates()

    processed_ratings = get_bbg_info()
    # processed_ratings
    q_a_spr_df = pd.merge(q_a_spr_df, processed_ratings,
                          how='left', on='uri').dropna(subset=['IG_HY'])

    q_a_spr_df = q_a_spr_df[['uri', 'call title', 'date', 'category', 'vader', 'region', 'IG_HY', 'vgi_sector_1',
                             'vgi_sector_2', 'vgi_sector_3', 'ticker', 'quarter']]

    return q_a_spr_df

# new queries to call heatmap data according to different user selections and behaviors

# helper functions for get_heatmap_data_separated


def normalize_pct_change_quarter(heatmap_df):
    """ A function to normalize the heatmap data before display.
    After discussion with credit analysts, decision that:
    - A relative measure of sentiment change is more informative than absolute sentiment
    - Due to annual patterns of earnings, the change YoY might make ssense
    Hence the following fuction:
    Quarter Q_i percentage change wrt the previous year's Q_i"""
    hm_df_normalized = heatmap_df.copy()
    hm_df_normalized = 100 * hm_df_normalized.pct_change(periods=4)

    return hm_df_normalized


def get_heatmap_sector_row(list_uri, section, startYear, endYear, cursor):
    """Helper function for forge_get_heatmap_sector_aggregated 
    -> returns rows for list of uris and specified transcript sections"""

    # TODO: figure out why offset needs to be -2 for the right quarters to show off.
    yearLowerBound = '{}-12-31'.format(int(startYear)-2)
    yearUpperBound= '{}-01-01'.format(int(endYear)+1)
    print(yearLowerBound, yearUpperBound)

    query_heatmap_row = '''with co as (
                     select distinct r.docid, e.uri
                     from Relationships r
                     left join EntityEquivalentTo e on e.docid = r.docid and e.entityid = r.objectentityid
                     where model = 'FACTSQUARED'
                            and predicatelabel = 'isCorporateRepresentative'
                     ),
              vader as (
                     select co.docid, p.sectionid,
                            avg(vader.score::float) as score
                     from co
                     join DialogueProperties p on p.docid = co.docid
                     left join SentenceScores vader on vader.docid = p.docid and vader.sectionid = p.sectionid
                                                 and vader.modelName = 'vader'
                                                 and vader.scoreType = 'sentence-sentiment'
                     group by 1,2
                            ), 
              sector_rows as (
                     select co.uri, any_value(d.title) as "call title",
                                   any_value(to_date(d.docdatetime)) as "date",
                                   DATE_TRUNC('QUARTER',"date") AS quarter,
                                   iff(p.type in ('Q', 'A'), p.type, 'MD') as "category",
                                   avg(vader.score) as vader
                     from co
                     join Documents d on d.docid = co.docid
                     join DialogueProperties p on p.docid = co.docid
                     join SourceTexts t on t.docid = p.docid and t.sectionid = p.sectionid
                     left join Vader on vader.docid = p.docid and vader.sectionid = p.sectionid
                     where co.uri in ({}) 
                     and (   p.type in ('Q', 'A')
                            or t.title in ('Q&A', 'Management Discussion'))
                     group by d.docid, co.uri, "category"
                     order by co.uri, "date" desc, "category"
                     )
              select quarter, avg(vader) as vader
              from sector_rows
              where "category" = '{}' 
              and quarter > '{}'
              and quarter < '{}'
              group by quarter'''

    # Uses transcripts
    formatted_uris = [x.replace("'", "''") for x in list_uri]
    print(formatted_uris)
    formatted_query = query_heatmap_row.format(
        ", ".join(["'"+x+"'" for x in formatted_uris]), section, yearLowerBound, yearUpperBound)

    request_df = forge_request(formatted_query, cursor)
    return request_df

def get_heatmap_sector_aggregated(sector_level, section, startYear, endYear, region, IG_HY, cursor, verbose=False):
    """If sector_level is specified with no sector_value,
    we want to return the heatmap at with sector_level aggregated sentiment
    hence a heatmap where each row (column for the df) is linked to one sector value of the sector_level"""
    mapping = get_bbg_info()

    sectors = mapping[sector_level].dropna().unique()

    heatmap_rows = []
    # for each sector in the sector level:
    # TODO potentially get rid of loop
    for s in sectors:
        # get the sector uris
        if verbose:
            print("fetching {} = {}".format(sector_level, s))
        s_uris = mapping[(mapping[sector_level] == s) &
                         (mapping['IG_HY'] == IG_HY) & 
                         (mapping['region'] == region)]
        s_uris = s_uris.uri.to_list()

        if not s_uris:
            continue
        # get the aggregated query
        heatmap_row = get_heatmap_sector_row(
            s_uris, section=section, startYear=startYear, endYear=endYear, cursor=cursor)

        # format for sector row
        heatmap_row.quarter = pd.to_datetime(
            heatmap_row.quarter).dt.to_period('Q').astype("str")

        # heatmap_row = heatmap_row.set_index('quarter')
        # heatmap_row = heatmap_row.rename(columns={'vader': s})
        heatmap_row["sector"] = s
        heatmap_row = heatmap_row.sort_values("quarter")

        # TODO refactor so we do the pivot and melt in function
        heatmap_row = heatmap_row.pivot_table(
            index='quarter', values='vader', columns='sector')

        heatmap_row = normalize_pct_change_quarter(heatmap_row)

        heatmap_row = pd.melt(heatmap_row.reset_index(),
                              id_vars='quarter',
                              var_name='sector',
                              value_name='vader')

        heatmap_row = heatmap_row.dropna(subset='vader')

        heatmap_row = heatmap_row.to_dict(orient="records")

        if verbose:
            print("{} data fetched.".format(s))
        # set aside for concatenation

        heatmap_rows = heatmap_rows + heatmap_row

    # concatenate data
    # heatmap_data = pd.concat(heatmap_rows, axis=1)
    # heatmap_data = heatmap_data.sort_index()
    return heatmap_rows


def get_heatmap_sector_uris(sector_level, sector_value, section, startYear, endYear, region, IG_HY, cursor, verbose=False):
    """If both sector_level and sector_value are specificied,
    we want to return the heatmap at the company-level granularity
    hence a heatmap where each row (column for the df) is linked to an URI"""

    mapping = get_bbg_info()
    uri_ticker_df = mapping[mapping[sector_level] == sector_value]
    uri_ticker_df = uri_ticker_df[uri_ticker_df['region'] == region]
    uri_ticker_df = uri_ticker_df[uri_ticker_df['IG_HY'] == IG_HY]
    uri_ticker_df = uri_ticker_df[["uri", "ticker"]]
    list_uris = uri_ticker_df.uri.to_list()


    if not list_uris:
        raise HTTPException(
            status_code=404,
            detail="No data for selected parameters",
            headers={"X-Error": "There goes my error"},
        )


    # TODO: figure out why offset needs to be -2 for the right quarters to show off.
    yearLowerBound = '{}-12-31'.format(int(startYear)-2)
    yearUpperBound= '{}-01-01'.format(int(endYear)+1)

    query_heatmap_row = '''with co as (
                     select distinct r.docid, e.uri
                     from Relationships r
                     left join EntityEquivalentTo e on e.docid = r.docid and e.entityid = r.objectentityid
                     where model = 'FACTSQUARED'
                            and predicatelabel = 'isCorporateRepresentative'
                     ),
              vader as (
                     select co.docid, p.sectionid,
                            avg(vader.score::float) as score
                     from co
                     join DialogueProperties p on p.docid = co.docid
                     left join SentenceScores vader on vader.docid = p.docid and vader.sectionid = p.sectionid
                                                 and vader.modelName = 'vader'
                                                 and vader.scoreType = 'sentence-sentiment'
                     group by 1,2
                            ), 
              sector_rows as (
                     select co.uri, any_value(d.title) as "call title",
                                   any_value(to_date(d.docdatetime)) as "date",
                                   DATE_TRUNC('QUARTER',"date") AS quarter,
                                   iff(p.type in ('Q', 'A'), p.type, 'MD') as "category",
                                   avg(vader.score) as vader
                     from co
                     join Documents d on d.docid = co.docid
                     join DialogueProperties p on p.docid = co.docid
                     join SourceTexts t on t.docid = p.docid and t.sectionid = p.sectionid
                     left join Vader on vader.docid = p.docid and vader.sectionid = p.sectionid
                     where co.uri in ({}) 
                     and (   p.type in ('Q', 'A')
                            or t.title in ('Q&A', 'Management Discussion'))
                     group by d.docid, co.uri, "category"
                     order by co.uri, "date" desc, "category"
                     )
              select uri, quarter, avg(vader) as vader
              from sector_rows
              where "category" = '{}'
              and quarter > '{}'
              and quarter < '{}'
              group by uri, quarter '''

    # Uses transcripts
    formatted_uris = [x.replace("'", "''") for x in list_uris]
    print(formatted_uris)
    # formatted_query = query_heatmap_row.format(
    #     ", ".join(["'"+x+"'" for x in formatted_uris]), section)
    formatted_query = query_heatmap_row.format(
        ", ".join(["'"+x+"'" for x in formatted_uris]), section, yearLowerBound, yearUpperBound)

    request_df = forge_request(formatted_query, cursor)

    request_df.quarter = pd.to_datetime(
        request_df.quarter).dt.to_period('Q').astype("str")

    request_df = request_df.merge(right=uri_ticker_df, how="left", on="uri")
    request_df = request_df.sort_values("quarter")
    request_df = request_df.pivot_table(
        index='quarter', values='vader', columns='uri')
    request_df = normalize_pct_change_quarter(request_df)

    request_df = pd.melt(request_df.reset_index(),
                         id_vars='quarter',
                         var_name='uri',
                         value_name='vader')

    # print(request_df.index.dtype)
    return request_df.dropna(subset='vader')

# replacement for get_heatmap_data_separated

def get_heatmap_data_separated(sector_level: str,
                               sector_value: str,
                               section: str,
                               startYear: str,
                               endYear: str,
                               region: str,
                               IG_HY: str,
                               cursor):
    """section is one of ('MD', 'Q', 'A')"""
    '''
      const uri_split = data.uri.split('/')

      const company_index = uri_split.indexOf("company") + 1
      const name_split = uri_split[company_index].split("_")
      let name = ''
      for (const i in name_split) {
        name += name_split[i] ? `${name_split[i][0].toUpperCase()}${name_split[i].substring(1)} ` : ''
      }
      data["company"] = name
    '''
    if sector_value:
        heatmap_data = get_heatmap_sector_uris(
            sector_level=sector_level, sector_value=sector_value, section=section, startYear=startYear, endYear=endYear, region=region, IG_HY=IG_HY, cursor=cursor)

        heatmap_data["company"] = heatmap_data.apply(lambda row: get_company_name(row), axis=1)
            
    else:
        heatmap_data = get_heatmap_sector_aggregated(
            sector_level=sector_level, section=section, startYear=startYear, endYear=endYear, region=region, IG_HY=IG_HY, cursor=cursor)

    # data should be reflected in relative change of sentiment
    # wrt the same quarter a year before

    # heatmap_data = normalize_pct_change_quarter(heatmap_data)

    return heatmap_data


