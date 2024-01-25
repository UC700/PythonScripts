# Importing Libraries

import psycopg2
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
# Suppress all warnings
warnings.filterwarnings("ignore")

# Data Connection Configuration
db_params = {
    'host': 'nifi1-in.unicommerce.infra',
    'port': '5432',
    'user': 'uniware_write',
    'password': 'uniware@1234',
    'database': 'dwh',
}

conn = psycopg2.connect(**db_params)

# Reading data from DWH

query = """ select tenant,facility,channel_code,channel_order_created_date as date,
count(*) as shipment_count,
sum(case when coalesce(extract(epoch from uniware_creation_timestamp - channel_order_creation_time)/60,0) + coalesce(shipment_creation_time,0) + coalesce(picklist_creation_time,0) + coalesce(picking_time,0) + coalesce(packing_time,0) > extract(epoch from fulfillment_time - channel_order_creation_time)/60 then 1 else 0 end) as breached_shipment_count
from insights_o2sla group by 1,2,3,4 """

df = pd.read_sql_query(query, conn)

aggregate_columns_2 = ['picking_time', 'packing_time','manifest_time','dispatch_time']


def sql_query(df,column):
    
    df_new = []
    
    median_column = "median_" + column
    mean_column = "mean_" + column

    query = f"""
    SELECT
        tenant,
        facility,
        channel_code,
        channel_order_created_date AS date,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {column}) AS {median_column},
        AVG({column}) AS {mean_column}
    FROM
        insights_o2sla
    WHERE
        {column} IS NOT NULL
    GROUP BY
        1, 2, 3, 4
    """

    df_new = pd.read_sql_query(query, conn)
    df = pd.merge(df, df_new, how='left', left_on=['tenant', 'facility', 'channel_code', 'date'], right_on=['tenant', 'facility', 'channel_code', 'date'])
    
    return df

# Example usage:
for column in aggregate_columns_2:
    df = sql_query(df,column)


aggregate_columns = ['median_picking_time', 'median_packing_time','median_manifest_time','median_dispatch_time' ]

# Function to calculate Z-score

def modified_z_score(data):
    median_value = np.median(data)
    mad_value = np.median(np.abs(data - median_value))
    
    modified_z_scores = 0.6745 * (data - median_value) / mad_value
    
    return modified_z_scores

    
# Calculate Z-scores for each column within groups at facility and facility-channel level
for column in aggregate_columns:
    z_score_column = f'{column}_z_score'
    df_new = []
    df_new = df[['tenant', 'facility', 'channel_code', 'date', column]].dropna()
    df_new[z_score_column] = df_new.groupby(['tenant','facility', 'channel_code'])[column].transform(modified_z_score)
    df = pd.merge(df, df_new[['tenant', 'facility', 'channel_code', 'date',z_score_column]], how='left', on=['tenant', 'facility', 'channel_code', 'date'])


start_date_last_30_days = (datetime.now() - timedelta(days=30)).date()

# Filter rows where the 'date' column is within the last 30 days
df = df[df['date'] >= start_date_last_30_days]

 # Calculating 25th percentile of shipment count for each tenant X facility and then filtering the rest of data
quartile_data = df.groupby(['tenant','facility'])['shipment_count'].describe(percentiles=[.25])
columns_to_drop = ['count','mean', 'std','min','50%','max']
quartile_data.drop(columns=columns_to_drop, inplace=True)
df = pd.merge(df, quartile_data, how='inner', left_on= ['tenant','facility'] , right_on= ['tenant','facility'])
df = df[df['shipment_count'] > df['25%']]
df.drop(columns = '25%', inplace=True)

# Filtering data basis of SLA Breach Threshold of 0.5 %
df['breach_percent'] = 100.00*df['breached_shipment_count'] / df['shipment_count']
df = df[df['breach_percent'] > 0.5]


engine = create_engine("postgresql+psycopg2://uniware_write:uniware%401234@nifi1-in.unicommerce.infra:5432/dwh")

# Specify the name of the PostgreSQL table where you want to insert the data
table_name = 'insights_o2sla_anomaly'

engine.execute("truncate table insights_o2sla_anomaly")

# # Insert the DataFrame into the PostgreSQL table
df.to_sql(table_name, engine.connect(),if_exists='replace', index=False)

# # Close the SQLAlchemy engine (optional)
engine.dispose()

print("Script Ran Successfully")
