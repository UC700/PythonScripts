import psycopg2
from sqlalchemy import create_engine
import pandas as pd
from sqlalchemy import create_engine

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

query = """select tenant,facility,channel_source_code as channel_source,channel_order_created_date as date,
count(*) as shipment_count,
sum(case when sla_breached_flag then 1 else 0 end) as breached_shipment_count,
sum(picking_time) as pt,sum(packing_time) as pat,
sum(manifest_time) as mt,sum(dispatch_time) as dt
from insights_o2sla where channel_source_code != 'CUSTOM' group by 1,2,3,4"""

df = pd.read_sql_query(query, conn)

aggregate_columns = ['pt', 'pat', 'mt', 'dt']

# Function to calculate Z-score
def calculate_z_score(series):
    z_score = (series - series.mean()) / series.std()
    return z_score

# Calculate Z-scores for each column within groups at facility and facility-channel level
for column in aggregate_columns:
    z_score_column = f'{column}_z_score_fac_cha'
    df[z_score_column] = df.groupby(['tenant','facility', 'channel_source'])[column].transform(calculate_z_score)

for column in aggregate_columns:
    z_score_column = f'{column}_z_score_fac'
    df[z_score_column] = df.groupby(['tenant','facility'])[column].transform(calculate_z_score)

 # Calculating 50th percentile of shipment count for each tenant and then filtering the rest of data
quartile_data = df.groupby(['tenant'])['shipment_count'].describe(percentiles=[.5])
df = pd.merge(df, quartile_data, how='inner', left_on='tenant', right_on='tenant')
df = df[df['shipment_count'] > df['50%']]
columns_to_drop = ['count','mean', 'std','min','50%','max']
df.drop(columns=columns_to_drop, inplace=True)

# Filtering data basis of SLA Breach Threshold of 0.5 %
df['breach_percent'] = 100.00*df['breached_shipment_count'] / df['shipment_count']
df = df[df['breach_percent'] > 0.5]

# Picking out max two reasons for SLA Breach on these dates

#Facility Level
z_score_columns = ['pt_z_score_fac', 'pat_z_score_fac', 'mt_z_score_fac', 'dt_z_score_fac']
df['Top2Columns_fac'] = df[z_score_columns].apply(lambda row: row.nlargest(2).index.tolist(), axis=1)
df['Top1Reason_fac'], df['Top2Reason_fac'] = zip(*df['Top2Columns_fac'].apply(lambda x: tuple(x[:2])))
df.drop('Top2Columns_fac', axis=1, inplace=True)


#Facility Channel Level
z_score_columns = ['pt_z_score_fac_cha', 'pat_z_score_fac_cha', 'mt_z_score_fac_cha', 'dt_z_score_fac_cha']
df['Top2Columns_fac_cha'] = df[z_score_columns].apply(lambda row: row.nlargest(2).index.tolist(), axis=1)
df['Top1Reason_fac_cha'], df['Top2Reason_fac_cha'] = zip(*df['Top2Columns_fac_cha'].apply(lambda x: tuple(x[:2])))
df.drop('Top2Columns_fac_cha', axis=1, inplace=True)

#Mapping Column Names to shipment flow

mapping_dict = {'pt_z_score_fac': 'Picking Time', 'pat_z_score_fac': 'Packing Time', 'mt_z_score_fac': 'Manifest Time', 'dt_z_score_fac': 'Dispatch Time'}

df['Top1Reason_fac'] = df['Top1Reason_fac'].map(mapping_dict)
df['Top2Reason_fac'] = df['Top2Reason_fac'].map(mapping_dict)

mapping_dict = {'pt_z_score_fac_cha': 'Picking Time', 'pat_z_score_fac_cha': 'Packing Time', 'mt_z_score_fac_cha': 'Manifest Time', 'dt_z_score_fac_cha': 'Dispatch Time'}

df['Top1Reason_fac_cha'] = df['Top1Reason_fac_cha'].map(mapping_dict)
df['Top2Reason_fac_cha'] = df['Top2Reason_fac_cha'].map(mapping_dict)

# Dropping z_score columns
df = df.drop(df.filter(like='z_score').columns, axis=1)

#Inserting Data into DWH

# Database connection parameters
ddb_params = {
    'host': 'nifi1-in.unicommerce.infra',
    'port': '5432',
    'user': 'uniware_write',
    'password': 'uniware%401234',
    'database': 'dwh',
}

engine = create_engine("postgresql+psycopg2://uniware_write:uniware%401234@nifi1-in.unicommerce.infra:5432/dwh")

# Specify the name of the PostgreSQL table where you want to insert the data
table_name = 'insights_o2sla_anomaly'

engine.execute("truncate table insights_o2sla_anomaly")

# # Insert the DataFrame into the PostgreSQL table
df.to_sql(table_name, engine.connect(),if_exists='replace', index=False)

# # Close the SQLAlchemy engine (optional)
engine.dispose()