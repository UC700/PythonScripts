# Importing Libraries
import psycopg2
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
import time

start_time = time.time()

# Suppress all warnings
warnings.filterwarnings("ignore")

# Data Connection Configuration

conn = psycopg2.connect(
    dbname='dwh',
    user='uniware_write',
    password='uniware@1234',
    host='dwhprod-in.unicommerce.infra',
    port='5432'
)

# Reading data from DWH for calculating SLA Breach

query = """ select tenant,facility,channel_code,channel_order_created_date as date,
count(*) as shipment_count,
sum(case when coalesce(extract(epoch from uniware_creation_timestamp - channel_order_creation_time)/60,0) + coalesce(shipment_creation_time,0) + coalesce(picklist_creation_time,0) + coalesce(picking_time,0) + coalesce(packing_time,0) > extract(epoch from fulfillment_time - channel_order_creation_time)/60 then 1 else 0 end) as breached_shipment_count
from insights_o2sla where channel_source_code not in ('CUSTOM','CLOUDTAIL') and channel_source_code not like '%B2B%' group by 1,2,3,4
"""

df = pd.read_sql_query(query, conn)

#Function to find daily median

def sql_query(df,column):
    
    df_new = []
    
    median_column = "median_" + column

    query = f"""
    SELECT
        tenant,
        facility,
        channel_code,
        channel_order_created_date AS date,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {column}) AS {median_column}
    FROM
        insights_o2sla
    WHERE
        {column} IS NOT NULL
        and channel_source_code not in ('CUSTOM','CLOUDTAIL') and channel_source_code not like '%B2B%'
        
    GROUP BY
        1, 2, 3, 4
    """

    df_new = pd.read_sql_query(query, conn)
    df = pd.merge(df, df_new, how='left', left_on=['tenant', 'facility', 'channel_code', 'date'], right_on=['tenant', 'facility', 'channel_code', 'date'])
    
    return df

# Calculate daily median
drilldown_periods = ['picking_time','packing_time']

for column in drilldown_periods:
    df = sql_query(df,column)


# Function to filter out facilities and channels with shipment count below a daily threshold

def filter_func(df,col_list,threshold):
    for column in col_list:
        test = []
        test = df.groupby(['tenant',column,'date'])['shipment_count'].sum().reset_index()
        test = test.groupby(['tenant',column])['shipment_count'].median().reset_index()
        test = test[test['shipment_count'] > threshold].drop(columns = ['shipment_count'])
        df = pd.merge(df, test, on=['tenant',column],how = 'inner')
    return df

# Filter facilities and channels
column = ['facility','channel_code']
df = filter_func(df,column,50)

# Function to calculate Z-score

def modified_z_score(data):
    median_value = np.median(data)
    mad_value = np.median(np.abs(data - median_value))
    
    modified_z_scores = 0.6745 * (data - median_value) / mad_value
    
    return modified_z_scores


# Calculate Z-scores 
for column in drilldown_periods:
    z_score_column = f'median_{column}_z_score'
    median_column = f'median_{column}'
    
    df_new = []
    df_new = df[['tenant', 'facility', 'channel_code', 'date', median_column]].dropna()
    df_new[z_score_column] = df_new.groupby(['tenant','facility', 'channel_code'])[median_column].transform(modified_z_score)
    df = pd.merge(df, df_new[['tenant', 'facility', 'channel_code', 'date',z_score_column]], how='left', on=['tenant', 'facility', 'channel_code', 'date'])


# Shipment Column z-score
df['breach_percent'] = 100.00*df['breached_shipment_count'] / df['shipment_count']
df['sla_breach_z_score'] = df.groupby(['tenant','facility', 'channel_code'])['breach_percent'].transform(modified_z_score)

# Filtering data basis of SLA Breach Threshold of 0.5 %
df['breach_percent'] = 100.00*df['breached_shipment_count'] / df['shipment_count']
df = df[df['breach_percent'] > 0.5]  

# Filtering out more than 3.5 z score
df = df[df['sla_breach_z_score'] > 3.5]

# Calculating Historical Median Facility X Channel

def sql_query_hist(df,column):
    
    df_new = []
    
    median_column = "Hist_median_" + column

    query = f"""
    SELECT
        tenant,
        facility,
        channel_code,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {column}) AS {median_column}
    FROM
        insights_o2sla
    WHERE
        {column} IS NOT NULL
        and channel_source_code not in ('CUSTOM','CLOUDTAIL') and channel_source_code not like '%B2B%'
        
    GROUP BY
        1, 2, 3
    """

    df_new = pd.read_sql_query(query, conn)
    df = pd.merge(df, df_new, how='left', left_on=['tenant', 'facility', 'channel_code'], right_on=['tenant', 'facility', 'channel_code'])
    
    return df

# Calculate hist median
periods = ['picking_time','packing_time']

for column in periods:
    df = sql_query_hist(df,column)


def reason(row):
    median = 0
    hist_median = 0
    if row['median_picking_time_z_score'] > 2.5 and row['median_packing_time_z_score'] > 2.5:
        median = row['median_picking_time'] + row['median_packing_time']
        hist_median = row['hist_median_picking_time'] + row['hist_median_packing_time']
        return 'Delay due to Slow Picking and Packing',median, hist_median
    elif row['median_picking_time_z_score'] > 2.0:
        median = row['median_picking_time']
        hist_median = row['hist_median_picking_time']
        return 'Delay due to Slow Picking',median, hist_median
    elif row['median_packing_time_z_score'] > 2.0:
        median = row['median_packing_time']
        hist_median = row['hist_median_packing_time']
        return 'Delay due to Slow Packing',median, hist_median
    else:
        return 'Delay due to Slow Picklist Creation',0,0

df['Anomaly_Reason'], df['Anomaly_Median'], df['Anomaly_Hist_Median'] = zip(*df.apply(reason, axis=1))


columns_to_drop = df.columns[df.columns.str.contains('manifest|dispatch|z_score')]
df = df.drop(columns=columns_to_drop)


# Filtering Data based on last week and last month threshold of anomalies
def filter_dates(group):
    # Calculate the date ranges
    last_week = (datetime.now() - timedelta(days=7)).date()
    last_month = (datetime.now() - timedelta(days=30)).date()
    last_quarter = (datetime.now() - timedelta(days=90)).date()
    # Filter dates based on conditions
    if len(group[group['date'] >= last_week]) > 3:
        group = group[group['date'] >= last_week]
    elif len(group[group['date'] >= last_month]) > 3:
        group = group[group['date'] >= last_month]
    elif len(group[group['date'] >= last_quarter]) > 3:
        group = group[group['date'] >= last_quarter]    
    return group

df = df.groupby(['tenant']).apply(filter_dates).reset_index(drop=True)

# Conosolidating Dates 
def check_consecutive_dates(dates):
 
    dates = sorted(dates)
    
    my_dict = {}
    my_list = []
    for i, date in enumerate(dates):
        if i == 0:
            my_list.append(date)
        elif (dates[i] - dates[i - 1]).days <= 2:
            my_list.append(date)
        else:
            my_dict[my_list[0]] = my_list
            my_list = [date]
            
    my_dict[my_list[0]] = my_list
    
    return my_dict

result_facility_channel = df.groupby(['tenant', 'facility', 'channel_code', 'Anomaly_Reason'])['date'].apply(lambda x: check_consecutive_dates(x))
consolidated_df_facility_channel = pd.DataFrame(result_facility_channel.reset_index()).dropna().drop(columns = ['level_4'])


# Addidng Shipment and breach% for anomalies

for index, row in consolidated_df_facility_channel.iterrows():
    dates_list = row['date']  
    
    total_shipment_count = 0
    total_breached_shipments = 0
    Time = 0
    Historical_Time = 0
    for date in dates_list:
  
        filtered_rows = df[(df['tenant'] == row['tenant']) & 
                             (df['facility'] == row['facility']) & 
                             (df['channel_code'] == row['channel_code']) & 
                             (df['Anomaly_Reason'] == row['Anomaly_Reason']) & 
                             (df['date'] == date)]
        
        shipment_count = filtered_rows['shipment_count'].sum()
        breached_shipment_count = filtered_rows['breached_shipment_count'].sum()
        Anomaly_Median = filtered_rows['Anomaly_Median'].sum()
        Anomaly_Hist_Median = filtered_rows['Anomaly_Hist_Median'].sum()

        
        total_shipment_count += shipment_count
        total_breached_shipments += breached_shipment_count
        Time += Anomaly_Median
        Historical_Time += Anomaly_Hist_Median

    consolidated_df_facility_channel.at[index, 'total_shipment_count'] = total_shipment_count
    consolidated_df_facility_channel.at[index, 'total_breached_shipments'] = total_breached_shipments
    consolidated_df_facility_channel.at[index, 'time'] = Time
    consolidated_df_facility_channel.at[index, 'historical_time'] = Historical_Time

consolidated_df_facility_channel['breach_percent'] = (100.00*consolidated_df_facility_channel['total_breached_shipments'] / consolidated_df_facility_channel['total_shipment_count']).round(2)

#Statement
consolidated_df_facility_channel['alert'] = consolidated_df_facility_channel.apply(lambda row: str(row['breach_percent']) + '% (' + str(row['total_breached_shipments']) + ') breach on ' + row['facility'] + ' and ' + row['channel_code'], axis=1)
consolidated_df_facility_channel['dates'] = consolidated_df_facility_channel.apply(lambda row: ','.join([date.strftime('%Y-%m-%d') for date in row['date']]), axis=1)
consolidated_df_facility_channel = consolidated_df_facility_channel.drop(columns = ['date','total_shipment_count','total_breached_shipments','breach_percent'])
consolidated_df_facility_channel['historical_time'].replace(0, '', inplace=True)
consolidated_df_facility_channel['time'].replace(0, '', inplace=True)


## NEW CODE


consolidated_df_facility_channel.rename(columns={'Anomaly_Reason': 'anomaly_reason'}, inplace=True)
consolidated_df_facility_channel['time'] = pd.to_numeric(consolidated_df_facility_channel['time'], errors='coerce')
consolidated_df_facility_channel['time'] = consolidated_df_facility_channel['time'].round(1)
consolidated_df_facility_channel['time'].fillna('', inplace=True)
consolidated_df_facility_channel['historical_time'] = pd.to_numeric(consolidated_df_facility_channel['historical_time'], errors='coerce')
consolidated_df_facility_channel['historical_time'] = consolidated_df_facility_channel['historical_time'].round(1)
new_dates = []
consolidated_df_facility_channel['dates'] = consolidated_df_facility_channel['dates'].apply(lambda x: f"'{x}'")
for date in consolidated_df_facility_channel['dates']:
    if ',' in date:
        date = date.replace(",","','")
        new_dates.append(date)
    else:
        new_dates.append(date)
consolidated_df_facility_channel['dates'] =  new_dates



#Inserting Data in DWH
engine = create_engine("postgresql+psycopg2://uniware_write:uniware%401234@dwhprod-in.unicommerce.infra:5432/dwh")

# Specify the name of the PostgreSQL table where you want to insert the data
table_name = 'insights_o2sla_anomaly'

# # Insert the DataFrame into the PostgreSQL table
consolidated_df_facility_channel.to_sql(table_name, engine,if_exists='replace', index=False)

# # Close the SQLAlchemy engine (optional)
engine.dispose()


end_time = time.time()

execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")
