import pandas as pd
import pyodbc
from tqdm import tqdm
import os
import json
import warnings

warnings.filterwarnings('ignore')

class DataPreparation:
    def __init__(self):
        with open('db_config.json', 'r') as config_file:
            cfg = json.load(config_file)        
        self.connection_string = f'Driver={{SQL Server}};Server={cfg["HOST"]};Database={cfg["DBNAME"]};UID={cfg["USERNAME"]};PWD={cfg["PASSWORD"]};Trusted_Connection=no;'

    def ETL(self, query, start_date, end_date, output_directory):
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        try:
            # Establish a connection to the SQL database
            connection = pyodbc.connect(self.connection_string)

            # Generate a list of dates within the specified range
            date_range = pd.date_range(start=start_date, end=end_date)
            print(f"Quering data from date {start_date} to {end_date}")
            for i, date in enumerate(tqdm(date_range)):
                formatted_date = date.strftime("%m/%d/%Y")
                # Replace date parameter in the SQL query with the current date
                formatted_query = query.replace("{{date}}", formatted_date)
                
                dtype = {'Category': 'category', 'Product Group': 'category', 'Subgroup': 'category', 'Item No_': 'category', 'Status Code': 'category'}
                df = pd.read_sql_query(formatted_query, connection, dtype=dtype, parse_dates=['Date'])
                df.to_parquet(f'{output_directory}/{i+1:03}-{date.strftime("%Y-%m-%d")}.parquet')

        except Exception as e:
            print(f"An error occurred: {str(e)}")

        finally:
            if connection:
                connection.close()

data_loader = DataPreparation()

query = '''
            SELECT Date, TSE.[Receipt No_], TSE.[Item Category Code] AS Category, I.[Product Group Code] AS [Product Group], ISG.Code AS [Subgroup], TSE.[Item No_], item_availability.[Status Code], TSE.[Quantity], TSE.[Price]
            FROM [HO$Trans_ Sales Entry] AS TSE left join [HO$Item] AS I ON TSE.[Item No_] = I.No_
            left join
                (SELECT [Item No_], [Status Code], ROW_NUMBER() OVER(PARTITION BY [Item No_] order by [Starting Date] DESC) AS RN
                FROM [HO$Item Status Link]) AS item_availability
            ON item_availability.[Item No_] = I.No_
            join [HO$Item Sub Group] AS ISG on I.[Item Sub Group] = ISG.Code 
            where TSE.Date = '{{date}}' 
            AND item_availability.RN=1
            AND (item_availability.[Status Code]='LIVE ALL' or item_availability.[Status Code]='NEW ALL')

            order by TSE.Date, TSE.[Receipt No_], TSE.[Item No_]
        '''

data_loader.ETL(query=query, start_date="7/1/2022", end_date="7/2/2022", output_directory="data/test")