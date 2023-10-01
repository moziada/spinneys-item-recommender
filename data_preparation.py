import pandas as pd
import pyodbc
from tqdm import tqdm
import os
import json
import warnings

warnings.filterwarnings('ignore')

class DataPreparation:
    def __init__(self, cfg_file_path: str):
        with open(cfg_file_path, 'r') as config_file:
            cfg = json.load(config_file)
        self.connection_string = f'Driver={{SQL Server}};Server={cfg["HOST"]};Database={cfg["DBNAME"]};UID={cfg["USERNAME"]};PWD={cfg["PASSWORD"]};Trusted_Connection=no;'

    def ETL(self, query, start_date, end_date, output_directory):
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        try:
            # Establish a connection to the SQL database
            connection = pyodbc.connect(self.connection_string)

            # Generate a list of dates within the specified range
            date_range = pd.date_range(start=start_date, end=end_date, freq='SM')
            print(f"Quering data from date {start_date} to {end_date}")
            dtype = {'Category': 'category', 'Product Group': 'category', 'Subgroup': 'category', 'Item No_': 'category', 'Status Code': 'category'}
            
            for i in tqdm(range(len(date_range)-1)):
                date_from = date_range[i]
                date_to = date_range[i+1]

                # Replace date parameter in the SQL query with the current date
                formatted_query = query.replace("{{date-from}}", date_from.strftime("%m/%d/%Y"))\
                                       .replace("{{date-to}}", date_to.strftime("%m/%d/%Y"))
                
                df = pd.read_sql_query(formatted_query, connection, dtype=dtype, parse_dates=['Date'])
                df.to_parquet(f'{output_directory}/{i+1:02}-{date_from.strftime("%Y-%m-%d")}.parquet')

        except Exception as e:
            print(f"An error occurred: {str(e)}")

        finally:
            if connection:
                connection.close()

data_loader = DataPreparation(cfg_file_path="config/server13_db_config.json")

query = '''
            SELECT Date, TSE.[Receipt No_], TSE.[Item Category Code] AS Category, I.[Product Group Code] AS [Product Group], ISG.Code AS [Subgroup], TSE.[Item No_], item_availability.[Status Code], TSE.[Quantity], TSE.[Price]
            FROM [Loyalty$Trans_ Sales Entry] AS TSE left join [Loyalty$Item] AS I ON TSE.[Item No_] = I.No_
            left join
                (SELECT [Item No_], [Status Code], ROW_NUMBER() OVER(PARTITION BY [Item No_] order by [Starting Date] DESC) AS RN
                FROM [Loyalty$Item Status Link]) AS item_availability
            ON item_availability.[Item No_] = I.No_
            join [Loyalty$Item Sub Group] AS ISG on I.[Item Sub Group] = ISG.Code 
            where (TSE.Date >= '{{date-from}}' and TSE.Date < '{{date-to}}')
            AND item_availability.RN=1
            AND (item_availability.[Status Code]='LIVE ALL' or item_availability.[Status Code]='NEW ALL' or item_availability.[Status Code]='BLOCK ALL')
            AND TSE.Quantity < 0
            order by TSE.Date, TSE.[Receipt No_], TSE.[Item No_]
        '''

data_loader.ETL(query=query, start_date="8/01/2020", end_date="8/01/2023", output_directory="data/Loyalty-30-09-2023")