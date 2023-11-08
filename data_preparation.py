import pandas as pd
import pyodbc
from tqdm import tqdm
import os
import json
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

class DataPreparation:
    def __init__(self, cfg_file_path: str):
        with open(cfg_file_path, 'r') as config_file:
            cfg = json.load(config_file)
        self.connection_string = f'Driver={{SQL Server}};Server={cfg["HOST"]};Database={cfg["DBNAME"]};UID={cfg["USERNAME"]};PWD={cfg["PASSWORD"]};Trusted_Connection=no;'
        try:
            # Establish a connection to the SQL database
            self.connection = pyodbc.connect(self.connection_string)
            print("----- Connection Established -----")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    def cache_items_info(self, query: str, output_directory: Path):
        if not os.path.exists(output_directory / "items data"):
            os.makedirs(output_directory / "items data")
        
        cursor = self.connection.cursor()
        print("Quering Products Info.....")
        cursor.execute(query)
        df = pd.read_sql("SELECT * FROM #ItemInfo", self.connection)
        df.to_parquet(output_directory / "items data" / "ItemInfo.parquet")
        print(f'Products Info Saved Into: {output_directory / "items data" / "ItemInfo.parquet"}')

    def extract_transactions(self, query: str, start_date: str, end_date: str, output_directory: Path):
        if not os.path.exists(output_directory / "transactions data"):
            os.makedirs(output_directory / "transactions data")
        
        try:
            # Generate a list of dates within the specified range
            date_range = pd.date_range(start=start_date, end=end_date, freq='SM')
            print(f"Quering Transactions from {start_date} to {end_date}.....")
            dtype = {'Item No_': 'category'}
            
            for i in tqdm(range(len(date_range)-1)):
                date_from = date_range[i]
                date_to = date_range[i+1]

                # Replace date parameter in the SQL query with the current date
                formatted_query = query.replace("{{date-from}}", date_from.strftime("%m/%d/%Y"))\
                                       .replace("{{date-to}}", date_to.strftime("%m/%d/%Y"))
                
                df = pd.read_sql_query(formatted_query, self.connection, dtype=dtype, parse_dates=['Date'])
                df.to_parquet(output_directory / "transactions data" / f'{i+1:02}-{date_from.strftime("%Y-%m-%d")}.parquet')

        except Exception as e:
            print(f"An error occurred: {str(e)}")

    def close_connection(self):
        if self.connection:
            self.connection.close()
            print("----- Connection Closed -----")
        

CFG_PATH = "config/server13_db_config.json"
DATA_PATH = Path("data/Loyalty-03-10-2023")
data_loader = DataPreparation(cfg_file_path=CFG_PATH)

query_1 =   '''
            DROP TABLE IF EXISTS #ItemInfo;

            SELECT [Category Code], [Category], [Product Group Code], [Product Group], [Subgroup Code], [Subgroup], [Item No_], [Item], [Product Size], [Status Code]
            INTO #ItemInfo
            FROM (SELECT
                    I.[Item Category Code] AS [Category Code], IG.[Description] AS [Category],
                    I.[Product Group Code], PG.[Description] AS [Product Group],
                    I.[Item Sub Group] AS [Subgroup Code], ISG.[Description] AS [Subgroup],
                    [Item No_], I.[Description 2] AS [Item], I.[Product Size], [Status Code],
                    ROW_NUMBER() OVER(PARTITION BY [Item No_] order by [Starting Date] DESC) AS RN
                FROM [Loyalty$Item Status Link] AS ISL
                JOIN [Loyalty$Item] AS I ON I.[No_] = ISL.[Item No_]
                JOIN [Loyalty$Item Category] AS IG ON IG.[Code] = I.[Item Category Code]
                JOIN [Loyalty$Product Group] AS PG on I.[Product Group Code] = PG.[Code]
                JOIN [Loyalty$Item Sub Group] AS ISG ON I.[Item Sub Group] = ISG.Code) item_availability
            WHERE item_availability.RN=1
            AND (item_availability.[Status Code]='LIVE ALL' or item_availability.[Status Code]='NEW ALL' or item_availability.[Status Code]='BLOCK ALL');
            '''
data_loader.cache_items_info(query_1, DATA_PATH)

query_2 = '''
            SELECT Date, TSE.[Receipt No_], I.[Item No_], TSE.[Quantity], TSE.[Price]
            FROM [Loyalty$Trans_ Sales Entry] AS TSE
            join #ItemInfo AS I ON I.[Item No_] = TSE.[Item No_]
            where (TSE.Date >= '{{date-from}}' and TSE.Date < '{{date-to}}')
            AND TSE.Quantity < 0
            order by TSE.Date, TSE.[Receipt No_], TSE.[Item No_]
        '''
data_loader.extract_transactions(query_2, start_date="10/01/2021", end_date="10/01/2023", output_directory=DATA_PATH)

data_loader.close_connection()