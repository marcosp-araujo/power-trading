
import sqlite3
import pandas as pd
import numpy as np
from src.libs.config_tools import Config_Manager

class Data_Manager:
    df_clean: pd.DataFrame
    df_train: pd.DataFrame
    df_valid: pd.DataFrame
    time_train: pd.Series
    x_train: pd.Series
    time_valid: pd.Series
    x_valid: pd.Series

    def __init__(self, 
                config:Config_Manager):
        self.config = config
        self.read_data()
        self.preprocessing()
        self.split_train_test()
        self.x_train_window, self.y_train_window = self.windowed_dataset_numpy(self.x_train)
        self.x_valid_window, self.y_valid_window = self.windowed_dataset_numpy(self.x_valid)

    def read_data(self):
        print(f"Loading data from:")
        print(f"    {self.config.data_path}")
        selected_columns = [self.config.time_column,
                            self.config.series_column]
        self.df_raw = pd.read_parquet(self.config.data_path,
                                        columns=selected_columns)

    def preprocessing(self)->pd.DataFrame:
        ''' (1) Eliminates NaN values from the specified series column.
            (2) Returns a cleaned DataFrame between "series_column" 
                and "start_time".
        '''   
        # Period selection:
        start = self.df_raw[self.config.time_column] >= self.config.start_time
        end = self.df_raw[self.config.time_column] <= self.config.end_time
        print(f"Selected period:")
        print(f"    {self.config.start_time} to {self.config.end_time}")
        
        # Get the selected period and drop NaN values in the series column
        df_selection = self.df_raw[start & end]
        df_clean = df_selection.dropna(subset=[self.config.series_column])
        nan_percentage = round((len(df_selection)-len(df_clean))*100/len(df_selection),2)
        print(f'Data loss after dropping NaNs in the series column: {nan_percentage}%')
        print(f"    {nan_percentage}%")
        self.df_clean = df_clean

    def split_train_test(self):
        '''Splits the DataFrame into training and validation 
            sets based on the split_time index.
        '''
        print(f"Splitting data into training and validation sets:")
        print(f"    Train size: {self.config.train_size*100}%")

        df_clean = self.df_clean
        self.train_index = int(len(df_clean)*self.config.train_size)
        df_train = df_clean.iloc[:self.train_index]
        df_valid = df_clean.iloc[self.train_index:]

        # Get the train set 
        self.time_train = df_train[self.config.time_column]
        self.x_train = df_train[self.config.series_column]

        # Get the validation set
        self.time_valid = df_valid[self.config.time_column]
        self.x_valid = df_valid[self.config.series_column]
  
    def windowed_dataset_numpy(self, series):

        series = np.array(series)
        full_window = self.config.window_size + self.config.horizon
        windows = []

        for i in range(len(series) - self.config.window_size):
            window = series[i: i + full_window]

            X = window[:self.config.window_size]
            y = window[-self.config.horizon:]

            windows.append((X, y))

        # Shuffle the windows
        np.random.shuffle(windows)

        X = np.array([item[0] for item in windows])
        y = np.array([item[1] for item in windows])

        return X, y

def get_table_from_db(db_file_path:str, 
                      table_name:str, 
                      save_file_path:str=None
                      ) ->pd.DataFrame:
    '''
    Actions:
        (1) Reads a table from a SQLite database and 
            returns it as a DataFrame.
        (2) Saves the table as a parquet file if 
           "save_file_path" is provided.
    Args:
        db_file_path: Path to the SQLite database file.
        table_name: Name of the table to read. The options are
            - "time_series_15min_singleindex"
            - "time_series_30min_singleindex"
            - "time_series_60min_singleindex"
        saves_file_path (optional): Path to save the 
        table as a parquet file.
    '''
    conn = sqlite3.connect(db_file_path)    
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Fixing timestamp columns to datetime format with UTC timezone
    df.loc[:,['cet_cest_timestamp']] = pd.to_datetime(df['cet_cest_timestamp'], utc=True)
    df.loc[:,['utc_timestamp']] = pd.to_datetime(df['utc_timestamp'], utc=True)

    if save_file_path:
        print( f"Saving table to {save_file_path}")
        df.to_parquet(save_file_path)
        
    return df

