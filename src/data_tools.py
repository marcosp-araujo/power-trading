
import sqlite3
import pandas as pd
import tensorflow as tf
from src.config_tools import Config_Manager

class Data_Manager:
    df_clean: pd.DataFrame
    df_train: pd.DataFrame
    df_valid: pd.DataFrame
    time_train: pd.Series
    x_train: pd.Series
    time_valid: pd.Series
    x_valid: pd.Series
    x_train_window: tf.data.Dataset
    x_valid_window: tf.data.Dataset

    def __init__(self, 
                config:Config_Manager, 
                streamlit = False):
        self.config = config
        self.read_data()
        self.preprocessing()
        self.split_train_test()
        
        if streamlit is False:
            self.x_train_window = self.windowed_dataset(self.x_train)
            self.x_valid_window = self.windowed_dataset(self.x_valid)

    def read_data(self):
        print(f"Loading data from:")
        print(f"    {self.config.data_path}")
        self.df = pd.read_parquet(self.config.data_path)

    def preprocessing(self)->pd.DataFrame:
        ''' (1) Eliminates NaN values from the specified series column.
            (2) Returns a cleaned DataFrame between "series_column" 
                and "start_time".
        '''   
        # Period selection:
        start = self.df[self.config.time_column] >= self.config.start_time
        end = self.df[self.config.time_column] <= self.config.end_time
        print(f"Selected period:")
        print(f"    {self.config.start_time} to {self.config.end_time}")
        # Get a shorter data sample for testing
        df_selection = self.df[start & end]
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

    def windowed_dataset(self, series):

        print("Creating windowed dataset:")
        print(f"    Window size: {self.config.window_size}")
        print(f"    Batch size: {self.config.batch_size}")
        print(f"    Shuffle buffer: {self.config.shuffle_buffer}")

        # Convert dtype
        series = tf.cast(series, tf.float32)

        # Add feature dimension
        series = tf.expand_dims(series, axis=-1)

        dataset = tf.data.Dataset.from_tensor_slices(series)

        dataset = dataset.window(
            self.config.window_size + self.config.horizon,
            shift=1,
            drop_remainder=True
        )

        def batch_window(window):
            return window.batch(self.config.window_size + self.config.horizon)

        def split_window(window):
            return (
                    window[:self.config.window_size],  # inputs
                    window[self.config.window_size:]   # multi-step targets
                    )

        dataset = dataset.flat_map(batch_window)

        dataset = dataset.map(
            split_window,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        dataset = dataset.cache()

        dataset = dataset.shuffle(self.config.shuffle_buffer)

        dataset = dataset.batch(self.config.batch_size)

        dataset_window = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset_window

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

