from ML.data_processing.data_normalization import DataNormalization
from ML.data_strategy.i_o_data_provider import IOProvider
from market_data.ohlc_data import obtain_ohlc_data
import pandas as pd

class Warehouse():
    """
    This Class is a playground for manual
    data processing.
    """
    def process_stock_data(self):

        # Obtain Data
        iop = IOProvider("AAPL")

        input_matrix = iop.obtain_input_matrix()

        output_vector, _ = iop.obtain_output_vector(
            "get_buy_signals"
        )

        # Normalize individual columns
        input_matrix = DataNormalization.normalize_all_columns_matrix_0_1(input_matrix)

        # Normalize Target
        # output_vector = DataNormalization.normalize_column_0_1(output_vector, None)

        # Intify
        output_vector = output_vector.astype('int')

        # Remove Rows with 0.
        input_matrix = pd.DataFrame(input_matrix)
        input_matrix.fillna(input_matrix.mean(), inplace=True)
        input_matrix.to_numpy()

        # Export Data
        pd.DataFrame(input_matrix).to_csv(
            r'/Users/rahul/Main/CloudStation/Spizen/spizen-forex/master-insight-trading-bot-2/ML/data_processing/norm_A.csv',
            index=False,
            header=True
        )

        pd.DataFrame(output_vector).to_csv(
            r'/Users/rahul/Main/CloudStation/Spizen/spizen-forex/master-insight-trading-bot-2/ML/data_processing/norm_b.csv',
            index=False,
            header=True
        )

        p = 0



if __name__ == '__main__':
    w = Warehouse()
    w.process_stock_data()
