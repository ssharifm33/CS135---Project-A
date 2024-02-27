import numpy as np
import pandas as pd
import os


def overview_data(x_train_df, y_train_df):

    N, n_cols = x_train_df.shape
    print("Shape of x_train_df: (%d, %d)" % (N,n_cols))
    print("Shape of y_train_df: %s" % str(y_train_df.shape))

    # print a row in the dataframe as a list of two strings: website and the review. 
    print("\nPrint three instance from the training set in lists:")
    print("Input (website, review):")
    print((x_train_df.iloc[0:3, :]).values.tolist())
    print("Label:")
    print(y_train_df.iloc[0:3, 0].tolist())

    # Print out the first five rows and last five rows
    print("\n")
    print("More data from training set:")
    tr_text_list = x_train_df['text'].values.tolist()
    rows = np.arange(0, 5)
    for row_id in rows:
        text = tr_text_list[row_id]
        print("row %5d | y = %d | %s" % (row_id, y_train_df.values[row_id,0], text))

    print("...")
    rows = np.arange(N - 5, N)
    for row_id in rows:
        text = tr_text_list[row_id]
        print("row %5d | y = %d | %s" % (row_id, y_train_df.values[row_id,0], text))




def main(data_dir='../data_reviews'):

    # overview the training data
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))

    overview_data(x_train_df, y_train_df)
    
if __name__ == '__main__':
    main()
