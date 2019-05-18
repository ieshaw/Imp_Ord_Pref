import pandas as pd
import sys

def main():

    pref_filename = "Data/cyber/sailors/S_clean.csv" 
    df = pd.read_csv(pref_filename, header=0, index_col=0)
    m,n = df.shape
    print("Number of Participants: {}".format(n))
    print("Number of Options: {}".format(m))
    print("Total Number of Preferences: {}".format(m*n))
    print("Percent Complete: {:.2f}%".format(100 *(1 - (1/(m*n)) * df.isna().sum().sum())))
    results_filename = "Data/cyber/sailors/results/results.csv" 
    r_df = pd.read_csv(results_filename, header=0, index_col=0)
    r_df.reset_index(inplace=True)
    r_df['pref_count'] = (r_df['Pref_Coverage'] * m * n).round()
    r_df['pref_diff'] = r_df['pref_count'] - r_df['pref_count'].shift(1)
    print(r_df.head())
    #r_df.to_csv("temp.csv")

if __name__ == '__main__':
    main()
