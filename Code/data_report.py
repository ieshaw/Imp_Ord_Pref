import pandas as pd
import sys

def main():
    if len(sys.argv) != 2:
        raise ValueError('''
                Provide the filename of preference csv.
                Example:
                        python complete_prefs.py path/to/pref.csv 
                ''')

    pref_filename = sys.argv[1] 
    df = pd.read_csv(pref_filename, header=0, index_col=0)
    m,n = df.shape
    print("Number of Participants: {}".format(n))
    print("Number of Options: {}".format(m))
    print("Percent Complete: {:.2f}%".format(100 *(1 - (1/(m*n)) * df.isna().sum().sum())))

if __name__ == '__main__':
    main()
