import numpy as np
import os
import pandas as pd
import sys

from iop import iop

def main():
    if len(sys.argv) != 2:
        raise ValueError('''
                Provide the filename of preference csv.
                Example:
                        python complete_prefs.py path/to/pref.csv 
                ''')

    pref_filename = sys.argv[1] 

    print_to_screen = True
    save_to_file = True

    temp = pref_filename.split('.')
    pref_filename = sys.argv[1] 
    output_filename = temp[0] + '_complete.' + temp[1]
    df = pd.read_csv(pref_filename, header=0, index_col=0)
    df.index = df.index.map(str)
    iop_e = iop(df) 
    iop_df = iop_e.complete_prefs(measure='cosine')
    
    if print_to_screen:
        print('-----------------------------------')
        print('Complete Preferences')
        print('Preference_Coverage: {}'.format(iop_e.pref_coverage))
        print('-----------------------------------')
        print(iop_df.head())
        print('-----------------------------------')

    if save_to_file:
        iop_df.to_csv(output_filename, header=True, index=True)

if __name__ == '__main__':
    main()
