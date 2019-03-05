import numpy as np
import pandas as pd
import sys

def pref_dict_to_df(pref_dict):
    '''
    input pref_dict: dictionary of dictionaries, outer keys deciders, inner keys ranked entities,           values ordinal ranking (1 being most preferred)
    output df: Pandas DataFrame with row index slate options, column headers deciders
                the entries are the preferences. Entry at row i, column j is the 
                preference ranking of decider j of slate option i
    '''
    out_df = pd.DataFrame.from_dict(pref_dict, dtype=int)
    return out_df

def clean_s(filename):
    '''
    input filename: str, string of preference csv
    output S_df: Pandas DataFrame with row index slate options, column headers deciders
                the entries are the preferences. Entry at row i, column j is the 
                preference ranking of decider j of slate option i
    '''
    df = pd.read_csv(filename)
    pref_dict = {}
    for col in df.columns:
        s = df[col].dropna()
        pref_dict[col] = pd.Series(s.index.values + 1, index=s).to_dict()
    return pref_dict_to_df(pref_dict)

def clean_o(filename):
    '''
    input filename: str, string of preference csv
    output S_df: Pandas DataFrame with row index slate options, column headers deciders
                the entries are the preferences. Entry at row i, column j is the 
                preference ranking of decider j of slate option i
    '''
    df = pd.read_csv(filename)
    df.drop(0, inplace=True)
    pref_dict = {}
    for col in df.columns:
        s = df[col].dropna()
        pref_dict[col] = pd.Series(s.index.values, index=s).to_dict()
    return pref_dict_to_df(pref_dict)

def find_complement(list_1, list_2):
    '''
    input list_1, list_2: lists of strings
    output missing: entries from list2 not in list1 
    '''
    return list(set(list_2) - set(list_1))

def grab_list(df, axis):
    '''
    input df: Pandas DataFrame
    input axis: str, 'row' or 'col'
    output out_list: list of indices from the axis
    '''
    if axis == 'col':
        out_list = df.columns
    else:
        out_list = df.index.values
    return out_list

def insert_complement(main_df, main_df_axis, compare_df, compare_df_axis):
    '''
    input main_df: Pandas DataFrame
    input main_df_axis: string, 'col' or 'row' suspected to be missing elements
    input compare_df: Pandas DataFrame to be compared to
    input compare_df_axis: string, 'col' or 'row' to be compared to 
    '''
    main_list = grab_list(main_df, main_df_axis)
    compare_list = grab_list(compare_df, compare_df_axis)
    complement = find_complement(main_list, compare_list)
    print(complement)
    print(len(complement))
    print(main_df_axis)
    if len(complement) > 0:
        if main_df_axis == 'col':
            main_df[complement] = np.nan
        elif main_df_axis == 'row':
            new_df = pd.DataFrame(np.nan, index=complement, columns=main_df.columns)
            print(new_df.head())
            main_df = main_df.append(new_df)
    return main_df

def deconflict(S_df, O_df):
    '''
    input S_df, O_df: Pandas DataFrames that SHOULD be tranpose shapes of one another,
                    preferences of opposite sides of two sided matching markets
                    un-expressed preferences are NaN
    output S_df, O_df: Pandas DataFrames that ARE transpose shapes of one another,
                    preferences of opposite sides of two sided matching markets
                    un-expressed preferences are NaN
    '''
    axis_list = ['row', 'col']
    for main_axis in axis_list:
        for compare_axis in axis_list:
            if compare_axis != main_axis:
                print('S_df')
                S_df = insert_complement(S_df, main_axis, O_df, compare_axis)
                print('O_df')
                O_df = insert_complement(O_df, compare_axis, S_df, main_axis)
    return S_df, O_df

def main():
    data_dir = 'Data/med/raw/redacted_final_'
    s_file = 'officer.csv'
    o_file = 'command.csv'
    output_dir = 'Data/clean/'
    save_to_file = False
    print_to_screen = True
    
    S_df = clean_s(data_dir + s_file)
    O_df = clean_o(data_dir + o_file)
    S_df, O_df = deconflict(S_df, O_df)

    if print_to_screen:
        print('-----------------------------')
        print('S DF')
        print('-----------------------------')
        print('Shape: {}'.format(S_df.shape))
        print('-----------------------------')
        print(S_df.head())
        print('-----------------------------')
        print('O DF')
        print('-----------------------------')
        print('Shape: {}'.format(O_df.shape))
        print('-----------------------------')
        print(O_df.head())
        print('-----------------------------')

    if save_to_file:
        S_df.to_csv(output_dir + 'S.csv', header=True, index=True)
        O_df.to_csv(output_dir + 'O.csv', header=True, index=True)


if __name__ == '__main__':
    main()
