import pandas as pd

def comp_pref(df):
    '''
    input: Pandas DataFrame with row index slate options, column headers deciders
            the entries are the preferences. Entry at row i, column j is the 
            preference ranking of decider j of slate option i. Preference m
            indicates un-expressed preference
    input: Pandas DataFrame with row index slate options, column headers deciders
            the entries are the preferences. Entry at row i, column j is the 
            complement preference ranking of decider j of slate option i
    '''
    m,n = df.shape
    return df - m

def main():
    print_to_screen = True
    save_to_file = False

    filename = 'Data/med/clean/S.csv'
    temp = filename.split('.')
    output_filename = temp[0] + '_complement' + temp[1]

    pref_df = pd.read_csv(filename, header=0, index_col=0)
    pref_df.index = pref_df.index.map(str)
    comp_df = comp_pref(pref_df)

    if print_to_screen:
        print('-----------------------------------')
        print('Original Preferences')
        print('-----------------------------------')
        print(pref_df.head())
        print('-----------------------------------')
        print('Seeker Similarity')
        print('-----------------------------------')
        print(comp_df.head())
        print('-----------------------------------')

    if save_to_file:
        comp_df.to_csv(output_filename, header=True, index=True)

if __name__ == '__main__':
    main()
