import pandas as pd
import sys

def load_df(filename):
    '''
    input filename: string, filename of csv with ordinal preferences of columns for 
            rows; NaN indicates unexpressed preference
    output: Pandas DataFrame with row index slate options, column headers deciders
            the entries are the preferences. Entry at row i, column j is the 
            complement ordinal preference ranking of decider j of slate option i
    '''
    df = pd.read_csv(filename, header=0, index_col=0)
    df.index = df.index.map(str)
    m,n = df.shape
    df.fillna(m, inplace=True)
    df = m - df
    ## Error Checking
    if df.values.max() > m or df.values.min() < 0:
        raise ValueError('''
                Preference CSV should have ordinal preferences
                of columns for rows. Somthing is wrong.
                ''')
    return df

def sim_weighted_euclidean(df):
    '''
    input: Pandas DataFrame with row index slate options, column headers deciders
            the entries are the preferences. Entry at row i, column j is the 
            complement ordinal preference ranking of decider j of slate option i
    output: sim_df, Pandas DataFrame with row index deciders, column headers deciders
            the entries are the similarities. Entry at row i, column j is the 
            similarity of decider j of decider i
    '''
    sim_df = pd.DataFrame(data=-1,index=df.columns, columns=df.columns, dtype=float)
    m,n = df.shape
    sim_coeff = 1/(2*(m**4))
    for i in df:
        for j in df:
            if i == j:
                sim_df[i][i] = 1
            elif sim_df[i][j] == -1:
                s = 1 - sim_coeff * ((df[i] + df[j]) * ((df[i] - df[j])**2)).sum()
                sim_df[i][j] = s
                sim_df[j][i] = s
    ## Error Checking
    mx = sim_df.values.max()
    mn = sim_df.values.min()
    if mx > 1:
        raise ValueError('''
                Weighted Euclidean Similarity values should be less than 1.
                Current Max Similarity: {}
                '''.format(mx))
    if mn < 0:
        raise ValueError('''
                Weighted Euclidean Similarity values should be greater than 1.
                Current Min Similarity: {}
                '''.format(mn))
    return sim_df

def sim_cosine(df):
    '''
    input: Pandas DataFrame with row index slate options, column headers deciders
            the entries are the preferences. Entry at row i, column j is the 
            complement ordinal preference ranking of decider j of slate option i
    output: sim_df, Pandas DataFrame with row index deciders, column headers deciders
            the entries are the similarities. Entry at row i, column j is the 
            similarity of decider j of decider i
    '''
    sim_df = pd.DataFrame(data=-1,index=df.columns, columns=df.columns, dtype=float)
    for i in df:
        for j in df:
            if sim_df[i][j] == -1:
                denom = df[i].sum() + df[j].sum()
                if denom == 0:
                    s = 0
                else:
                    s = ((df[i] * df[j]).sum())/denom
                sim_df[i][j] = s
                sim_df[j][i] = s
    mn = sim_df.values.min()
    if mn < 0:
        raise ValueError('''
                Cosine Similarity values should be greater than or equal to 0.
                Current Min Similarity: {}
                '''.format(mn))
    return sim_df

def sim_euclidean(df):
    '''
    input: Pandas DataFrame with row index slate options, column headers deciders
            the entries are the preferences. Entry at row i, column j is the 
            complement ordinal preference ranking of decider j of slate option i
    output: sim_df, Pandas DataFrame with row index deciders, column headers deciders
            the entries are the similarities. Entry at row i, column j is the 
            similarity of decider j of decider i
    '''
    sim_df = pd.DataFrame(data=-1,index=df.columns, columns=df.columns, dtype=float)
    for i in df:
        for j in df:
            if i == j:
                sim_df[i][i] = 1
            elif sim_df[i][j] == -1:
                denom = (df[i]**2).sum() + (df[j]**2).sum()
                if denom == 0:
                    s = 0
                else:
                    s = 1 - (((df[i] - df[j])**2).sum())/denom
                sim_df[i][j] = s
                sim_df[j][i] = s
    sim_df.fillna(0, inplace=True)
    ## Error Checking
    mx = sim_df.values.max()
    mn = sim_df.values.min()
    if mx > 1:
        raise ValueError('''
                Euclidean Similarity values should be less than or equal to 1.
                Current Max Similarity: {}
                '''.format(mx))
    if mn < 0:
        raise ValueError('''
                Euclidean Similarity values should be greater than or equal to 0.
                Current Min Similarity: {}
                '''.format(mn))
    return sim_df

def main():
    if len(sys.argv) != 2:
        raise ValueError('''
                Provide the filename of preference csv.
                Example:
                        python similarity.py path/to/file.csv
                ''')

    filename = sys.argv[1] 

    print_to_screen = True
    save_to_file = True

    temp = filename.split('.')
    output_filename = temp[0] + '_similarity' + temp[1]
    pref_df = load_df(filename) 
    #sim_df = sim_weighted_euclidean(pref_df)
    #sim_df = sim_euclidean(pref_df)
    sim_df = sim_cosine(pref_df)

    if print_to_screen:
        print('-----------------------------------')
        print('Similarity')
        print('-----------------------------------')
        print(sim_df.head())
        print('-----------------------------------')

    if save_to_file:
        sim_df.to_csv(output_filename, header=True, index=True)

if __name__ == '__main__':
    main()
