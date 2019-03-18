import numpy as np
import pandas as pd
import unittest

from iop import iop

class Test_iop_class(unittest.TestCase):

    def setUp(self):
        num_seekers = 5
        num_jobs= 3
        self.seekers = ['S_{}'.format(i) for i in range(1,num_seekers + 1)]
        self.jobs = ['J_{}'.format(i) for i in range(1,num_jobs + 1)]
        self.empty_df = pd.DataFrame(np.nan, columns=self.seekers, 
                index=self.jobs)
        input_df = pd.DataFrame(np.nan, columns=self.seekers, 
                index=self.jobs)
        input_df[self.seekers[0]] = [i for i in range(1,num_jobs + 1)]
        input_df[self.seekers[1]] = [(num_jobs - i) for i in range(0,num_jobs)]
        for i in range(2,num_seekers):
            input_df.at[self.jobs[0],self.seekers[i]] = 1
        self.small_iop = iop(input_df)

    def test_input_checking_empty_df(self):
        with self.assertRaises(ValueError) as cm:
            iop(self.empty_df)

    def test_input_checking_negative_pref(self):
        neg_df = self.empty_df.copy(deep=True)
        neg_df.at['J_1','S_1'] = - 1 
        with self.assertRaises(ValueError) as cm:
            iop(neg_df)

    def test_input_checking_too_large_pref(self):
        bigg_df = self.empty_df.copy(deep=True)
        m,n = bigg_df.shape
        bigg_df.at['J_1','S_1'] = m+1 
        with self.assertRaises(ValueError) as cm:
            iop(bigg_df)

    def test_input_checking_skip_prefs(self):
        skip_df = self.empty_df.copy(deep=True)
        skip_df.at[self.jobs[0],'S_1'] = 1 
        skip_df.at[self.jobs[1],'S_1'] = 3 
        with self.assertRaises(ValueError) as cm:
            iop(skip_df)

    def test_input_checking_empty_columns(self):
        partial_df = self.empty_df.copy(deep=True)
        partial_df.at['J_1','S_1'] = 1 
        with self.assertRaises(ValueError) as cm:
            iop(partial_df)

    def test_filter_df(self):
        test_df = self.empty_df.copy(deep=True)
        filter_df = self.empty_df.copy(deep=True)
        filter_df.fillna(0, inplace=True)
        for seeker in self.seekers:
            for i in [1,2]:
                test_df.at[self.jobs[i],seeker] = i
                filter_df.at[self.jobs[i],seeker] = 1
        iop_e = iop(test_df)
        self.assertEqual(iop_e.filter_df.sum().sum(), filter_df.sum().sum())
        
    #TODO: Test Check Input: Multiple same pref: 1,1,2,2
    #TODO: Test Preference Coverage
    #TODO: Test Weighted Euclidean Similarity
    #TODO: Test Euclidean Similarity
    #TODO: Test Cosine Similarity
    #TODO: Test Score
    #TODO: Test Implied Prefs from Score
    #TODO: Test Destructor

if __name__ == '__main__':
    unittest.main()
