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
        self.input_df = pd.DataFrame(np.nan, columns=self.seekers, 
                index=self.jobs)
        self.input_df[self.seekers[0]] = [i for i in range(1,num_jobs + 1)]
        self.input_df[self.seekers[1]] = [(num_jobs - i) for i in range(0,num_jobs)]
        self.input_df.at[self.jobs[1],self.seekers[2]] = 1
        for i in range(3,num_seekers):
            self.input_df.at[self.jobs[0],self.seekers[i]] = 1
        self.small_iop = iop(self.input_df)

    def test_input_checking_empty_df(self):
        with self.assertRaises(ValueError) as cm:
            iop(self.empty_df)

    def test_input_checking_negative_pref(self):
        neg_df = self.input_df.copy(deep=True)
        neg_df.at['J_1','S_1'] = - 1 
        with self.assertRaises(ValueError) as cm:
            iop(neg_df)

    def test_input_checking_too_large_pref(self):
        bigg_df = self.input_df.copy(deep=True)
        m,n = bigg_df.shape
        bigg_df.at['J_1','S_1'] = m+1 
        with self.assertRaises(ValueError) as cm:
            iop(bigg_df)

    def test_input_checking_skip_prefs(self):
        skip_df = self.empty_df.copy(deep=True)
        skip_df.at[self.jobs[0],'S_3'] = 1 
        skip_df.at[self.jobs[1],'S_3'] = 3 
        with self.assertRaises(ValueError) as cm:
            iop(skip_df)

    def test_input_checking_empty_columns(self):
        partial_df = self.empty_df.copy(deep=True)
        partial_df.at['J_1','S_1'] = 1 
        with self.assertRaises(ValueError) as cm:
            iop(partial_df)

    def test_input_checking_repeated_prefs(self):
        partial_df = self.input_df.copy(deep=True)
        partial_df.at['J_1','S_3'] = 1 
        partial_df.at['J_2','S_3'] = 1 
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
        for i in filter_df.index:
            for j in filter_df.columns:
                self.assertEqual(filter_df.at[i,j],iop_e.filter_df.at[i,j])
        
    def test_pref_coverage(self):
        test_df = self.empty_df.copy(deep=True)
        filter_df = self.empty_df.copy(deep=True)
        filter_df.fillna(0, inplace=True)
        for seeker in self.seekers:
            test_df.at[self.jobs[0],seeker] = 1
        iop_e = iop(test_df)
        self.assertEqual(iop_e.pref_coverage, float(1)/float(len(self.jobs)))

    def test_cosine_similarity(self):
        entries = np.array([
            [1,float(1)/float(5),float(1)/np.sqrt(5),float(2)/np.sqrt(5),float(2)/np.sqrt(5)],
            [1,1,float(1)/np.sqrt(5),0,0],
            [1,1,1,0,0],
            [1,1,1,1,1],
            [1,1,1,1,1]])
        sim_df = pd.DataFrame((entries * entries.T),index=self.seekers,columns=self.seekers)
        iop_sim_df = iop.sim_cosine(self.small_iop.pref_df_c)
        for i in iop_sim_df.index:
            for j in iop_sim_df.columns:
                self.assertEqual(round(iop_sim_df.at[i,j],4),round(sim_df.at[i,j],4))

    def test_euclidean_similarity(self):
        entries = np.array([
            [0,np.sqrt(8)/float(5),float(1)/float(2),float(1)/(2*np.sqrt(5)),float(1)/(2*np.sqrt(5))],
            [1,0,float(1)/float(2),float(3)/(2*np.sqrt(5)),float(3)/(2*np.sqrt(5))],
            [1,1,0,np.sqrt(2)/float(2),np.sqrt(2)/float(2)],
            [1,1,1,0,0],
            [1,1,1,1,0]])
        sim_df = pd.DataFrame(1 - (entries * entries.T),index=self.seekers,columns=self.seekers)
        iop_sim_df = iop.sim_euclidean(self.small_iop.pref_df_c)
        for i in iop_sim_df.index:
            for j in iop_sim_df.columns:
                self.assertEqual(round(iop_sim_df.at[i,j],4),round(sim_df.at[i,j],4))

    def test_weighted_euclidean_similarity(self):
        entries = np.array([
            [0,16,11,1,1],
            [1,0,11,17,17],
            [1,1,0,16,16],
            [1,1,1,0,0],
            [1,1,1,1,0]],dtype=float)
        sim_df = pd.DataFrame(1 - (0.5/(len(self.jobs)**4))*(entries * entries.T),
                index=self.seekers,columns=self.seekers)
        iop_sim_df = iop.sim_weighted_euclidean(self.small_iop.pref_df_c)
        for i in iop_sim_df.index:
            for j in iop_sim_df.columns:
                self.assertEqual(iop_sim_df.at[i,j],sim_df.at[i,j])

    def test_score(self):
        iop_sim_df = self.small_iop.similarity(measure='cosine')
        iop_score_df = self.small_iop.score(iop_sim_df)
        self.assertEqual(round(iop_score_df.at['J_1','S_3'],4), round(2 + float(2)/np.sqrt(5),4))
        self.assertEqual(round(iop_score_df.at['J_3','S_3'],4), round(2 + float(2)/np.sqrt(5),4))
        self.assertEqual(round(iop_score_df.at['J_2','S_4'],4), round(2 + float(2)/np.sqrt(5),4))
        self.assertEqual(round(iop_score_df.at['J_2','S_5'],4), round(2 + float(2)/np.sqrt(5),4))
        self.assertTrue(iop_score_df.at['J_3','S_4'] > -0.01)
        self.assertTrue(iop_score_df.at['J_3','S_5'] > -0.01)
        self.assertTrue(iop_score_df.at['J_3','S_4'] < 1.01)
        self.assertTrue(iop_score_df.at['J_3','S_5'] < 1.01)

    def test_implied_prefs(self):
        iop_sim_df = self.small_iop.similarity(measure='cosine')
        iop_score_df = self.small_iop.score(iop_sim_df)
        implied_df = self.small_iop.implied_prefs(iop_score_df)
        self.assertEqual(implied_df.at['J_1','S_3'],3)
        self.assertEqual(implied_df.at['J_3','S_3'],2)
        self.assertEqual(implied_df.at['J_2','S_4'],2)
        self.assertEqual(implied_df.at['J_3','S_4'],3)
        self.assertEqual(implied_df.at['J_2','S_5'],2)
        self.assertEqual(implied_df.at['J_3','S_5'],3)

    def test_complte_prefs(self):
        implied_df = self.small_iop.complete_prefs(measure='cosine')
        self.assertEqual(implied_df.at['J_1','S_3'],3)
        self.assertEqual(implied_df.at['J_3','S_3'],2)
        self.assertEqual(implied_df.at['J_2','S_4'],2)
        self.assertEqual(implied_df.at['J_3','S_4'],3)
        self.assertEqual(implied_df.at['J_2','S_5'],2)
        self.assertEqual(implied_df.at['J_3','S_5'],3)

    def test_dropout(self):
        test_iop = iop(self.small_iop.dropout(perc=0.50))
        self.assertTrue(test_iop.pref_coverage <= 0.5)

    def test_dropout_check_perc_lower(self):
        with self.assertRaises(ValueError) as cm:
            self.small_iop.dropout(perc=0.1)

    def test_dropout_check_perc_higher(self):
        with self.assertRaises(ValueError) as cm:
            self.small_iop.dropout(perc=0.9)

    def test_destructor(self):
        test_iop = iop(self.input_df)
        self.assertTrue(test_iop.pref_coverage > 0)
        del test_iop
        with self.assertRaises(UnboundLocalError) as cm:
            test_iop.pref_coverage

if __name__ == '__main__':
    unittest.main()
