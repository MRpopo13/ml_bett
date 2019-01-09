import numpy as np


## 3.04	3.29	2.3	1.92	1.85	4.0	0.0	0.0	0.0	0.0	2	1	0.0	1	2	1	0	3
## // odd_h odds_d odds_a odds_o odds_u pow_eq HDA_poiss    HDA_avg HDA_avg_wht HDA_poiss_scor  mean_h_goal mean_a_goals   h_conc_pg h_scor_pg h_point_pg a_conc_pg a_scor_pg a_point_pg HDA_res  diff_goals  h_goal  a_goal  ov_under    goals
##     1     2      3      4       5       6         7         8       9           10              11          12              13          14          15      16      17          18          19      20          21      22      23          24

def equalize_dataset(name_file):
    dataset = np.loadtxt(name_file, delimiter="\t")

    eq_dataset = [[data for data in dataset if data[18] == 0],
                  [data for data in dataset if data[18] == 1],
                  [data for data in dataset if data[18] == 2]]

    size_max = min([len(eq) for eq in eq_dataset])
    clean_dataset = [eq[:size_max] for eq in eq_dataset]
    return np.concatenate((clean_dataset[0], clean_dataset[1], clean_dataset[2]), axis=0)
