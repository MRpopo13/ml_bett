from sklearn.externals import joblib
import numpy as np
from ml_bett.utils import *

# pow_eq HDA_poiss    HDA_avg HDA_avg_wht HDA_poiss_scor
# 0      1               2       3           4

## Hh_scor_pg    Hh_conc_pg  Hh_point_pg Ha_scor_pg Ha_conc_pg  Ha_point_pg
#     5             6          7           8          9          10

## Ah_scor_pg    Ah_conc_pg  Ah_point_pg Aa_scor_pg Aa_conc_pg  Aa_point_pg
#     11          12          13           14          15          16

past_features = np.loadtxt("resources\\past_features.csv", delimiter="\t")



#HDA_res  diff_goals  h_goal  a_goal  ov_under   btts   goals
    #17      18          19      20      21      22      23

## 0	                1	        0               1   	            1	                 1
##correct-result    correct-OU      correct-BTTS  correct-avgPow    correct-avgWeightPow    correct-poissScore
## 24                   25              26              27                  28                  29

past_results = np.loadtxt("resources\\past_results.csv", delimiter="\t")


results_test = past_results[:, 0]

odds = past_features[:,5:13]

def test_past_data(pkl_file):
    clf_model = joblib.load(pkl_file)

    predict_result = clf_model.predict(odds)


    # print([sum([1 if x == 0 else 0 for x in predict_result]) / len(predict_result) * 100,
    #        sum([1 if x == 1 else 0 for x in predict_result]) / len(predict_result) * 100])
    #
    # comp_res = [odds[i][1] if predict_result[i] == 1 else odds[i][1] for i in
    #             range(len(odds))]
    # # predict_proba_result = clf_model.predict_proba(past_test)
    # predict_result = [(comp_res[i], results_test[i]) for i in range(len(comp_res)) if
    #                   # predict_proba_result[i][1] * 100 > 85]
    #                   predict_result[i] == 1]
    #
    # print("Percentage of correct prediction {:.3f}%".format(len(predict_result) / len(odds) * 100))

    # comp = [1 if data[0] == data[1] else 0 for data in predict_result]
    comp = [1 if predict_result[i] == results_test[i] else 0 for i in range(len(predict_result))]

    print("past accuracy is {:.3f}%".format(sum(comp) / len(comp) * 100))

    print_repartition_results(predict_result)






if __name__ == '__main__':
    test_past_data(get_model_pkl_file('neural'))
