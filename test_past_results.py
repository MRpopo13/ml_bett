from sklearn.externals import joblib
import numpy as np
from ml_bett.utils import *

# pow_eq HDA_poiss    HDA_avg HDA_avg_wht HDA_poiss_scor  mean_h_goal mean_a_goals   h_conc_pg h_scor_pg h_point_pg a_conc_pg a_scor_pg a_point_pg
# -1.0	    0.0	        0.0	       0.0	        0.0	            2	        1      	    1.0	        1.0	    1.5     	1.83	1.5     	1.0
# 1         2           3           4           5               6       7               8           9       10          11          12      13
past_features = np.loadtxt("resources\\past_features.csv", delimiter="\t")

# HDA_res  diff_goals  h_goal  a_goal  ov_under    goals
#   0        1          2      3         4          5
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
