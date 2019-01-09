import numpy as np
from sklearn.externals import joblib
from ml_bett.utils import *
# pow_eq HDA_poiss    HDA_avg HDA_avg_wht HDA_poiss_scor
# 0      1               2       3           4

## Hh_scor_pg    Hh_conc_pg  Hh_point_pg Ha_scor_pg Ha_conc_pg  Ha_point_pg
#     5             6          7           8          9          10

## Ah_scor_pg    Ah_conc_pg  Ah_point_pg Aa_scor_pg Aa_conc_pg  Aa_point_pg
#     11          12          13           14          15          16

past_features = np.loadtxt("resources\\past_features.csv", delimiter="\t")


# HDA_res  diff_goals  h_goal  a_goal  ov_under   btts   goals
# 17      18          19      20      21      22      23

## 0	                1	        0               1   	            1	                 1
##correct-result    correct-OU      correct-BTTS  correct-avgPow    correct-avgWeightPow    correct-poissScore
## 24                   25              26              27                  28                  29

# past_results = np.loadtxt("resources\\past_results.csv", delimiter="\t")
#
# results_test = past_results[:, 0]
#
# odds = past_features[:, 5:13]


def test_past_data(pkl_file):
    clf_model = joblib.load(pkl_file)

    # predict_result = clf_model.predict(odds)

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
    # comp = [1 if predict_result[i] == results_test[i] else 0 for i in range(len(predict_result))]
    #
    # print("past accuracy is {:.3f}%".format(sum(comp) / len(comp) * 100))
    #
    # print_repartition_results(predict_result)


def test_data_for_indirect_model(pkl_file, features, ind_prono, results):
    print('For model {}'.format(pkl_file))
    clf_model = joblib.load(pkl_file)

    x_scaler = joblib.load('output/feature_scaler.scl')

    x_trans = x_scaler.transform(features)
    predict_result = clf_model.predict(x_trans)
    proba_pred = clf_model.predict_proba(x_trans)
    datas = [[features[i][ind_prono], results[i]] for i in range(len(features)) if predict_result[i] == 1
             and max(proba_pred[i]) * 100 > 60]

    correct = [d for d in datas if d[0] == d[1]]
    incorrect = [d for d in datas if d[0] != d[1]]

    print("Precision is {:3.2f}. Relevance is {:3.2f}.".format(len(correct) / len(datas) * 100,
                                                               len(datas) / len(features) * 100))


def test_models_combined(models, ind_prono, features, results):
    clf_models = [joblib.load(get_model_pkl_file(pkl)) for pkl in models]
    x_scaler = joblib.load('output/feature_scaler.scl')

    x_trans = x_scaler.transform(features)

    predicts = [[clf.predict(x_trans), clf.predict_proba(x_trans)] for clf in clf_models]

    pronos = []
    for i in range(len(features)):
        max_prob = 0
        datas = [[pred[0][i], max(pred[1][i])] for pred in predicts]
        index = 0
        for ind in range(4):
            prob = datas[ind][1] * 100
            if prob > max_prob:
                max_prob = prob
                index = ind
        if max_prob > 60 and datas[index][0] == 1:
            pronos.append([features[i][ind_prono[index]], results[i], index])

    correct = [d for d in pronos if d[0] == d[1]]
    incorrect = [d for d in pronos if d[0] != d[1]]

    print("Precision is {:3.2f}. Relevance is {:3.2f}.".format(len(correct) / len(pronos) * 100,
                                                               len(pronos) / len(features) * 100))


def launch_all_models():
    models = ['neural_all_cor_60.14',
              'neural_all_avg_63.99',
              'neural_all_avgW_65.77',
              'neural_all_poiss_62.55']

    ind_prono = [1, 2, 3, 4]
    test_models_combined(models, ind_prono, past_features[:, :17], past_features[:, 17])

    #
    for i in range(len(models)):
        test_data_for_indirect_model(get_model_pkl_file(models[i]), past_features[:, :17], 1 + i,
                                     past_features[:, 17])


if __name__ == '__main__':
    # test_past_data(get_model_pkl_file('neural_all_cor_60.14'))
    # test_data_for_indirect_model(get_model_pkl_file('neural_all_cor_60.14'), past_features[:, :17], 1,
    #                              past_features[:, 17])
    launch_all_models()
