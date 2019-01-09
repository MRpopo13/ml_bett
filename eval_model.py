from statistics import *

import numpy as np
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.externals import joblib

from ml_bett.utils import *

## 3.04	3.29	2.3	1.92	1.85	4.0	0.0	0.0	0.0	0.0	2	1	0.0	1	2	1	0	3
## // odd_h odds_d odds_a odds_o odds_u pow_eq HDA_poiss    HDA_avg HDA_avg_wht HDA_poiss_scor
##     1     2      3      4       5       6         7         8       9           10

## Hh_scor_pg    Hh_conc_pg  Hh_point_pg Ha_scor_pg Ha_conc_pg  Ha_point_pg
#     11          12          13           14          15          16

## Ah_scor_pg    Ah_conc_pg  Ah_point_pg Aa_scor_pg Aa_conc_pg  Aa_point_pg
#     17          18          19           20          21          22


# HDA_res  diff_goals  h_goal  a_goal  ov_under   btts   goals
# 23      24          25      26      27      28      29

## 0	                1	        0               1   	            1	                 1
##correct-result    correct-OU      correct-BTTS  correct-avgPow    correct-avgWeightPow    correct-poissScore
## 30                   31              32              33                  34                  35

name_file = "resources\\features3.csv"
dataset = np.loadtxt(name_file, delimiter="\t")

labels_list = [(22, 'fin'), (26, 'OU'), (27, 'BTTS'), (30, 'corOU'), (31, 'corBTTS'),
               (29, 'cor'), (32, 'avg'), (33, 'avgW'),
               (34, 'poiss')]

features_list = [([5, 22], 'all'), ([5, 10], 'preds'), ([10, 22], 'stats'), ([0, 22], 'odds+'), ([0, 5], 'odds')]


def eval_distance_real_repartition(pred_arr, real_arr):
    real_repartition = repartition_results(real_arr)
    pred_repartition = repartition_results(pred_arr)
    print([pred_repartition[i] - real_repartition[i] for i in range(3)])
    print_repartition_results(pred_arr)
    print_repartition_results(real_arr)


def common_eval(features, labels, pkl_file):
    clf_model = joblib.load(pkl_file)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.3,
                                                                        random_state=0)
    x_scaler = preprocessing.StandardScaler()
    x_scaler.fit(x_train)
    x_test = x_scaler.transform(x_test)
    return clf_model, x_test, y_test, x_scaler


def evaluate_final_result_model(pkl_file, features, labels):
    clf_model, x_test, y_test, x_scaler = common_eval(features, labels, pkl_file)

    acc = clf_model.score(x_test, y_test)
    print('Accuracy of model {} is {:3.2f}%'.format(pkl_file, acc * 100))
    eval_distance_real_repartition(clf_model.predict(x_test), y_test)


def compare_pred_and_results(old_pred, pred, result):
    diff = [pred[i] - result[i] for i in range(len(pred))]
    diff_old = [old_pred[i] - result[i] for i in range(len(pred))]
    print_repartition_results([pred[i] for i in range(len(pred)) if diff[i] == 0])
    print_repartition_results([pred[i] for i in range(len(pred)) if diff[i] != 0])
    print_repartition_results([old_pred[i] for i in range(len(pred)) if diff_old[i] == 0])
    print_repartition_results([old_pred[i] for i in range(len(pred)) if diff_old[i] != 0])


def make_graph(full_data, old_pred, pred, result):
    diff = [pred[i] - result[i] for i in range(len(pred))]
    with open('output/graph_plots/datas.csv', 'w+') as out:
        for i in range(len(full_data)):
            out.write('{:3.2f}\t{}\t{}\t{}\t{}\n'.format(full_data[i], pred[i], old_pred[i], result[i], diff[i]))


def analyze_past_pred(old_pred, model_pred, result):
    size = len(model_pred)
    print("repartition inital prono")
    print_repartition_results(old_pred)
    print("repartition actual result")
    print_repartition_results(result)

    correct = [(old_pred[i], result[i]) for i in range(size) if model_pred[i] == 1]
    incorrect = [(old_pred[i], result[i]) for i in range(size) if model_pred[i] == 0]
    print("repartition correct prono")
    print_repartition_results([cor[0] for cor in correct])
    print("repartition incorrect prono")
    print_repartition_results([cor[0] for cor in incorrect])

    print('size correct {}'.format(len(correct) / size * 100))
    print('size incorrect {}'.format(len(incorrect) / size * 100))

    print("Percentage of correct prono for predicted correct {:3.2f}".
          format(sum([1 if cor[0] == cor[1] else 0 for cor in correct]) / len(correct) * 100))
    print("Percentage of incorrect prono for predicted incorrect {:3.2f}".
          format(sum([1 if cor[0] != cor[1] else 0 for cor in incorrect]) / len(incorrect) * 100))


def evaluate_prediction_correct_model(pkl_file, features, labels, ind_res, ind_prono):
    clf_model, x_test, y_test, x_scaler = common_eval(features, labels, pkl_file)

    acc = clf_model.score(x_test, y_test[:, ind_res])
    print('Accuracy of model {} is {:3.2f}%'.format(pkl_file, acc * 100))

    model_predict = clf_model.predict(x_test)
    proba = clf_model.predict_proba(x_test)

    x_test = x_scaler.inverse_transform(x_test)
    filt_data = [[x_test[i][ind_prono], model_predict[i], y_test[i][0]] for i in range(len(x_test)) if
                 max(proba[i]) * 100 > 80]
    print('Percentage interesting is {:3.2f}%'.format(len(filt_data) / len(x_test) * 100))
    # new_pred = get_prediction_for_cor(model_predict, x_test[:, ind_prono], 59, 69)
    #
    # print('Accuracy of model refined is {:3.2f}%'.format(
    #     sum([1 for i in range(len(new_pred)) if new_pred[i] == y_test[:, 0][i]]) / len(new_pred) * 100))
    # eval_distance_real_repartition(new_pred, y_test[:, 0])
    # analyze_past_pred(x_test[:, ind_prono], model_predict, y_test[:, 0])
    analyze_past_pred([f[0] for f in filt_data], [f[1] for f in filt_data], [f[2] for f in filt_data])


def analyze_bett(arr):
    # correct = [[old_pred[i], result[i], bett[i][int(old_pred[i])], sum(1 / bett[i])] for i in range(size) if
    #            model_predict[i] == 1]
    size = len(arr)
    avg_odd = sum([a[2] for a in arr]) / size
    avg_equi = sum([a[4] for a in arr]) / size
    med_odd = median([a[2] for a in arr])
    med_equi = median([a[4] for a in arr])
    odd_80 = np.percentile([a[2] for a in arr], 95)
    equi_80 = np.percentile([a[4] for a in arr], 80)
    print_repartition_results([a[0] for a in arr])
    print_repartition_results([a[1] for a in arr])
    prob = np.percentile([a[5] for a in arr], 50)
    print('Average odd {:3.2f} \nAverage equi {:3.2f} \nMed odd {:3.2f} \nMed equi {:3.2f} '
          '\np80 odd {:3.2f} \nP80 equi {:3.2f} \nProba median {:3.2f} \n'.format(
        avg_odd, avg_equi,
        med_odd,
        med_equi, odd_80, equi_80, prob))


def analyze_pred(arr):
    # [old_pred[i], result[i], abs(equi[i]), max(proba[i])
    size = len(arr)
    avg_equi = sum([a[2] for a in arr]) / size
    med_equi = median([a[2] for a in arr])
    equi_80 = np.percentile([a[2] for a in arr], 80)
    print_repartition_results([a[0] for a in arr])
    print_repartition_results([a[1] for a in arr])
    prob = np.percentile([a[3] for a in arr], 80)
    print('Average equi {:3.2f}\nMed equi {:3.2f}'
          '\nP80 equi {:3.2f} \nProba median {:3.2f} \n'.format(
        avg_equi, med_equi, equi_80, prob))


def evaluate_bett_gain_for_model(pkl_file, features, labels, ind_res, ind_prono):
    clf_model, x_test, y_test, x_scaler = common_eval(features, labels, pkl_file)

    acc = clf_model.score(x_test, y_test[:, ind_res])
    print('Accuracy of model {} is {:3.2f}%'.format(pkl_file, acc * 100))

    model_predict = clf_model.predict(x_test)
    proba = clf_model.predict_proba(x_test)
    x_test = x_scaler.inverse_transform(x_test)
    size = len(x_test)

    old_pred = x_test[:, ind_prono]
    result = y_test[:, 0]
    bett = x_test[:, :3]
    equi = x_test[:, 5]

    pred_corre = [[old_pred[i], result[i], bett[i][int(old_pred[i])], sum(1 / bett[i]), abs(equi[i]), max(proba[i])] for
                  i in range(size)
                  if
                  model_predict[i] == 1 and max(proba[i]) > 0.7]

    gain = 2
    rapport = 1
    k = 1

    # bett = [k for corr in pred_corre]

    # correct_values = [corr for corr in pred_corre if corr[3] / corr[2] < 0.8]
    correct_values = [corr for corr in pred_corre]

    bett = [gain / (corr[2] - 1) * k if corr[3] / corr[2] > 0.74 else gain / (rapport * (corr[2] - 1)) for corr in
            correct_values]

    nb_correct = [corr for corr in correct_values if corr[0] == corr[1]]
    incorrect = [corr for corr in correct_values if corr[0] != corr[1]]
    analyze_bett(incorrect)
    analyze_bett(nb_correct)
    print('Percentage pred_corre {:3.2f}'.format(len(nb_correct) / len(correct_values) * 100))

    mapGain = [bett[i] * (correct_values[i][2] - 1) if correct_values[i][0] == correct_values[i][1] else -1 * bett[i]
               for i in
               range(len(correct_values))]
    # mapGain = [k * (pred_corre[i][2] - 1) if pred_corre[i][0] == pred_corre[i][1] else -1 * k
    #            for i in
    #            range(len(pred_corre))]
    gain = sum(mapGain)
    print('Your gain is {:3.2f} for {:3.2f} betts \n ROI is {:3.2f}%'.format(gain, len(correct_values),
                                                                             gain / (sum(bett)) * 100))


def find_prob_eff_model(pkl_file, features, labels, ind_res, ind_prono):
    clf_model, x_test, y_test, x_scaler = common_eval(features, labels, pkl_file)

    acc = clf_model.score(x_test, y_test[:, ind_res])
    print('Accuracy of model {} is {:3.2f}%'.format(pkl_file, acc * 100))

    model_predict = clf_model.predict(x_test)
    proba = clf_model.predict_proba(x_test)
    x_test = x_scaler.inverse_transform(x_test)
    size = len(x_test)

    old_pred = x_test[:, ind_prono]
    result = y_test[:, 0]
    equi = x_test[:, 5]

    pred_corre = [[old_pred[i], result[i], abs(equi[i]), max(proba[i])] for
                  i in range(size)
                  if model_predict[i] == 1
                  and max(proba[i]) > 0.7
                  ]

    correct_values = [corr for corr in pred_corre]
    nb_correct = [corr for corr in correct_values if corr[0] == corr[1]]
    incorrect = [corr for corr in correct_values if corr[0] != corr[1]]
    analyze_pred(nb_correct)
    analyze_pred(incorrect)
    print('{:3.2f}% correct//{:3.2f}% incorrect'.format(len(nb_correct) / len(correct_values),
                                                        len(incorrect) / len(correct_values)))


if __name__ == '__main__':
    # evaluate_prediction_correct_model(get_model_pkl_file('neural_odds+_cor_64.77'), dataset[:, :22], dataset[:, 22:34],
    #                                   7, 6)
    # evaluate_bett_gain_for_model(get_model_pkl_file('neural_odds+_cor_64.79'), dataset[:, :22],
    #                              dataset[:, 22:34],
    #                              7, 6)
    find_prob_eff_model(get_model_pkl_file('neural_all_cor_60.14'), dataset[:, 5:22], dataset[:, 22:34],
                        7, 1)
