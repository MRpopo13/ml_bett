import numpy as np
from sklearn.externals import joblib

from ml_bett.utils import *

# pow_eq HDA_poiss    HDA_avg HDA_avg_wht HDA_poiss_scor
# 0      1               2       3           4

## Hh_scor_pg    Hh_conc_pg  Hh_point_pg Ha_scor_pg Ha_conc_pg  Ha_point_pg
#     5             6          7           8          9          10

## Ah_scor_pg    Ah_conc_pg  Ah_point_pg Aa_scor_pg Aa_conc_pg  Aa_point_pg
#     11          12          13           14          15          16

test_dataset = np.loadtxt("resources\\testing.csv", delimiter="\t")


def test_current_data(pkl_file):
    clf_model = joblib.load(pkl_file)

    print(clf_model.predict(test_dataset))
    print(clf_model.predict_proba(test_dataset))


def predict_data(pkl_file, features, ind_prono):
    print('For model {}'.format(pkl_file))
    clf_model = joblib.load(pkl_file)

    x_scaler = joblib.load('output/feature_scaler.scl')

    x_trans = x_scaler.transform(features)
    predict_result = clf_model.predict(x_trans)
    proba_pred = clf_model.predict_proba(x_trans)
    datas = [[features[i][ind_prono], predict_result[i], proba_pred[i], i] for i in range(len(features)) if
             predict_result[i] == 1 and
             max(proba_pred[i]) * 100 > 52]

    with open('resources/match.txt') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    for d in datas:
        print('For match {} prono is {} I think this prono is {} and proba is {:3.2f}'.format(content[d[3]],
                                                                                              format_prono([0]),
                                                                                              correct_incorrec(d[1]),
                                                                                              max(d[2]) * 100))
    # correct = [d for d in datas if d[0] == d[1]]
    # incorrect = [d for d in datas if d[0] != d[1]]
    #
    # print("Precision is {:3.2f}. Relevance is {:3.2f}.".format(len(correct) / len(datas) * 100,
    #                                                            len(datas) / len(features) * 100))


def correct_incorrec(number):
    if number == 0:
        return 'False'
    if number == 1:
        return 'True'


def format_prono(prono):
    if prono == 0:
        return 'HOME'
    if prono == 1:
        return 'DRAW'
    return 'AWAY'


def launch_all_models():
    models = ['neural_all_cor_60.21',
              'neural_all_avg_64.02',
              'neural_all_avgW_65.50',
              'neural_all_poiss_62.73']

    ind_prono = [1, 2, 3, 4]
    # test_models_combined(models, ind_prono, past_features[:, :17], past_features[:, 17])

    #
    for i in range(len(models)):
        predict_data(get_model_pkl_file(models[i]), test_dataset[:, :17], 1 + i)


if __name__ == '__main__':
    launch_all_models()
    # test_current_data(get_model_pkl_file('model'))
