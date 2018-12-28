import numpy as np
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.externals import joblib

from ml_bett.utils import *

## 3.04	3.29	2.3	1.92	1.85	4.0	0.0	0.0	0.0	0.0	2	1	0.0	1	2	1	0	3
## // odd_h odds_d odds_a odds_o odds_u pow_eq HDA_poiss    HDA_avg HDA_avg_wht HDA_poiss_scor  mean_h_goal mean_a_goals
##     1     2      3      4       5       6         7         8       9           10              11          12
## h_conc_pg h_scor_pg h_point_pg a_conc_pg a_scor_pg a_point_pg HDA_res  diff_goals  h_goal  a_goal  ov_under    goals
#     #13          14          15      16      17          18          19      20          21      22      23          24
## 0	                1	        0
##correct-result    correct-OU      correct-BTTS
## 25                   26              27

name_file = "resources\\features3.csv"
dataset = np.loadtxt(name_file, delimiter="\t")
odds = dataset[:, 10:18]
label_train = dataset[:, 18]


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
    print_repartition_results(old_pred)
    print_repartition_results(result)

    correct = [(old_pred[i], result[i]) for i in range(size) if model_pred[i] == 1]
    incorrect = [(old_pred[i], result[i]) for i in range(size) if model_pred[i] == 0]
    print_repartition_results([cor[0] for cor in correct])
    print_repartition_results([cor[0] for cor in incorrect])

    print('size correct {}'.format(len(correct)/size*100))
    print('size incorrect {}'.format(len(incorrect)/size*100))

    print(sum([1 if cor[0] == cor[1] else 0 for cor in correct]) / len(correct) * 100)
    print(sum([1 if cor[0] != cor[1] else 0 for cor in incorrect]) / len(incorrect) * 100)


def evaluate_prediction_correct_model(pkl_file, features, labels):
    clf_model, x_test, y_test, x_scaler = common_eval(features, labels, pkl_file)

    acc = clf_model.score(x_test, y_test[:, -1])
    print('Accuracy of model {} is {:3.2f}%'.format(pkl_file, acc * 100))

    model_predict = clf_model.predict(x_test)
    x_test = x_scaler.inverse_transform(x_test)
    new_pred = get_prediction_for_cor(model_predict, x_test[:, 1])
    print('Accuracy of model done is {:3.2f}%'.format(
        sum([1 for i in range(len(new_pred)) if new_pred[i] == y_test[:, 0][i]]) / len(new_pred) * 100))
    eval_distance_real_repartition(new_pred, y_test[:, 0])
    # analyze_past_pred(x_test[:, 1], model_predict, y_test[:, 0])


if __name__ == '__main__':
    evaluate_prediction_correct_model(get_model_pkl_file('neural_all_cor_61.17'), dataset[:, 5:18], dataset[:, 18:25])
