import numpy as np
# from ml_bett.equilibrate_dataset import equalize_dataset
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import svm
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

## 3.04	3.29	2.3	1.92	1.85	4.0	0.0	0.0	0.0	0.0	2	1	0.0	1	2	1	0	3
## // odd_h odds_d odds_a odds_o odds_u pow_eq HDA_poiss    HDA_avg HDA_avg_wht HDA_poiss_scor  mean_h_goal mean_a_goals   h_conc_pg h_scor_pg h_point_pg a_conc_pg a_scor_pg a_point_pg HDA_res  diff_goals  h_goal  a_goal  ov_under    goals
##     1     2      3      4       5       6         7         8       9           10              11          12              13          14          15      16      17          18          19      20          21      22      23          24

## 0	                1	        0
##correct-result    correct-OU      correct-BTTS
## 25                   26              27

name_file = "resources\\features3.csv"
dataset = np.loadtxt(name_file, delimiter="\t")
# pow_eq HDA_poiss    HDA_avg HDA_avg_wht HDA_poiss_scor  mean_h_goal mean_a_goals   h_conc_pg h_scor_pg h_point_pg a_conc_pg a_scor_pg a_point_pg
# -1.0	    0.0	        0.0	       0.0	        0.0	            2	        1      	    1.0	        1.0	    1.5     	1.83	1.5     	1.0
test_dataset = np.loadtxt("resources\\testing.csv", delimiter="\t")

past_features = np.loadtxt("resources\\past_features.csv", delimiter="\t")
past_results = np.loadtxt("resources\\past_results.csv", delimiter="\t")

# clean_data = equalize_dataset(name_file)
# odds = np.concatenate((dataset[:, 13:16], dataset[:, 17], dataset[:, 6]), axis=1)
odds = dataset[:, 5:10]
results = dataset[:, 24]
# seq = map(lambda x: 0 if x == 0 else 1, results)
# seq = map(lambda x: x - 1, results)
seq = results
results = np.fromiter(seq, dtype=np.int)

clf_logistic = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
clf_svc = svm.SVC(gamma=0.001, C=1., decision_function_shape='ovo', cache_size=1000, kernel='rbf')
clf_svr = svm.SVR()
clf_neural = MLPClassifier(solver='lbfgs', alpha=1e-3,
                           activation='logistic', random_state=1)

results_test = past_results[:, 0]


def hist_proof(clf_model):
    past_test = past_features[:, :5]
    predict_result = clf_model.predict(past_test)
    comp_res = [past_test[i][1] if predict_result[i] == 1 else past_test[i][1] for i in
                range(len(past_test))]
    # predict_proba_result = clf_model.predict_proba(past_test)
    predict_result = [(comp_res[i], results_test[i]) for i in range(len(comp_res)) if
                      # predict_proba_result[i][1] * 100 > 85]
                      predict_result[i] == 1]

    # print([sum([1 if x == 0 else 0 for x in predict_result]) / len(predict_result) * 100,
    #        sum([1 if x == 1 else 0 for x in predict_result]) / len(predict_result) * 100])

    print("Number of correct pred ", len(predict_result) / len(past_test) * 100)

    comp = [1 if data[0] == data[1] else 0 for data in predict_result]

    print("past accuracy is ", sum(comp) / len(comp) * 100)

    print([sum([1 if x[0] == 0 else 0 for x in predict_result]) / len(predict_result) * 100,
           sum([1 if x[0] == 1 else 0 for x in predict_result]) / len(predict_result) * 100,
           sum([1 if x[0] == 2 else 0 for x in predict_result]) / len(predict_result) * 100])


def evaluate_model(clf_model):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(odds, results, test_size=0.1,
                                                                        random_state=0)

    # print([sum([1 if y == 0 else 0 for y in y_train]) / len(y_train) * 100,
    #        sum([1 if y == 1 else 0 for y in y_train]) / len(y_train) * 100,
    #        sum([1 if y == 2 else 0 for y in y_train]) / len(y_train) * 100])
    #
    # print([sum([1 if y == 0 else 0 for y in y_test]) / len(y_test) * 100,
    #        sum([1 if y == 1 else 0 for y in y_test]) / len(y_test) * 100,
    #        sum([1 if y == 2 else 0 for y in y_test]) / len(y_test) * 100])

    x_scaler = preprocessing.StandardScaler()
    x_train = x_scaler.fit_transform(x_train)
    x_test = x_scaler.transform(x_test)

    clf_model.fit(x_train, y_train)

    joblib.dump(clf_model, 'output\\model.pkl')  ## save model to pkl file

    acc = clf_model.score(x_test, y_test)
    print(acc * 100)

    hist_proof(clf_model)
    # print(clf_model.predict(test_dataset))
    # print(clf_model.predict_proba(test_dataset))


if __name__ == '__main__':
    evaluate_model(clf_svc)
