# from ml_bett.equilibrate_dataset import equalize_dataset
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from ml_bett.eval_model import *

## 3.04	3.29	2.3	1.92	1.85	4.0	0.0	0.0	0.0	0.0	2	1	0.0	1	2	1	0	3
## // odd_h odds_d odds_a odds_o odds_u pow_eq HDA_poiss    HDA_avg HDA_avg_wht HDA_poiss_scor
##     1     2      3      4       5       6         7         8       9           10

## Hh_scor_pg    Hh_conc_pg  Hh_point_pg Ha_scor_pg Ha_conc_pg  Ha_point_pg
#     11          12          13           14          15          16

## Ah_scor_pg    Ah_conc_pg  Ah_point_pg Aa_scor_pg Aa_conc_pg  Aa_point_pg
#     17          18          19           20          21          22


#HDA_res  diff_goals  h_goal  a_goal  ov_under   btts   goals
    #23      24          25      26      27      28      29

## 0	                1	        0               1   	            1	                 1
##correct-result    correct-OU      correct-BTTS  correct-avgPow    correct-avgWeightPow    correct-poissScore
## 30                   31              32              33                  34                  35

name_file = "resources\\features3.csv"
dataset = np.loadtxt(name_file, delimiter="\t")

# clean_data = equalize_dataset(name_file)
# odds = np.concatenate((dataset[:, 13:16], dataset[:, 17], dataset[:, 6]), axis=1)
odds = dataset[:, 5:22]
results = dataset[:, 22]
# seq = map(lambda x: 0 if x == 0 else 1, results)
# seq = map(lambda x: x - 1, results)
seq = results
results = np.fromiter(seq, dtype=np.int)

clf_logistic = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
clf_svc = svm.SVC(gamma=0.001, C=1., decision_function_shape='ovo', cache_size=1000, kernel='rbf')
clf_svr = svm.SVR()
clf_neural = MLPClassifier(solver='sgd', alpha=1e-3,
                           activation='relu', random_state=1, learning_rate='adaptive',
                           learning_rate_init=1e-2, batch_size=64, tol=1e-2, max_iter=300)

map_clf_model = [(clf_logistic, 'logistic'), (clf_neural, 'neural')]
# , (clf_svc, 'svc')

features_list = [([5, 22], 'all'), ([5, 10], 'preds'), ([10, 22], 'stats'), ([0, 22], 'odds+'), ([0, 5], 'odds') ]

labels_list = [(22, 'fin'), (26, 'OU'), (27, 'BTTS'),(30, 'corOU'), (31, 'corBTTS'),
               (29, 'cor'),(32, 'avg'), (33, 'avgW'),
               (34, 'poiss')]

#
def train_model(clf_model, pkl_file, features, labels):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.3,
                                                                        random_state=0)
    x_scaler = preprocessing.StandardScaler()
    x_train = x_scaler.fit_transform(x_train)
    x_test = x_scaler.transform(x_test)

    clf_model.fit(x_train, y_train)

    acc = clf_model.score(x_test, y_test) * 100
    print('Accuracy of model {} is {:3.2f}%'.format(pkl_file, acc))

    train_accuracy = accuracy_score(y_train, clf_model.predict(x_train))
    test_accuracy = accuracy_score(y_test, clf_model.predict(x_test))
    # print_repartition_results(y_train)
    # print_repartition_results(y_test)
    # print_repartition_results(clf_model.predict(x_test))
    # eval_distance_real_repartition(clf_model.predict(x_test), y_test)

    print('Train accuracy is {:3.2f}% test accuracy is {:3.2f}%'.format(train_accuracy*100, test_accuracy*100))
    if acc > 58:
        joblib.dump(clf_model,
                    get_model_pkl_file(pkl_file + '_' + '{:3.2f}'.format(acc)))  ## save model to pkl file


def call_pipeline(map_model, map_features, map_labels):
    for m in map_model:
        model = m[0]
        for f in map_features:
            features = dataset[:, f[0][0]:f[0][1]]
            for r in map_labels:
                labels = dataset[:, r[0]]
                core_name = m[1] + '_' + f[1] + '_' + r[1]
                train_model(model, core_name, features, labels)


def launch_spec_model(model_name, feature_name, label_name):
    clf = [el[0] for el in map_clf_model if el[1] == model_name][0]
    feat_ind = [el[0] for el in features_list if el[1] == feature_name][0]
    lab_ind = [el[0] for el in labels_list if el[1] == label_name][0]
    features = dataset[:, feat_ind[0]:feat_ind[1]]
    labels = dataset[:, lab_ind]
    core_name = model_name + '_' + feature_name + '_' + label_name
    train_model(clf, core_name, features, labels)


if __name__ == '__main__':
    # call_pipeline(map_clf_model, features_list, labels_list)
    launch_spec_model('neural', 'odds+', 'cor')
