# from ml_bett.equilibrate_dataset import equalize_dataset

from ml_bett.eval_model import *

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


def produce_new_features(models, features, rest_features):
    clf_models = [joblib.load(get_model_pkl_file(pkl)) for pkl in models]
    x_scaler = joblib.load('output/feature_scaler.scl')

    x_trans = x_scaler.transform(features)

    with(open('resources/next_features.csv', 'w+')) as out:
        for i in range(len(features)):
            feat = features[i].reshape(1, -1)
            trans_feat = x_scaler.transform(feat)
            preds = [str(clf.predict(trans_feat)[0]) for clf in clf_models]
            feat_str = '\t'.join([str(f) for f in features[i]])
            rest_feat_str = '\t'.join([str(f) for f in rest_features[i]])
            out.write(
                feat_str + '\t' + preds[0] + '\t' + preds[1] + '\t' + preds[2] + '\t' + preds[
                    3] + '\t' + rest_feat_str + '\n')


def launch_all_models():
    models = ['neural_all_cor_60.14',
              'neural_all_avg_63.99',
              'neural_all_avgW_65.77',
              'neural_all_poiss_62.55']
    produce_new_features(models, dataset[:, :17], dataset[:, 17:])


if __name__ == '__main__':
    launch_all_models()
