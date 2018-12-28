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


if __name__ == '__main__':
    test_current_data(get_model_pkl_file('model'))
