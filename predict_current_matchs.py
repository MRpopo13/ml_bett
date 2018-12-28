import numpy as np
from sklearn.externals import joblib

from ml_bett.utils import *

# pow_eq HDA_poiss    HDA_avg HDA_avg_wht HDA_poiss_scor  mean_h_goal mean_a_goals   h_conc_pg h_scor_pg h_point_pg a_conc_pg a_scor_pg a_point_pg
# -1.0	    0.0	        0.0	       0.0	        0.0	            2	        1      	    1.0	        1.0	    1.5     	1.83	1.5     	1.0
test_dataset = np.loadtxt("resources\\testing.csv", delimiter="\t")


def test_current_data(pkl_file):
    clf_model = joblib.load(pkl_file)

    print(clf_model.predict(test_dataset))
    print(clf_model.predict_proba(test_dataset))


if __name__ == '__main__':
    test_current_data(get_model_pkl_file('model'))
