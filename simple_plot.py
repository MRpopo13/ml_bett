import pylab


##13.45	0.0	            0.0	    1.0	        0.0
# pow_eq HDA_poiss    HDA_avg HDA_avg_wht HDA_poiss_scor
# 0      1               2       3           4

## Hh_scor_pg    Hh_conc_pg  Hh_point_pg Ha_scor_pg Ha_conc_pg  Ha_point_pg
#     5             6          7           8          9          10

## Ah_scor_pg    Ah_conc_pg  Ah_point_pg Aa_scor_pg Aa_conc_pg  Aa_point_pg
#     11          12          13           14          15          16


#HDA_res  diff_goals  h_goal  a_goal  ov_under   btts   goals
    #17      18          19      20      21      22      23

## 0	                1	        0               1   	            1	                 1
##correct-result    correct-OU      correct-BTTS  correct-avgPow    correct-avgWeightPow    correct-poissScore
## 24                   25              26              27                  28                  29

def plot_simple():
    data = pylab.loadtxt('resources/plot.csv')

    # pylab.plot(data[:,0], data[:,13], 'rx', label='predicted!value')
    pylab.plot(data[:, 0], data[:, 18], 'bo', label='actual!value')

    pylab.legend()
    pylab.title("Title of Plot")
    pylab.xlabel("Power Equilibrium")
    pylab.ylabel("Is correct prediction")

    pylab.show()


# -10.34	2	1.0	2.0	0.0
# equ    new_pred old_pred   res diff
def plot_analys_result():
    data = pylab.loadtxt('output/graph_plots/datas.csv')

    # pylab.plot(data[:,0], data[:,13], 'rx', label='predicted!value')
    pylab.plot(data[:, 2], data[:, 0], 'rx', label='actual_value')

    pylab.legend()
    pylab.title("Title of Plot")
    pylab.ylabel("Power Equilibrium")
    pylab.xlabel("Is correct prediction")

    pylab.show()


if __name__ == '__main__':
    plot_analys_result()
