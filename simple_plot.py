import pylab
##13.45	0.0	            0.0	    1.0	        0.0	            1	        0
#pow_eq HDA_poiss    HDA_avg HDA_avg_wht HDA_poiss_scor  mean_h_goal mean_a_goals
#0      1               2       3           4               5           6

##1.33	        0.67	0.83	1.17	    0.33	0.67	    1.0	        0	        1	    1	    1	    2
#h_conc_pg h_scor_pg h_point_pg a_conc_pg a_scor_pg a_point_pg HDA_res  diff_goals  h_goal  a_goal  ov_under  goals
#7       8           9           10      11          12         13         14       15       16       17      18

##  0	                1	        0
##correct-result    correct-OU      correct-BTTS
## 19                   20              21

def plot_simple():
    data = pylab.loadtxt('resources/plot.csv')

    # pylab.plot(data[:,0], data[:,13], 'rx', label='predicted!value')
    pylab.plot(data[:,0], data[:,18], 'bo', label='actual!value')

    pylab.legend()
    pylab.title("Title of Plot")
    pylab.xlabel("Power Equilibrium")
    pylab.ylabel("Is correct prediction")

    pylab.show()


if __name__ == '__main__':
    plot_simple()
