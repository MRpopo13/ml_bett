#########The feature used is the following one
######### odd_h odds_d odds_a odds_o odds_u
#########  1     2      3      4       5
#########HDA_res  diff_goals  h_goal  a_goal  ov_under   btts   goals
######### 6         7            8      9       10       11     12
#########
#########
from random import randrange

import numpy
import pandas as pd

header = ['odd_h', 'odd_d', 'odd_a', 'odd_ov', 'odd_un', 'HDA', 'diff_goals', 'h_goal', 'a_goal', 'ov_under', 'btts',
          'goals', 'favorite', 'surprise']


def add_is_surprise_result(line, begin):
    res = line[5]
    odds_result = line[:3]
    favorite = odds_result.index(min(odds_result))
    if round(float(res)) == favorite:
        return begin + "\t" + str(favorite) + "\t0"
    return begin + "\t" + str(favorite) + "\t1"


def cure_betts_data(src_file):
    with open(src_file) as src:
        with open(src_file + '.out', 'w') as out:
            lines = get_splitted_lines_from_src_file(src)
            part = [line[:5] + line[22:29] for line in lines]
            for line in part:
                begin = '\t'.join(line)
                out.writelines(add_is_surprise_result(line, begin) + '\n')


def is_barbell_bett(line):
    trunc = line[:3]
    for el in trunc:
        if float(el) < 2:
            return False
    return True


def filter_out_barbell_betts(src_file):
    with open(src_file) as src:
        lines = get_splitted_lines_from_src_file(src)

        filtered = [line for line in lines if is_barbell_bett(line)]
    with open('resources/barbell_betts.csv', 'w+') as out:
        for line in filtered:
            begin = '\t'.join(line)
            out.writelines(begin + '\n')


def repartition_result_betts(betts_file):
    with open(betts_file) as src_file:
        lists_of_list = [[], [], []]
        lines = get_splitted_lines_from_src_file(src_file)
        for line in lines:
            lists_of_list[round(float(line[5]))].append(line)
        pool_size = len(lines)
        print(f"Number of betts {pool_size}")
        print(f"Number of home_win {len(lists_of_list[0])} which is {len(lists_of_list[0]) / pool_size *100}")
        print(f"Number of draw {len(lists_of_list[1])} which is {len(lists_of_list[1]) / pool_size *100}")
        print(f"Number of away_win {len(lists_of_list[2])} which is {len(lists_of_list[2]) / pool_size *100}")


def goals_scored(betts_file):
    # with open(betts_file) as src_file:
    #     lines = get_splitted_lines_from_src_file(src_file)
    #     total_goals = lines[7:][9:]
    #     print(total_goals)
    data = pd.read_csv(betts_file, sep="\t", names=header)

    print(f"Average goals per match {data['goals'].mean():.2f} and the median is {data['goals'].median()}")
    print(f"Average home goals per match {data['h_goal'].mean():.2f} and the median is {data['h_goal'].median()}")
    print(f"Average away goals per match {data['a_goal'].mean():.2f} and the median is {data['a_goal'].median()}")
    print(f"Average over odd per match {data['odd_ov'].mean():.2f} and the median is {data['odd_ov'].median()}")
    print(f"Average under odd per match {data['odd_un'].mean():.2f} and the median is {data['odd_un'].median()}")

    nb_over = data['ov_under'].sum()
    nb_under = len(data) - nb_over
    print(f"Number and percentage of over ={nb_over} and {nb_over/len(data)*100:.2f}")
    print(f"Number and percentage of under ={nb_under} and {nb_under/len(data)*100:.2f}")
    # pd.
    # print(total_goals)


def get_avg_odd_favourite(data):
    odds = data[['odd_h', 'odd_d', 'odd_a']].values.tolist()
    prono = data['favorite'].values.tolist()
    odd_favourite = [odds[prono[i]] for i in range(len(data))]
    return numpy.median(odd_favourite)


def surprising_result(betts_file):
    data = pd.read_csv(betts_file, sep="\t", names=header)
    pool_data = len(data)

    surprise = data[data['surprise'] == 1]
    nb_sup = len(surprise)
    avg_odd_surprise = get_avg_odd_favourite(surprise)

    correct = data[data['surprise'] == 0]
    nb_correct = len(correct)
    avg_odd_correct = get_avg_odd_favourite(correct)

    home_favorite_data = data[data['favorite'] == 0]
    home_favorite_size = len(home_favorite_data)
    home_favorite_avg_odd = home_favorite_data['odd_h'].mean()

    draw_favorite = len(data[data['favorite'] == 1])

    away_favorite_data = data[data['favorite'] == 2]
    away_favorite_size = len(away_favorite_data)
    away_favorite_avg_odd = away_favorite_data['odd_a'].mean()
    print(f"The number of surprise is {nb_sup} which represents {nb_sup/pool_data*100:.2f}%"
          f"and the average odd is {avg_odd_surprise:.2f}")

    print(f"The number of correct is {nb_correct} which represents {nb_correct/pool_data*100:.2f}%"
          f"and the average odd is {avg_odd_correct:.2f}")

    print(
        f"The number of home_favorite_size is {home_favorite_size} which represents {home_favorite_size/pool_data*100:.2f}%"
        f"and the average odd is {home_favorite_avg_odd:.2f}")

    print(f"The number of draw_favorite is {draw_favorite} which represents {draw_favorite/pool_data*100:.2f}%")

    print(
        f"The number of away_favorite_size is {away_favorite_size} which represents {away_favorite_size/pool_data*100:.2f}%"
        f"and the average odd is {away_favorite_avg_odd:.2f}")

    print("\n\n\n")
    #############HOME FAVORITE
    home_favorite_draw = len(data[(data['favorite'] == 0) & (data['HDA'] == 1.0)])
    home_favorite_away = len(data[(data['favorite'] == 0) & (data['HDA'] == 2.0)])
    home_favorite_sup = len(data[(data['favorite'] == 0) & (data['HDA'] != 0.0)])
    correct_home_favourite = data[(data['favorite'] == 0) & (data['HDA'] == 0.0)]
    home_favorite_corr_size = len(correct_home_favourite)
    avg_odd_home_favorite_corr = correct_home_favourite['odd_h'].mean()

    print(
        f"The number of home_favorite_draw is {home_favorite_draw} which represents {home_favorite_draw/home_favorite_size*100:.2f}%")
    print(
        f"The number of home_favorite_away is {home_favorite_away} which represents {home_favorite_away/home_favorite_size*100:.2f}%")
    print(
        f"The number of home_favorite_surprises is {home_favorite_sup} which represents {home_favorite_sup/home_favorite_size*100:.2f}%")
    print(
        f"The number of home_favorite_correct is {home_favorite_corr_size} which represents {home_favorite_corr_size/home_favorite_size*100:.2f}%"
        f" and the average odd is {avg_odd_home_favorite_corr:.2f}")

    print("\n\n\n")
    #############away FAVORITE
    away_favorite_draw = len(data[(data['favorite'] == 2) & (data['HDA'] == 1.0)])
    away_favorite_home = len(data[(data['favorite'] == 2) & (data['HDA'] == 0.0)])
    away_favorite_sup = len(data[(data['favorite'] == 2) & (data['HDA'] != 2.0)])
    away_fav_corr_data = data[(data['favorite'] == 2) & (data['HDA'] == 2.0)]
    away_fav_corr_avg_odd = away_fav_corr_data['odd_a'].mean()
    away_favorite_corr_size = len(away_fav_corr_data)

    print(
        f"The number of away_favorite_draw is {away_favorite_draw} which represents {away_favorite_draw/away_favorite_size*100:.2f}%")
    print(
        f"The number of away_favorite_home is {away_favorite_home} which represents {away_favorite_home/away_favorite_size*100:.2f}%")
    print(
        f"The number of away_favorite_surprise is {away_favorite_sup} which represents {away_favorite_sup/away_favorite_size*100:.2f}%")
    print(
        f"The number of away_favorite_correct is {away_favorite_corr_size} which represents {away_favorite_corr_size/away_favorite_size*100:.2f}%"
        f" and the average odd is {away_fav_corr_avg_odd:.2f}")

    print("\n\n\n")


##############################
##############################
# """"""""""""""""""""""""""""""
################################


def all_simulations(src_file):
    data = pd.read_csv(src_file, sep="\t", names=header)
    # data = data[(data['odd_a'] > 2.5) & (data['odd_h'] > 2.5)]
    size = len(data)
    print(f"Number of element is {size}")
    #####ALWAYS HOME
    gains = compute_gains_always_same(data, 0.0, 'odd_h')
    sum_gains = sum(gains)
    print(f"The gain for always home is {sum_gains} which is a ROI of {sum_gains/size*100:.2f}%")

    #####ALWAYS DRAW
    gains = compute_gains_always_same(data, 1.0, 'odd_d')
    sum_gains = sum(gains)
    print(f"The gain for always draw is {sum_gains} which is a ROI of {sum_gains/size*100:.2f}%")

    #####ALWAYS AWAY
    gains = compute_gains_always_same(data, 2.0, 'odd_a')
    sum_gains = sum(gains)
    print(f"The gain for always away is {sum_gains} which is a ROI of {sum_gains/size*100:.2f}%")

    #######ALWAYS FAVOURITE
    gains = compute_gains_always_favourite(data)
    sum_gains = sum(gains)
    print(f"The gain for always favourite is {sum_gains} which is a ROI of {sum_gains/size*100:.2f}%")

    #######ALWAYS 2nd FAVOURITE
    gains = compute_gains_always_second_favourite(data)
    sum_gains = sum(gains)
    print(f"The gain for always second favourite is {sum_gains} which is a ROI of {sum_gains/size*100:.2f}%")

    #######ALWAYS UNDERDOG
    gains = compute_gains_always_underdog(data)
    sum_gains = sum(gains)
    print(f"The gain for always underdog is {sum_gains} which is a ROI of {sum_gains/size*100:.2f}%")

    #######RANDOM RESULT
    gains = compute_gains_random_result(data)
    sum_gains = sum(gains)
    print(f"The gain for random results is {sum_gains} which is a ROI of {sum_gains/size*100:.2f}%")

    #######WEIGHTED RANDOM RESULT
    gains = compute_gains_random_result_weighted(data)
    sum_gains = sum(gains)
    print(f"The gain for weighted random results is {sum_gains} which is a ROI of {sum_gains/size*100:.2f}%")

    #######ALWAYS FAVOURITE variable stakes
    [gains, stake] = compute_gains_always_favourite_variable_stake(data, 2.75, 2.7, 0)
    sum_gains = sum(gains)
    print(
        f"The gain for always favourite with variable stakes is {sum_gains} which is a ROI of {sum_gains/stake*100:.2f}%"
        f"for a total number of betts {len(gains)}")

    #######ALWAYS UNDERDOG variable stakes
    [gains, stake] = compute_gains_always_underdog_variable_stake(data, 2.3, 0.1, 1)
    sum_gains = sum(gains)
    print(
        f"The gain for always underdog with variable stakes is {sum_gains} which is a ROI of {sum_gains/stake*100:.2f}%")


def compute_roi_variable_stakes(src_file):
    data = pd.read_csv(src_file, sep="\t", names=header)
    max_roi = -10.
    max_odd, max_low, max_high = 0., 0., 0.
    #######ALWAYS FAVOURITE variable stakes
    for odd in range(200, 290, 5):
        for low in range(50):
            for high in range(50):
                [gains, stake] = compute_gains_always_favourite_variable_stake(data, odd / 100, low / 10, high / 10)
                sum_gains = sum(gains)
                roi = sum_gains / stake * 100 if stake > 0 else -100
                if roi > max_roi:
                    max_roi = roi
                    max_odd = odd / 100
                    max_low = low / 10
                    max_high = high / 10
                    print(
                        f"The best compbination for now threshold:{max_odd} low:{max_low} high:{max_high} for a roi of{max_roi:.2f}%")

    print(
        f"\n\n\n*********The best compbination threshold:{max_odd} low:{max_low} high:{max_high} for a roi of{max_roi:.2f}%")


def compute_gains_always_same(data, prono_index, odd_header):
    odd_result = data[[odd_header, 'HDA']].values.tolist()
    return [val[0] - 1 if val[1] == prono_index else -1 for val in odd_result]


def compute_gains_always_favourite(data):
    prono = data['favorite'].values.tolist()
    odds = data[['odd_h', 'odd_d', 'odd_a']].values.tolist()
    result = data['HDA'].values.tolist()
    return [odds[i][prono[i]] - 1 if result[i] == prono[i] else -1 for i in range(len(result))]


def compute_gains_always_favourite_variable_stake(data, threshold, low, high):
    prono = data['favorite'].values.tolist()
    odds = data[['odd_h', 'odd_d', 'odd_a']].values.tolist()
    result = data['HDA'].values.tolist()
    stake = [low if odds[i][prono[i]] > threshold else high for i in range(len(odds))]
    nb_elements = len([1 for el in stake if el > 0])
    print(f"nb elements is {nb_elements}")
    return [(odds[i][prono[i]] - 1) * stake[i] if result[i] == prono[i] else -1 * stake[i] for i in
            range(len(result))], sum(
        stake)


def compute_gains_always_underdog(data):
    odds = data[['odd_h', 'odd_d', 'odd_a']].values.tolist()
    prono = [odds[i].index(max(odds[i])) for i in range(len(odds))]
    result = data['HDA'].values.tolist()
    return [odds[i][prono[i]] - 1 if result[i] == prono[i] else -1 for i in range(len(result))]


def compute_gains_always_underdog_variable_stake(data, threshold, low, high):
    odds = data[['odd_h', 'odd_d', 'odd_a']].values.tolist()
    prono = [odds[i].index(max(odds[i])) for i in range(len(odds))]
    result = data['HDA'].values.tolist()
    stake = [low if odds[i][prono[i]] > threshold else high for i in range(len(odds))]
    return [(odds[i][prono[i]] - 1) * stake[i] if result[i] == prono[i] else -1 * stake[i] for i in
            range(len(result))], sum(
        stake)


def compute_gains_random_result(data):
    odds = data[['odd_h', 'odd_d', 'odd_a']].values.tolist()
    prono = [randrange(3) for i in range(len(odds))]
    result = data['HDA'].values.tolist()
    return [odds[i][prono[i]] - 1 if result[i] == prono[i] else -1 for i in range(len(result))]


def compute_gains_random_result_weighted(data):
    odds = data[['odd_h', 'odd_d', 'odd_a']].values.tolist()
    proba = [randrange(101) for i in range(len(odds))]
    prono = []
    for i in range(len(proba)):
        if proba[i] < 39:
            prono.append(0)
        elif 39 <= proba[i] < 68:
            prono.append(1)
        else:
            prono.append(2)
    result = data['HDA'].values.tolist()
    return [odds[i][prono[i]] - 1 if result[i] == prono[i] else -1 for i in range(len(result))]


def compute_gains_always_second_favourite(data):
    favorites = data['favorite'].values.tolist()
    odds = data[['odd_h', 'odd_d', 'odd_a']].values.tolist()
    underdogs = [odds[i].index(max(odds[i])) for i in range(len(odds))]
    choices = [0, 1, 2]
    prono = []
    for i in range(len(underdogs)):
        choices_copy = choices.copy()
        choices_copy.remove(favorites[i])
        choices_copy.remove(underdogs[i])
        prono.append(choices_copy[0])
    result = data['HDA'].values.tolist()
    return [odds[i][prono[i]] - 1 if result[i] == prono[i] else -1 for i in range(len(result))]


####################################
###### Utils method
def get_splitted_lines_from_src_file(src_file):
    content = src_file.read().splitlines()
    return [line.split('\t') for line in content]


###########################
if __name__ == '__main__':
    # cure_betts_data('resources/features3.csv')
    # filter_out_barbell_betts('resources/features3.csv.out')
    # repartition_result_betts('resources/barbell_betts.csv')
    # goals_scored('resources/barbell_betts.csv')
    # surprising_result('resources/barbell_betts.csv')
    all_simulations('resources/barbell_betts.csv')
    # all_simulations('resources/features3.csv.out')
    # compute_roi_variable_stakes('resources/barbell_betts.csv')
