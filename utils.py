import random
from collections import Counter


def print_repartition_results(arr):
    """
    For an array containing a list of match results print the repartition of those results
    """
    print(repartition_results(arr))


def repartition_results(arr):
    return [sum([1 if x == 0 else 0 for x in arr]) / len(arr) * 100,
            sum([1 if x == 1 else 0 for x in arr]) / len(arr) * 100,
            sum([1 if x == 2 else 0 for x in arr]) / len(arr) * 100]


def alt_result(false_res):
    """
    :param false_res: actual prognostic assumed false
    :return: the new prognostic
    """
    if false_res == 2 or false_res == 0:
        return 1
    else:
        return 0


def simulate_randomness(weights):
    simu = [compute_results_weighted_probs(weights) for _ in range(10)]
    counter = Counter(simu)
    occ = [counter[0], counter[1], counter[2]]
    return occ.index(max(occ))


def compute_results_weighted_probs(weights):
    """
    :param weights: [a, b] where a high boundary home victory, b low boundary away victory
    :return:
    """
    odd = random.randint(0, 100)
    a, b = weights[0], weights[0] + weights[1]
    if odd >= b:
        return 2
    if a <= odd < b:
        return 1
    if odd < a:
        return 0


def empirical_new_result(result, is_correct):
    """
    :param result: result prognosticated
    :param is_correct: is correct prono
    :return:
    We make this assumption : if prono correct 61% just
    """
    prob_true_correct = 59
    prob_false_correct = 100 - prob_true_correct

    prob_true_incorrect = 62
    prob_false_incorrect = 100 - prob_true_incorrect

    ##if correct, prob that true result is
    prob_true_home = int(2 / 3 * prob_false_correct)
    prob_true_AD = prob_false_correct - prob_true_home
    prob_true_ad_eq = int(prob_false_correct / 2)

    ##if incorrect, prob that true result is
    prob_false_home = int(2 / 3 * prob_true_incorrect)
    prob_false_AD = prob_true_incorrect - prob_false_home
    prob_false_ad_eq = int(prob_true_incorrect / 2)

    a, b = 0, 0
    if result == 0 and is_correct:
        a, b = prob_true_correct, prob_true_ad_eq
    if result == 0 and not is_correct:
        a, b = prob_false_incorrect, prob_false_ad_eq
    if result == 1 and is_correct:
        a, b = prob_true_home, prob_true_correct
    if result == 1 and not is_correct:
        a, b = prob_false_home, prob_false_incorrect
    if result == 2 and is_correct:
        a, b = prob_true_home, prob_true_AD
    if result == 2 and not is_correct:
        a, b = prob_false_home, prob_false_AD

    return simulate_randomness([a, b])


def get_prediction_for_cor(model_predict, actual_prediction):
    """"
    For a prediction from the correct_model return the resulting prediction
    """
    # res = [int(actual_prediction[i]) if model_predict[i] == 1 else alt_result(actual_prediction[i])
    #        for i in range(len(actual_prediction))]
    res = [empirical_new_result(actual_prediction[i], model_predict[i] == 1)
           for i in range(len(actual_prediction))]
    return res


def get_model_pkl_file(core_name):
    return 'output/models/' + core_name + '.pkl'
