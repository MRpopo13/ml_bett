#########The feature used is the following one
######### odd_h odds_d odds_a odds_o odds_u
#########  1     2      3      4       5
#########HDA_res  diff_goals  h_goal  a_goal  ov_under   btts   goals
######### 6         7            8      9       10       11     12
#########
#########


def cure_betts_data(src_file):
    with open(src_file) as src:
        with open(src_file + '.out', 'w') as out:
            content = src.read().splitlines()
            lines = [line.split('\t') for line in content]
            part = [line[:5] + line[22:29] for line in lines]
            for line in part:
                begin = '\t'.join(line)
                out.writelines(begin + '\n')


def is_barbell_bett(line):
    trunc = line[:3]
    for el in trunc:
        if float(el) < 2:
            return False
    return True


def filter_out_barbell_betts(src_file):
    with open(src_file) as src:
        content = src.read().splitlines()
        lines = [line.split('\t') for line in content]

        filtered = [line for line in lines if is_barbell_bett(line)]
    with open('resources/output.csv', 'w+') as out:
        for line in filtered:
            begin = '\t'.join(line)
            out.writelines(begin + '\n')


if __name__ == '__main__':
    # cure_betts_data('resources/features3.csv')
    filter_out_barbell_betts('resources/features3.csv.out')
