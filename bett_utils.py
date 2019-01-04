def find_avg_odd_for_prec(prec):
    """
    :param prec: precision of prediction method
    :return: print the avg odd needed to have a positive roi
        gain = unit * (odd - 1)
        odd = gain / unit - 1
        Gw - Gl
        Gw = prec * unit * (odd - 1)
        Pw = (1-prec) * unit
        Gw - PW > 0 => prec *odd - 1
                    => odd > 1 / prec
                    => prec > 1 /odd
    """
    unit = 1
    odd = 2.5


if __name__ == '__main__':
    find_avg_odd_for_prec(prec)
