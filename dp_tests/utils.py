

def dict_same(one, two):
    if one.keys() != two.keys():
        return False

    for k in one:
        o, t = one[k], two[k]
        if o.shape != t.shape:
            return False

        if (o != t).any():
            return False

    return True

