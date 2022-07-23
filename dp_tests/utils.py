

def dict_same(one, two):
    assert one.keys() == two.keys()
    try:
        for k in one:
            o, t = one[k], two[k]
            assert o.shape == t.shape
            assert (o == t).all()
    except AssertionError as e:
        raise AssertionError(f"Mismatch in {k}:\n{o}\n{t}") from e

