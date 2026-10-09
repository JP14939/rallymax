from demo.dtw_demo import cases, dtw_distance


def test_dtw_orders_similarity():
    ref, rows = cases()
    d = {label: dtw_distance(ref, seq) for label, seq in rows}
    assert d['same swing'] == 0
    assert d['same swing at half speed'] < d['a different swing']
    assert d['same swing with noise'] < d['a different swing']
