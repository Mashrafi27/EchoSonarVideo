from echo_rl.data.frames import evenly_spaced, midframe


def test_evenly_spaced_basic():
    assert evenly_spaced(42, 5) == [0, 10, 20, 31, 41]


def test_evenly_spaced_edges():
    assert evenly_spaced(10, 1) == [0]
    assert evenly_spaced(3, 5) == [0, 1, 2]
    r = evenly_spaced(100, 8)
    assert r[0] == 0 and r[-1] == 99 and len(set(r)) == 8


def test_midframe():
    assert midframe(42) == 21
