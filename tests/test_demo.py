from maskcast.demo import DEFAULT_SCRIPT, face_path, script_at


def test_script_loops():
    total = sum(s[3] for s in DEFAULT_SCRIPT)
    assert script_at(0.0) == DEFAULT_SCRIPT[0]
    assert script_at(total + 0.01) == DEFAULT_SCRIPT[0]
    assert script_at(DEFAULT_SCRIPT[0][3] + 0.01) == DEFAULT_SCRIPT[1]


def test_face_path_stays_in_frame():
    for i in range(1000):
        x, y = face_path(i * 0.05)
        assert -1 <= x <= 1 and -1 <= y <= 1
