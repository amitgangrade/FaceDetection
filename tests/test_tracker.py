from facedetect.util.tracker import IOUTracker, iou


def test_iou_identical():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_no_overlap():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_half_overlap():
    result = iou((0, 0, 10, 10), (5, 0, 15, 10))
    assert 0.3 < result < 0.4  # 50/150


def test_tracker_persists_id_across_frames():
    t = IOUTracker(iou_threshold=0.3, history_len=5)
    ids1 = t.update([(10, 10, 50, 50)])
    assert ids1 == [1]
    ids2 = t.update([(12, 12, 52, 52)])  # slight movement, heavy overlap
    assert ids2 == [1]


def test_tracker_issues_new_id_for_new_face():
    t = IOUTracker(iou_threshold=0.3)
    t.update([(10, 10, 50, 50)])
    ids = t.update([(10, 10, 50, 50), (200, 200, 250, 250)])
    assert 1 in ids and 2 in ids


def test_dominant_returns_label_when_threshold_met():
    t = IOUTracker(iou_threshold=0.3, history_len=7)
    t.update([(0, 0, 10, 10)])
    track = t.get(1)
    for _ in range(4):
        track.record("alice")
    for _ in range(1):
        track.record("unknown")
    assert track.dominant(min_agree=4) == "alice"


def test_dominant_returns_none_when_no_majority():
    t = IOUTracker(iou_threshold=0.3, history_len=7)
    t.update([(0, 0, 10, 10)])
    track = t.get(1)
    track.record("alice")
    track.record("bob")
    track.record("unknown")
    assert track.dominant(min_agree=4) is None


def test_track_dies_after_too_many_misses():
    t = IOUTracker(iou_threshold=0.3, max_missed=2)
    t.update([(0, 0, 10, 10)])
    assert t.get(1) is not None
    for _ in range(3):
        t.update([])
    assert t.get(1) is None
