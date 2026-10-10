import numpy as np

from roadspeed.detect.base import Detection
from roadspeed.tracking import Tracker, TrackerConfig, iou_matrix


def det(x, y, w=40, h=30, score=1.0):
    return Detection(bbox=np.array([x, y, x + w, y + h], float), score=score)


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    iou = iou_matrix(a, b)
    assert np.allclose(iou, [[1.0, 1 / 3, 0.0]])


def test_two_objects_keep_ids():
    tr = Tracker(TrackerConfig(), fps=30)
    for k in range(40):
        tr.update([det(10 + 4 * k, 100), det(400 - 3 * k, 300)], k, k / 30)
    tracks = tr.finish()
    assert len(tracks) == 2
    assert all(len(t.observations) == 40 for t in tracks)


def test_short_occlusion_same_id():
    tr = Tracker(TrackerConfig(max_lost_s=1.0), fps=30)
    for k in range(60):
        dets = [] if 20 <= k < 35 else [det(10 + 5 * k, 200)]  # 0.5 s gap
        tr.update(dets, k, k / 30)
    tracks = tr.finish()
    assert len(tracks) == 1
    assert len(tracks[0].observations) == 45


def test_long_occlusion_new_id():
    tr = Tracker(TrackerConfig(max_lost_s=0.3), fps=30)
    for k in range(80):
        dets = [] if 20 <= k < 50 else [det(10 + 5 * k, 200)]
        tr.update(dets, k, k / 30)
    assert len(tr.finish()) == 2


def test_low_score_detections_extend_tracks_only():
    tr = Tracker(TrackerConfig(), fps=30)
    for k in range(30):
        s = 1.0 if k < 10 else 0.3
        tr.update([det(10 + 2 * k, 50, score=s), det(600, 50 + 3 * k, score=0.3)], k, k / 30)
    tracks = tr.finish()
    assert len(tracks) == 1  # the always-low-score object never starts a track
    assert len(tracks[0].observations) == 30


def test_fast_small_box_centre_fallback():
    # a 10x8 box moving 12 px/frame: predicted IoU can be zero early on
    tr = Tracker(TrackerConfig(), fps=30)
    for k in range(30):
        tr.update([det(10 + 12 * k, 100, w=10, h=8)], k, k / 30)
    tracks = tr.finish()
    assert len(tracks) == 1 and len(tracks[0].observations) == 30


def test_merged_blob_assigned_to_front_vehicle():
    tr = Tracker(TrackerConfig(), fps=30)
    for k in range(40):
        a = det(100 + 2 * k, 300, w=60, h=40)  # front vehicle (lower in the image)
        b = det(200 - 2 * k, 270, w=50, h=35)  # vehicle behind, moving the other way
        if 15 <= k < 25:
            x1 = min(a.bbox[0], b.bbox[0])
            y1 = min(a.bbox[1], b.bbox[1])
            x2 = max(a.bbox[2], b.bbox[2])
            y2 = max(a.bbox[3], b.bbox[3])
            dets = [Detection(bbox=np.array([x1, y1, x2, y2]), score=1.0)]
        else:
            dets = [a, b]
        tr.update(dets, k, k / 30)
    tracks = tr.finish()
    assert len(tracks) == 2
    front = max(tracks, key=lambda t: t.observations[0].bbox[3])
    xs = [o.bbox[0] for o in front.observations]
    assert np.all(np.diff(xs) > 0)  # never jumped to the other vehicle
