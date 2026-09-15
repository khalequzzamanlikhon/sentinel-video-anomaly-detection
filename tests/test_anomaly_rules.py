from sentinel.anomaly.abandoned_object import AbandonedObjectDetector
from sentinel.anomaly.crowd_analyzer import CrowdAnalyzer
from sentinel.anomaly.fall_detector import FallDetector
from sentinel.anomaly.loitering_detector import LoiteringDetector
from sentinel.anomaly.wrongway_detector import WrongWayDetector
from sentinel.tracking.trajectory_store import TrajectoryStore


def test_fall_detector_requires_persistence_before_firing():
    store = TrajectoryStore()
    detector = FallDetector(
        aspect_ratio_drop_pct=50, window_sec=0.5, min_persist_sec=0.5, velocity_threshold_px_per_sec=40
    )

    # Standing: tall narrow box (aspect ratio 3.0), no sudden drop yet -> never fires.
    t = 0.0
    while t <= 2.0:
        store.update(1, 0, "person", (90, 70, 110, 130), 0.9, timestamp=t)
        assert detector.evaluate(store.get(1), "cam1", now=t) is None
        t = round(t + 0.2, 2)

    # Falls: short wide box, same center (no residual velocity).
    fired = None
    t = 2.1
    while t <= 3.5:
        store.update(1, 0, "person", (70, 90, 130, 110), 0.9, timestamp=t)
        e = detector.evaluate(store.get(1), "cam1", now=t)
        if e is not None:
            fired = e
            break
        t = round(t + 0.1, 2)

    assert fired is not None
    assert fired.event_type == "fall"
    assert 0.0 <= fired.confidence <= 1.0


def test_fall_detector_ignores_brief_dip_like_tying_a_shoe():
    """A person crouching briefly (aspect ratio dips) then standing back up before the
    persistence window elapses must NOT trigger a fall alert (Challenge 2)."""
    store = TrajectoryStore()
    detector = FallDetector(
        aspect_ratio_drop_pct=50, window_sec=0.5, min_persist_sec=0.5, velocity_threshold_px_per_sec=40
    )
    t = 0.0
    while t <= 2.0:
        store.update(1, 0, "person", (90, 70, 110, 130), 0.9, timestamp=t)
        detector.evaluate(store.get(1), "cam1", now=t)
        t = round(t + 0.2, 2)

    # Brief crouch for only 0.2s, then back to standing.
    store.update(1, 0, "person", (70, 90, 130, 110), 0.9, timestamp=2.1)
    e = detector.evaluate(store.get(1), "cam1", now=2.1)
    assert e is None

    store.update(1, 0, "person", (90, 70, 110, 130), 0.9, timestamp=2.3)
    e = detector.evaluate(store.get(1), "cam1", now=2.3)
    assert e is None


def test_loitering_fires_once_after_min_duration():
    store = TrajectoryStore()
    detector = LoiteringDetector(radius_px=60, min_duration_sec=5)

    fired = []
    t = 0.0
    while t <= 6.0:
        store.update(1, 0, "person", (95, 95, 105, 135), 0.9, timestamp=t)
        e = detector.evaluate(store.get(1), "cam1", now=t)
        if e:
            fired.append(e)
        t = round(t + 0.5, 2)

    assert len(fired) == 1
    assert fired[0].event_type == "loitering"


def test_abandoned_object_fires_when_no_owner_nearby():
    store = TrajectoryStore()
    detector = AbandonedObjectDetector(
        classes=[26], stationary_sec=5, owner_distance_px=100, stationary_radius_px=20
    )

    fired = []
    t = 0.0
    while t <= 6.0:
        store.update(2, 26, "handbag", (500, 500, 520, 520), 0.9, timestamp=t)
        fired.extend(detector.evaluate(store, "cam1", now=t))
        t = round(t + 0.5, 2)

    assert any(e.event_type == "abandoned_object" for e in fired)


def test_abandoned_object_does_not_fire_when_owner_stays_close():
    store = TrajectoryStore()
    detector = AbandonedObjectDetector(
        classes=[26], stationary_sec=5, owner_distance_px=100, stationary_radius_px=20
    )

    fired = []
    t = 0.0
    while t <= 6.0:
        store.update(2, 26, "handbag", (500, 500, 520, 520), 0.9, timestamp=t)
        store.update(3, 0, "person", (510, 510, 530, 550), 0.9, timestamp=t)  # right next to bag
        fired.extend(detector.evaluate(store, "cam1", now=t))
        t = round(t + 0.5, 2)

    assert fired == []


def test_crowd_density_requires_sustained_count():
    store = TrajectoryStore()
    analyzer = CrowdAnalyzer(person_count_threshold=3, sustained_sec=5)

    fired = None
    t = 0.0
    while t <= 6.0:
        for i in range(3):
            store.update(10 + i, 0, "person", (i * 10, 0, i * 10 + 10, 20), 0.9, timestamp=t)
        e = analyzer.evaluate(store, "cam1", now=t)
        if e:
            fired = e
        t = round(t + 1.0, 2)

    assert fired is not None
    assert fired.details["person_count"] == 3


def test_crowd_density_resets_when_count_drops():
    store = TrajectoryStore()
    analyzer = CrowdAnalyzer(person_count_threshold=3, sustained_sec=5)

    for t in [0.0, 1.0, 2.0]:
        for i in range(3):
            store.update(10 + i, 0, "person", (i * 10, 0, i * 10 + 10, 20), 0.9, timestamp=t)
        assert analyzer.evaluate(store, "cam1", now=t) is None

    # crowd disperses
    store.tracks.clear()
    store.update(10, 0, "person", (0, 0, 10, 20), 0.9, timestamp=3.0)
    assert analyzer.evaluate(store, "cam1", now=3.0) is None


def test_wrong_way_fires_on_sustained_reverse_movement():
    store = TrajectoryStore()
    detector = WrongWayDetector(angle_threshold_deg=90, min_speed_px_per_sec=15, min_persist_sec=1.0)

    fired = None
    x = 500.0
    t = 0.0
    while t <= 3.0:
        store.update(1, 0, "person", (x, 100, x + 10, 140), 0.9, timestamp=t)
        e = detector.evaluate(store.get(1), "cam1", expected_flow_deg=0, now=t)
        if e:
            fired = e
            break
        x -= 5
        t = round(t + 0.2, 2)

    assert fired is not None
    assert fired.event_type == "wrong_way"


def test_wrong_way_does_not_fire_with_expected_flow_direction():
    store = TrajectoryStore()
    detector = WrongWayDetector(angle_threshold_deg=90, min_speed_px_per_sec=15, min_persist_sec=1.0)

    x = 100.0
    t = 0.0
    while t <= 3.0:
        store.update(1, 0, "person", (x, 100, x + 10, 140), 0.9, timestamp=t)
        e = detector.evaluate(store.get(1), "cam1", expected_flow_deg=0, now=t)
        assert e is None
        x += 5  # moving in the expected (0 deg) direction
        t = round(t + 0.2, 2)
