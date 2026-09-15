from sentinel.tracking.trajectory_store import TrajectoryStore


def test_update_creates_track_and_appends_points():
    store = TrajectoryStore()
    store.update(1, 0, "person", (0, 0, 10, 20), 0.9, timestamp=0.0)
    store.update(1, 0, "person", (5, 0, 15, 20), 0.9, timestamp=0.1)

    track = store.get(1)
    assert track is not None
    assert len(track.points) == 2
    assert track.points[-1].x == 10.0  # center x of (5,0,15,20)


def test_velocity_computed_over_window():
    store = TrajectoryStore()
    for i in range(5):
        store.update(1, 0, "person", (i * 10, 0, i * 10 + 10, 20), 0.9, timestamp=i * 0.1)
    track = store.get(1)
    vx, vy = track.velocity(window_sec=0.5)
    assert vx > 0
    assert vy == 0


def test_prune_stale_removes_idle_tracks():
    store = TrajectoryStore(stale_ttl_sec=1.0)
    store.update(1, 0, "person", (0, 0, 10, 20), 0.9, timestamp=0.0)
    store.update(2, 0, "person", (0, 0, 10, 20), 0.9, timestamp=5.0)

    removed = store.prune_stale(now=5.5)
    assert removed == [1]
    assert store.get(1) is None
    assert store.get(2) is not None


def test_max_radius_over_detects_small_movement():
    store = TrajectoryStore()
    for i in range(10):
        # tiny jitter around (100, 100)
        store.update(1, 0, "person", (100 + i % 2, 100, 110 + i % 2, 120), 0.9, timestamp=i * 1.0)
    track = store.get(1)
    radius = track.max_radius_over(seconds=10, now=9.0)
    assert radius < 5


def test_by_class_filters_correctly():
    store = TrajectoryStore()
    store.update(1, 0, "person", (0, 0, 10, 20), 0.9, timestamp=0.0)
    store.update(2, 26, "handbag", (0, 0, 10, 20), 0.9, timestamp=0.0)

    persons = store.by_class(["person"])
    assert len(persons) == 1
    assert persons[0].track_id == 1
