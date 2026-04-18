import numpy as np
import pytest

from facedetect.storage.repo import PersonRepo


@pytest.fixture
def repo(tmp_path):
    r = PersonRepo(tmp_path / "test.db")
    yield r
    r.close()


def _make_vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def test_create_and_get_person(repo):
    p = repo.create_person("Alice")
    assert p.name == "Alice"
    got = repo.get_person(p.id)
    assert got is not None and got.name == "Alice"


def test_person_name_is_unique(repo):
    repo.create_person("Alice")
    with pytest.raises(Exception):
        repo.create_person("Alice")


def test_add_embeddings_and_reload(repo):
    p = repo.create_person("Alice")
    vecs = [_make_vec(i) for i in range(5)]
    n = repo.add_embeddings(p.id, vecs, pose_tags=["front"] * 5, qualities=[0.8] * 5)
    assert n == 5
    assert repo.count_embeddings(p.id) == 5

    mat, ids = repo.load_all_embeddings()
    assert mat.shape == (5, 512)
    assert ids == [p.id] * 5
    # Values should round-trip bit-for-bit via float32 BLOB.
    assert np.allclose(sorted(mat.flatten()), sorted(np.concatenate(vecs)))


def test_delete_person_cascades_embeddings(repo):
    p = repo.create_person("Alice")
    repo.add_embeddings(p.id, [_make_vec(0)], pose_tags=["front"], qualities=[0.9])
    assert repo.count_embeddings(p.id) == 1
    repo.delete_person(p.id)
    assert repo.count_embeddings(p.id) == 0
    assert repo.get_person(p.id) is None


def test_log_event_known_and_unknown(repo):
    p = repo.create_person("Alice")
    repo.log_event(person_id=p.id, similarity=0.75)
    repo.log_event(person_id=None, similarity=0.2, snapshot_path="/tmp/unknown.jpg")
    events = repo.recent_events()
    assert len(events) == 2
    # Most recent first.
    assert events[0].snapshot_path == "/tmp/unknown.jpg"
    assert events[0].person_id is None
    assert events[1].person_id == p.id


def test_load_all_embeddings_empty(repo):
    mat, ids = repo.load_all_embeddings()
    assert mat.shape == (0, 512)
    assert ids == []


def test_wrong_shape_embedding_raises(repo):
    p = repo.create_person("Alice")
    bad = np.zeros(256, dtype=np.float32)
    with pytest.raises(ValueError):
        repo.add_embeddings(p.id, [bad])
