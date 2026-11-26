import numpy as np

from ragpipe.retrieval import VectorIndex

STORE = np.array([
    [1.0, 0.0],   # 0: east
    [0.0, 1.0],   # 1: north
    [0.7, 0.7],   # 2: north-east
    [-1.0, 0.0],  # 3: west
], dtype=np.float32)


def test_returns_top_k_by_cosine_similarity_in_order():
    idx = VectorIndex(STORE)
    ids, scores = idx.search(np.array([[1.0, 0.1]]), k=3)
    assert ids[0].tolist() == [0, 2, 1]
    assert scores[0][0] > scores[0][1] > scores[0][2]


def test_cosine_ignores_vector_magnitude():
    idx = VectorIndex(STORE * np.array([[10.0], [1.0], [1.0], [1.0]], dtype=np.float32))
    ids, scores = idx.search(np.array([[5.0, 0.0]]), k=1)
    assert ids[0].tolist() == [0] and np.isclose(scores[0][0], 1.0, atol=1e-5)


def test_k_is_configurable():
    idx = VectorIndex(STORE)
    for k in (1, 2, 4):
        ids, _ = idx.search(np.array([[0.0, 1.0]]), k=k)
        assert ids.shape == (1, k)


def test_can_exclude_the_query_itself_when_searching_the_training_set():
    idx = VectorIndex(STORE)
    ids, _ = idx.search(STORE[[2]], k=2, exclude=[[2]])
    assert 2 not in ids[0].tolist()
