"""Dense retrieval over stored encoder embeddings (exact cosine top-k)."""
import numpy as np


def l2_normalize(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


class VectorIndex:
    """Brute-force cosine index. Vectors are L2-normalised once, so cosine
    similarity becomes a single matrix product. Exact search over ~25k x 256 is
    a few milliseconds per batch, so an approximate index (FAISS/HNSW) is not needed."""

    def __init__(self, vectors):
        self.vectors = l2_normalize(vectors)

    def __len__(self):
        return len(self.vectors)

    def search(self, queries, k=5, exclude=None, batch_size=1024):
        """Return (ids, scores), each (n_queries, k), sorted by descending cosine.

        exclude: optional per-query lists of store ids to skip (e.g. the query's own
        row when the queries come from the training set itself)."""
        q = l2_normalize(np.atleast_2d(queries))
        all_ids, all_scores = [], []
        for start in range(0, len(q), batch_size):
            sims = q[start:start + batch_size] @ self.vectors.T
            if exclude is not None:
                for row, ex in enumerate(exclude[start:start + batch_size]):
                    sims[row, list(ex)] = -np.inf
            top = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
            top_scores = np.take_along_axis(sims, top, axis=1)
            order = np.argsort(-top_scores, axis=1)
            all_ids.append(np.take_along_axis(top, order, axis=1))
            all_scores.append(np.take_along_axis(top_scores, order, axis=1))
        return np.concatenate(all_ids), np.concatenate(all_scores)
