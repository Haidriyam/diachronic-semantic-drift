"""
Diachronic Semantic Drift & Fossilization Engine
Aligns historical vector spaces using Orthogonal Procrustes to quantify semantic shift.
"""

import math
import numpy as np
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

def train_epoch_embeddings(sentences, vector_dim=100, window=5, min_count=2):
    """Trains Word2Vec skip-gram embeddings for an epoch."""
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_dim,
        window=window,
        min_count=min_count,
        sg=1,
        seed=42,
        workers=1
    )
    return model

def orthogonal_procrustes_alignment(base_model, target_model):
    """
    Aligns target_model vector space to base_model using Orthogonal Procrustes:
    Finds orthogonal matrix Q maximizing trace(W_base * (W_target * Q)^T).
    """
    common_vocab = sorted(list(set(base_model.wv.index_to_key).intersection(set(target_model.wv.index_to_key))))
    
    if len(common_vocab) < 10:
        raise ValueError("Vocabulary overlap between epochs too sparse for alignment.")

    w_base = np.array([base_model.wv[word] for word in common_vocab])
    w_target = np.array([target_model.wv[word] for word in common_vocab])

    # Singular Value Decomposition of the correlation matrix
    m = np.dot(w_target.T, w_base)
    u, _, vt = np.linalg.svd(m)
    q = np.dot(u, vt)

    # Transform target vectors into aligned coordinate frame
    aligned_vectors = {}
    for word in target_model.wv.index_to_key:
        aligned_vectors[word] = np.dot(target_model.wv[word], q)

    return common_vocab, aligned_vectors

def compute_fossilization_metrics(base_model, aligned_vectors, common_vocab):
    """
    Computes semantic displacement distance and stability:
    High distance = High drift (polysemy / shift)
    Near-zero distance + stagnant neighbors = Lexical Fossilization
    """
    drift_records = []
    for word in common_vocab:
        vec_t1 = base_model.wv[word]
        vec_t2 = aligned_vectors[word]
        
        # Cosine distance: 1 - cosine_similarity
        dist = float(cosine(vec_t1, vec_t2))
        drift_records.append((word, dist))

    # Sort by semantic displacement
    drift_records.sort(key=lambda x: x[1], reverse=True)
    return drift_records

if __name__ == "__main__":
    # Epoch 1 (e.g., Early 19th Century context: 'nice' = precise/delicate, 'awful' = awe-inspiring)
    epoch_1_corpus = [
        ["the", "nice", "calculation", "required", "exact", "precision"],
        ["an", "awful", "splendor", "filled", "the", "cathedral", "nave"],
        ["the", "network", "of", "canals", "spanned", "the", "province"],
        ["a", "curious", "specimen", "demanded", "nice", "examination"]
    ] * 20

    # Epoch 2 (e.g., Late 19th / Early 20th context: 'nice' = pleasant/agreeable, 'network' = telegraph/rails)
    epoch_2_corpus = [
        ["we", "had", "a", "nice", "pleasant", "dinner", "evening"],
        ["an", "awful", "terrible", "accident", "befell", "the", "train"],
        ["the", "telegraph", "network", "linked", "every", "major", "station"],
        ["she", "was", "a", "nice", "kind", "person", "at", "heart"]
    ] * 20

    print("[*] Training historical vector representations...")
    m1 = train_epoch_embeddings(epoch_1_corpus)
    m2 = train_epoch_embeddings(epoch_2_corpus)

    print("[*] Performing Orthogonal Procrustes coordinate alignment (SVD)...")
    shared_words, aligned_t2 = orthogonal_procrustes_alignment(m1, m2)

    results = compute_fossilization_metrics(m1, aligned_t2, shared_words)

    print("\n" + "=" * 50)
    print(f"{'Lexical Token':<18}{'Semantic Displacement':<25}{'Status':<15}")
    print("=" * 50)
    for word, score in results[:6]:
        status = "Active Drift" if score > 0.4 else "Fossilized / Stable"
        print(f"{word:<18}{score:<25.4f}{status:<15}")
    print("=" * 50)
