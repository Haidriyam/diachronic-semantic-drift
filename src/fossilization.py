"""
Diachronic Semantic Drift & Lexical Fossilization Engine
Aligns chronological vector spaces via Orthogonal Procrustes (SVD)
to quantify semantic displacement vs. syntactic stagnation.
"""

import numpy as np
from scipy.spatial.distance import cosine
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

class DiachronicDriftEngine:
    def __init__(self, vector_dim: int = 100, window: int = 5, min_count: int = 1):
        self.vector_dim = vector_dim
        self.window = window
        self.min_count = min_count

    def train_epoch_model(self, tokenized_corpus: list) -> Word2Vec:
        """Trains Word2Vec skip-gram embeddings for a specific temporal epoch."""
        model = Word2Vec(
            sentences=tokenized_corpus,
            vector_size=self.vector_dim,
            window=self.window,
            min_count=self.min_count,
            sg=1,
            seed=42,
            workers=1
        )
        return model

    @staticmethod
    def align_vector_spaces(base_model: Word2Vec, target_model: Word2Vec):
        """
        Orthogonal Procrustes Alignment:
        Finds optimal orthogonal matrix Q to align target space W_target to W_base
        such that trace(W_base * (W_target * Q)^T) is maximized.
        """
        shared_vocab = sorted(list(
            set(base_model.wv.index_to_key).intersection(set(target_model.wv.index_to_key))
        ))
        
        if len(shared_vocab) < 5:
            raise ValueError("Insufficient shared vocabulary for matrix alignment.")

        # Construct shared projection matrices
        matrix_base = np.array([base_model.wv[w] for w in shared_vocab])
        matrix_target = np.array([target_model.wv[w] for w in shared_vocab])

        # SVD on projection cross-product
        correlation_matrix = np.dot(matrix_target.T, matrix_base)
        u, _, vt = np.linalg.svd(correlation_matrix)
        orthogonal_q = np.dot(u, vt)

        # Align target vocabulary into base coordinate frame
        aligned_embeddings = {}
        for word in target_model.wv.index_to_key:
            aligned_embeddings[word] = np.dot(target_model.wv[word], orthogonal_q)

        return shared_vocab, aligned_embeddings

    @staticmethod
    def calculate_drift_metrics(base_model: Word2Vec, aligned_target: dict, shared_vocab: list):
        """Computes Cosine Displacement Distance: d = 1 - cos(v_t1, v_t2)."""
        metrics = []
        for word in shared_vocab:
            vec_base = base_model.wv[word]
            vec_aligned = aligned_target[word]
            displacement = float(cosine(vec_base, vec_aligned))
            metrics.append({
                "word": word,
                "displacement": displacement,
                "fossilized": displacement < 0.20
            })
        metrics.sort(key=lambda x: x["displacement"], reverse=True)
        return metrics

    def render_drift_plot(self, metrics: list, output_filename="semantic_drift.png"):
        """Plots semantic displacement values across target words."""
        words = [m["word"] for m in metrics]
        scores = [m["displacement"] for m in metrics]
        colors = ["#d9534f" if s > 0.25 else "#1b4965" for s in scores]

        plt.figure(figsize=(10, 5), dpi=300)
        bars = plt.barh(words, scores, color=colors, edgecolor="black", linewidth=0.6)
        plt.axvline(x=0.20, color="#6c757d", linestyle="--", linewidth=1.2, label="Fossilization Threshold (<0.20)")
        
        plt.title("Diachronic Semantic Displacement Across Temporal Epochs", fontsize=12, pad=12, weight="bold")
        plt.xlabel("Cosine Semantic Displacement Metric (1 - Cosine Similarity)", fontsize=10)
        plt.gca().invert_yaxis()
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_filename)
        print(f"[+] Saved visualization artifact: {output_filename}")


if __name__ == "__main__":
    # Epoch 1 (Early-Century / Classical Register)
    corpus_epoch1 = [
        ["the", "nice", "calculation", "required", "exact", "precision"],
        ["an", "awful", "splendor", "filled", "the", "cathedral"],
        ["statutory", "provisions", "remain", "hereinabove", "stipulated"],
        ["indenture", "witnesseth", "legal", "obligation", "hereunto"]
    ] * 25

    # Epoch 2 (Modern Industrial / Collocation Shifts)
    corpus_epoch2 = [
        ["we", "enjoyed", "a", "nice", "pleasant", "dinner"],
        ["an", "awful", "disaster", "befell", "the", "railway"],
        ["statutory", "provisions", "remain", "hereinabove", "stipulated"],
        ["indenture", "witnesseth", "legal", "obligation", "hereunto"]
    ] * 25

    engine = DiachronicDriftEngine()
    print("[*] Training Epoch 1 embeddings...")
    m1 = engine.train_epoch_model(corpus_epoch1)

    print("[*] Training Epoch 2 embeddings...")
    m2 = engine.train_epoch_model(corpus_epoch2)

    print("[*] Executing Orthogonal Procrustes alignment via SVD...")
    vocab, aligned_vecs = engine.align_vector_spaces(m1, m2)

    print("\n--- Empirical Semantic Drift Results ---")
    results = engine.calculate_drift_metrics(m1, aligned_vecs, vocab)
    print(f"{'Token':<16}{'Displacement':<20}{'Classification':<15}")
    print("-" * 51)
    for r in results:
        status = "Active Drift" if not r["fossilized"] else "Fossilized"
        print(f"{r['word']:<16}{r['displacement']:<20.4f}{status:<15}")

    engine.render_drift_plot(results)
