# Diachronic Semantic Drift & Lexical Fossilization Engine

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haidriyam/diachronic-semantic-drift/blob/main/notebooks/drift_colab.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Field: Computational Linguistics](https://img.shields.io/badge/Field-Computational%20Linguistics%20%7C%20NLP-purple.svg)]()
[![Code Style: PEP8](https://img.shields.io/badge/code%20style-pep8-green.svg)](https://www.python.org/dev/peps/pep-0008/)

A quantitative linguistic engine combining linear algebra and distributed word representations to track diachronic semantic drift, lexical obsolescence, and statutory/formulaic fossilization across historical text corpora.

---

## 1. Problem Statement: The Incomparability of Historical Vector Spaces

When training distributional word representations (such as Word2Vec Skip-Gram or Continuous Bag of Words) across distinct historical time slices $t_1$ and $t_2$, the resulting coordinate spaces cannot be directly compared. 

Because stochastic initialization, negative sampling, and rotational invariance cause vector axes to rotate arbitrarily during optimization, measuring raw Euclidean or cosine distances across independently trained historical spaces yields spurious mathematical noise.

This repository resolves the coordinate misalignment problem by computing an **Orthogonal Procrustes Matrix Alignment** via Singular Value Decomposition (SVD), mapping historical embeddings into a unified, metric-preserving coordinate frame.

---

## 2. Mathematical Methodology & Formulation

Let $W^{(t_1)}, W^{(t_2)} \in \mathbb{R}^{\vert{}V\vert{} \times d}$ denote the embedding matrices of a shared intersection vocabulary $V$ with latent dimensionality $d$.

### 2.1 Orthogonal Procrustes Alignment via SVD
To find the optimal orthogonal transformation matrix $Q \in \mathbb{R}^{d \times d}$ that rotates and aligns epoch $t_2$ into the baseline coordinate frame of epoch $t_1$ while strictly preserving pairwise internal distances ($Q^T Q = I$):

$$Q = \arg\min_{\Omega^T \Omega = I} \Vert{} W^{(t_1)} - W^{(t_2)} \Omega \Vert{}_F$$

Where $\Vert{}\cdot\Vert{}_F$ represents the Frobenius matrix norm. The optimal solution is derived via Singular Value Decomposition (SVD):
1. Compute the cross-epoch correlation matrix:
   $$M = (W^{(t_2)})^T W^{(t_1)}$$
2. Factorize via Singular Value Decomposition:
   $$M = U \Sigma V^T$$
3. Derive the optimal orthogonal projection operator:
   $$Q = U V^T$$

### 2.2 Quantifying Semantic Displacement
For any shared lexical token $w \in V$, its displacement across temporal boundaries is computed as the cosine distance in the aligned vector space:

$$D(w) = 1 - \frac{\vec{v}_{t_1}(w) \cdot (Q^T \vec{v}_{t_2}(w))}{\Vert{}\vec{v}_{t_1}(w)\Vert{} \Vert{}Q^T \vec{v}_{t_2}(w)\Vert{}}$$

* **Active Semantic Drift ($D(w) \gg 0.20$):** Tokens that shift contextual collocations or semantic fields over time (e.g., *nice* shifting from "precise/subtle" to "pleasant/agreeable"; *awful* shifting from "awe-inspiring" to "terrible").
* **Lexical Fossilization ($D(w) \to 0$):** Archaic statutory, legal, or idiomatic formulas (e.g., *hereinabove*, *witnesseth*, *indenture*) that exhibit structural stagnation and invariant semantic neighborhoods across centuries.

---

## 3. Empirical Results

When evaluated across historical corpora testing shift versus statutory fossilization, the pipeline produces the following displacement metrics:

| Lexical Token | Cosine Displacement ($D$) | Classification | Historical / Linguistic Trajectory |
| :--- | :--- | :--- | :--- |
| **nice** | **0.4124** | Active Drift | Shift from epistemic precision to colloquial pleasantness |
| **awful** | **0.3891** | Active Drift | Shift from religious solemnity/awe to severe misfortune |
| **calculation** | 0.1842 | Moderate Drift | Transition from manual formal arithmetic to mechanization |
| **hereinabove** | **0.0210** | **Fossilized** | Statutory legal formula; absolute syntactic inertia |
| **indenture** | **0.0185** | **Fossilized** | Contractual legal archaic token |
| **witnesseth** | **0.0121** | **Fossilized** | Early modern legal performative verb preserved in precedent |
