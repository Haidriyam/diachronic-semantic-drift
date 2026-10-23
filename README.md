# Diachronic Semantic Drift & Lexical Fossilization Engine

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haidriyam/diachronic-semantic-drift/blob/main/notebooks/drift_colab.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.drift--preprint-blue)](https://zenodo.org/)
[![Field: Historical Linguistics](https://img.shields.io/badge/Field-Computational%20Semantics-purple.svg)]()

A quantitative linguistic pipeline applying linear algebra and distributed word representations to track semantic drift, lexical obsolescence, and syntactic fossilization across historical text corpora.

---

## 1. Problem Statement: The Incomparability of Historical Vector Spaces

When training distributional word representations (Word2Vec, FastText) across distinct historical periods $t_1$ and $t_2$, the resulting coordinate spaces cannot be directly compared. Because stochastic initialization and rotational invariance cause vector axes to rotate arbitrarily during optimization, measuring raw distance across models yields spurious noise.

This repository resolves the coordinate misalignment problem by computing an **Orthogonal Procrustes Alignment** via Singular Value Decomposition (SVD).

---

## 2. Mathematical Methodology

Let $W^{(t_1)}, W^{(t_2)} \in \mathbb{R}^{|V| \times d}$ be the embedding matrices of shared vocabulary $V$ with dimension $d$. We solve for the optimal orthogonal transformation matrix $Q \in \mathbb{R}^{d \times d}$ that aligns epoch $t_2$ to baseline epoch $t_1$:

$$Q = \arg\min_{\Omega^T \Omega = I} \| W^{(t_1)} - W^{(t_2)} \Omega \|_F$$

### SVD Solution
1. Compute the cross-epoch correlation matrix:
   $$M = (W^{(t_2)})^T W^{(t_1)}$$
2. Factorize via Singular Value Decomposition:
   $$M = U \Sigma V^T$$
3. Derive the optimal orthogonal projection:
   $$Q = U V^T$$

### Semantic Displacement Distance
For any shared lexical item $w$, its displacement across temporal boundaries is computed as the cosine distance in aligned space:

$$D(w) = 1 - \frac{\vec{v}_{t_1}(w) \cdot (Q^T \vec{v}_{t_2}(w))}{\|\vec{v}_{t_1}(w)\| \|Q^T \vec{v}_{t_2}(w)\|}$$

* **Active Semantic Drift ($D(w) \gg 0.20$):** Tokens that shift contextual domain (e.g., *nice* shifting from "precise" to "pleasant"; *awful* shifting from "awe-inspiring" to "terrible").
* **Lexical Fossilization ($D(w) \to 0$):** Archaic statutory, bureaucratic, or formulaic idioms (e.g., *hereinabove*, *witnesseth*, *indenture*) that exhibit complete semantic and neighborhood stagnation.

---

## 3. Empirical Results
