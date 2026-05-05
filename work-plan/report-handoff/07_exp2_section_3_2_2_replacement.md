# Experiment 2 Report Patch: Section 3.2.2

Use the following replacement text for the thesis report section titled `3.2.2 Hybrid Reranker: Balancing Factual and Style Relevance`.

## Replacement Text

In the personalized `R1` condition, retrieval is not limited to query augmentation alone. After hybrid retrieval over the D2L corpus using dense FAISS search, sparse BM25 search, and reciprocal rank fusion (RRF), the candidate chunks are passed through a secondary hybrid reranking stage. This reranker is designed to preserve factual relevance as the dominant retrieval signal while allowing learner-profile preferences to act as a limited secondary preference signal.

For each candidate chunk $c_i$, the final reranking score $\rho_i$ is computed as:

```tex
\begin{equation}
\rho_i = \alpha \cdot \hat{s}_{\mathrm{sem}}(c_i) + (1 - \alpha) \cdot s_{\mathrm{style}}(c_i)
\label{eq:exp2-hybrid-reranker}
\end{equation}
```

where $\alpha = 0.75$. This weighting ensures that semantic retrieval relevance remains the dominant factor, while style alignment contributes only a smaller adjustment.

The semantic component $\hat{s}_{\mathrm{sem}}(c_i)$ is the candidate's fused retrieval score after RRF, normalized to the interval $[0,1]$ across the current candidate set:

```tex
\begin{equation}
\hat{s}_{\mathrm{sem}}(c_i) = \frac{s_{\mathrm{rrf}}(c_i) - s_{\min}}{s_{\max} - s_{\min}}
\label{eq:exp2-semantic-normalization}
\end{equation}
```

Thus, the implementation does not compute a fresh cosine similarity between the chunk embedding and the original query at reranking time. Instead, it reuses the fused retrieval score already produced by the hybrid retrieval pipeline.

The style component $s_{\mathrm{style}}(c_i)$ is a heuristic tag-alignment score rather than an embedding similarity to an FSLSM style descriptor. Each learner profile provides a set of preferred content tags (`reranking_bias`) and deprioritized tags (`deprioritize`). For a given chunk, the reranker counts how many preferred tags and deprioritized tags are matched in the chunk text, then maps the net result to the interval $[0,1]$:

```tex
\begin{equation}
s_{\mathrm{style}}(c_i) = \frac{\left(h_i^{+} - h_i^{-}\right) + M}{2M}
\label{eq:exp2-style-score}
\end{equation}
```

where $h_i^{+}$ is the number of matched preferred tags, $h_i^{-}$ is the number of matched deprioritized tags, and $M$ is the number of active preferred tags for the current learner profile. This formulation gives higher scores to chunks whose textual structure and content markers better match the learner's profile-derived preferences, while still constraining the contribution of style to a secondary role.

Accordingly, the Experiment 2 reranker should be described as a weighted combination of normalized hybrid retrieval relevance and heuristic FSLSM tag alignment. It should not be described as a cosine-similarity reranker over chunk embeddings and style-descriptor embeddings, because that is not the method implemented in the evaluated pipeline.

## Short Thesis Note

If you want one sentence for the methodology summary table:

`R1 retrieval = hybrid BM25+FAISS+RRF retrieval, followed by profile-aware reranking using normalized RRF relevance and heuristic FSLSM tag alignment (alpha = 0.75).`
