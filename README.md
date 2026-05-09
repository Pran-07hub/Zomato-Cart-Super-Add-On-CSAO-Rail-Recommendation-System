CSAO Rail-Cart Super Add-On Recommendation System



A production-grade, real-time add-on recommendation system designed to intelligently suggest relevant food items to users as they build their Zomato cart-maximising Average Order Value (AOV), CSAO attach rate, and Cart-to-Order conversion within a strict latency budget of 200–300ms.



Problem

As users build their cart on Zomato, there is a significant opportunity to recommend relevant add-ons (sides, beverages, desserts) in real time. The challenge: personalise these suggestions across millions of daily sessions while handling cold-start users, sequential cart dynamics, and a hard sub-300ms inference constraint.



Solution Architecture



A two-stage ML pipeline:



| Stage | Component | Purpose | Latency |

|---|---|---|---|

| Stage 1 | Item2Vec + FAISS | Retrieve top-50 candidates from 15K items | \~25ms |

| Stage 2 | LightGBM LambdaRank | Precision rank with full feature set | \~60ms |

| Encoder | GRU Cart Encoder | Sequential cart state as ranking feature | \~15ms |

| AI Edge | LLM Embeddings | Semantic food knowledge + cold-start | Pre-computed |

| Post-process | MMR Re-ranking | Diversity injection into final Top-10 | \~5ms |



\*\*Total end-to-end latency: \~160ms ✓\*\*



Dataset

Fully synthetic dataset mimicking Zomato's operational environment:

\- 50,000 users · 2,000 restaurants · 15,000 menu items

\- 500,000 orders · 1.2M cart interaction events

\- Realistic patterns: city-wise behaviour, peak-hour spikes, cold-start simulation (30% sparse users), Poisson-distributed cart sizes



Key Technical Contributions

\- \*\*GRU Sequential Encoder\*\*-treats the cart as an ordered sequence, not a bag of items, capturing meal completion dynamics (e.g., Biryani → Salan → Gulab Jamun → Drink)

\- \*\*LLM Semantic Embeddings\*\*-fine-tuned sentence-transformers on a food corpus to capture item affinity that sparse interaction data misses; powers cold-start for new items

\- \*\*MMR Re-ranking\*\*-Maximal Marginal Relevance post-processing for recommendation diversity

\- \*\*Cold-start handling\*\*-city + meal-time popularity fallback, cuisine-level priors, and LLM embedding priors for zero-history items



Results



| Model | NDCG@10 | Precision@10 | AUC |

|---|---|---|---|

| Popularity Baseline | 0.31 | 0.22 | 0.61 |

| Matrix Factorisation | 0.44 | 0.35 | 0.72 |

| GRU + LightGBM | 0.57 | 0.47 | 0.83 |

| + LLM Embeddings | \*\*0.61\*\* | \*\*0.51\*\* | \*\*0.85\*\* |



\*\*Projected business impact:\*\* +10–14% AOV lift · >28% CSAO attach rate · >40% add-on acceptance rate



Production Design

\- \*\*Redis\*\* feature store with pre-computed embeddings (\~50ms feature fetch)

\- \*\*FAISS IVF index\*\* with approximate nearest-neighbour search

\- \*\*Kubernetes\*\* HPA for auto-scaling + canary deployment strategy (1% → 5% → 20% → 100%)

\- \*\*A/B testing framework\*\*-14-day, 100K+ sessions per arm, user-level randomisation, guardrail metrics (cart abandonment, CSAT, C2O ratio)



Tech Stack

`Python` · `LightGBM` · `FAISS` · `PyTorch (GRU)` · `Sentence-Transformers` · `Redis` · `FastAPI` · `Optuna` · `NumPy` · `Pandas` · `Scikit-learn`



