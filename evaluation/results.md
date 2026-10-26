Leave-one-out over 46 users, 221 menu items (random HR@10 ≈ 0.045).

| Model | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|
| Random | 0.022 | 0.022 | 0.009 | 0.009 |
| Popularity (time-decayed) | 0.457 | 0.457 | 0.457 | 0.457 |
| Content (TF-IDF profile) | 0.457 | 0.609 | 0.398 | 0.449 |
| Collaborative (SVD) | 0.630 | 0.739 | 0.546 | 0.583 |
| BPR | 0.717 | 0.761 | 0.693 | 0.708 |
| Personalized blend (0.55 SVD + 0.45 BPR) | 0.783 | 0.804 | 0.734 | 0.740 |
| Hybrid (SVD + BPR + content + popularity) | 0.804 | 0.826 | 0.762 | 0.769 |
