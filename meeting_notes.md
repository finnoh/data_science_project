# Meeting Notes

## 01 - 26th Oct
Two core parts: recommendation engine (e.g. graph constructed by knn or GMM) and season prediction (MC simulation, bayesian model)

### Clustering
- Strategies fort the GM (slider rebuild - all-in)
  - All-in: only this season counts
- Imperfections in the market, trades are very hard
- Dimensionality reduction
- Include cap constraints
- knn graph or GMM model (criteria for GM), single model per position
- GM can weight categories (100 pts to spend)

### Prediction
- Distribution of predicted wins
- Simulation of whole season or just of "our" team?
- Only simulate the ones that change?
- "Quick Mode" or slider for number of simulated paths
- Prediction parameter for GM

### Other
- Recommendation and Prediction together?
- Shapley values of players onto predicted wins
- Summary site for players (image, stats, bio, hotzone)
