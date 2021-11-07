# Distribution
- Poisson distribution? Problem is, that scores do not have the same unit e.g. 3pts and 2 pts
  - Assumptions like no streakiness or constant rate over time could be argues for, see `gabel_2012`
- For Plus-Minus we should use a Skellam distribution?
  - Possible advantage: Mass under the distribution that is right of x=0 can be interpreted as probability to win?

# Approaches / Ideas
- Use Gaussian Processes to predict players career performance
- Hier. Bayesian Regression for predicting positional value based on box-scores (we need extension to individual player level)
- Couple players contribute significantly, a lot of others only marginally (graphic)

# Measures for performance
- PER $Efficiency = \frac{Pts + Reb + Ass}{GP}$
- PIE: Player Impact Estimate, which percentage of events in a game a player has contributed to (tries to incorporate defensive stats)
## Measures on pbp data
- Regression based and other more complex Approaches
- Impact score by Deshpande, *divide game time into 5min intervals and evaluate who is on the court and game score?*

## Shooting
- Not controling for position and defender distance can create problems!

## Defense
- What does *not* happen, because a player is on the court? (Dwight Effect)

# Production Curves
Development of player production over their career
## Bayesian approaches
- `Page 2013`, Gaussian Processes


## Clusterin based approaches
- `Silver 2015, 2019`

# Network approaches
- Ideas on performance changes for trading players in `VazDeMelo_2012`
