# Slow Momentum with Fast Reversion

> [!IMPORTANT]
> ## Latest Work
> [**DeePM: Regime-Robust Deep Learning for Systematic Macro Portfolio Management**](https://github.com/kieranjwood/deepm) extends the Momentum Transformer to end-to-end portfolio construction, re-implemented in PyTorch. Key contributions:
>
> 1. **Graph neural networks** encoding macroeconomic priors across assets
> 2. **Multi-asset cross-sectional attention** with a causal lag (Directed Delay) mechanism
> 3. **Portfolio-level loss** — optimises on a pooled portfolio Sharpe ratio rather than univariate per-asset objectives
> 4. **Regime-robust minimax optimisation** — a SoftMin proxy for Entropic Value-at-Risk (EVaR) that penalises the worst historical subperiods
> 5. **Realistic transaction costs in the loss** — asset-specific costs baked directly into the training objective
> 6. **Two-pass exact gradient accumulation** — correct gradients for the coupled Sharpe-ratio objective at scale
>
> In backtests from 2010--2025, DeePM roughly doubles the net risk-adjusted returns of classical trend-following and improves upon the Momentum Transformer by approximately fifty percent. See the [paper](https://arxiv.org/abs/2601.05975) and [GitHub](https://github.com/kieranjwood/deepm) for full details.

## About
This code accompanies the the paper [Slow Momentum with Fast Reversion: A Trading Strategy Using Deep Learning and Changepoint Detection](https://jfds.pm-research.com/content/iijjfds/4/1/111.full.pdf) and [preprint](https://arxiv.org/pdf/2105.13727.pdf).

> :warning: This work has now been improved upon with the paper [Trading with the Momentum Transformer: An Intelligent and Interpretable Architecture](https://arxiv.org/pdf/2112.08534.pdf). Please refer to the [this repo](https://github.com/kieranjwood/trading-momentum-transformer) for the implementation of both the *Slow Momentum with Fast Reversion* and *Trading with the Momentum Transformer* papers.

> Momentum strategies are an important part of alternative investments and are at the heart of commodity trading advisors (CTAs). These strategies have however been found to have difficulties adjusting to rapid changes in market conditions, such as during the 2020 market crash. In particular, immediately after momentum turning points, where a trend reverses from an uptrend (downtrend) to a downtrend (uptrend), time-series momentum (TSMOM) strategies are prone to making bad bets. To improve the response to regime change, we introduce a novel approach, where we insert an online change-point detection (CPD) module into a Deep Momentum Network (DMN) pipeline, which uses an LSTM deep-learning architecture to simultaneously learn both trend estimation and position sizing. Furthermore, our model is able to optimise the way in which it balances 1) a slow momentum strategy which exploits persisting trends, but does not overreact to localised price moves, and 2) a fast mean-reversion strategy regime by quickly flipping its position, then swapping it back again to exploit localised price moves. Our CPD module outputs a changepoint location and severity score, allowing our model to learn to respond to varying degrees of disequilibrium, or smaller and more localised changepoints, in a data driven manner. Using a portfolio of 50, liquid, continuous futures contracts over the period 1990-2020, the addition of the CPD module leads to an improvement in Sharpe ratio of one-third. Even more notably, this module is especially beneficial in periods of significant nonstationarity, and in particular, over the most recent years tested (2015-2020) the performance boost is approximately two-thirds. This is especially interesting as traditional momentum strategies have been underperforming in this period.

## Using the code
1. Create a Nasdaq Data Link account to access the [free Quandl dataset](https://data.nasdaq.com/data/CHRIS-wiki-continuous-futures/documentation). This dataset provides continuous contracts for 600+ futures, built on top of raw data from CME, ICE, LIFFE etc.
2. Download the Quandl data with: `python -m data.download_quandl_data <<API_KEY>>`
3. Create Momentum Transformer input features with: `python -m examples.create_features_quandl`. In this example we use the 100 futures tickers which have i) the longest history ii) more than 90% of trading days have data iii) data up until at least Dec 2021.
4. Run the changepoint detection module: `python -m examples.concurent_cpd_quandl`

## Currently Implemented
- [x] CPD Module
- [x] Deep Momentum Network - see [Momementum Transformer](https://github.com/kieranjwood/trading-momentum-transformer) repo

# References
Please cite our paper with:
```bib
@article {Wood111,
	author = {Wood, Kieran and Roberts, Stephen and Zohren, Stefan},
	title = {Slow Momentum with Fast Reversion: A Trading Strategy Using Deep Learning and Changepoint Detection},
	volume = {4},
	number = {1},
	pages = {111--129},
	year = {2022},
	doi = {10.3905/jfds.2021.1.081},
	publisher = {Institutional Investor Journals Umbrella},
	issn = {2640-3943},
	URL = {https://jfds.pm-research.com/content/4/1/111},
	eprint = {https://jfds.pm-research.com/content/4/1/111.full.pdf},
	journal = {The Journal of Financial Data Science}
}
```

