# Slow Momentum with Fast Reversion
## About
This code accompanies the the paper [Slow Momentum with Fast Reversion: A Trading Strategy Using Deep Learning and Changepoint Detection](https://arxiv.org/pdf/2105.13727.pdf).

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
@article{wood2021slow,
   title={Slow Momentum with Fast Reversion: A Trading Strategy Using Deep Learning and Changepoint Detection},
   ISSN={2640-3943},
   url={http://dx.doi.org/10.3905/jfds.2021.1.081},
   DOI={10.3905/jfds.2021.1.081},
   journal={The Journal of Financial Data Science},
   publisher={Pageant Media US},
   author={Wood, Kieran and Roberts, Stephen and Zohren, Stefan},
   year={2021},
   month={Dec},
   pages={jfds.2021.1.081}
}
```

