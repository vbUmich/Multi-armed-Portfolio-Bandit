# Multi-armed Portfolio Bandit

## 1. Introduction

### 1.1 Overview
Active asset managers are commonly faced with the dilemma of selecting competing investment strategies as market conditions evolve to favor one strategy over a set of others. Reinforcement learning offers a solution to supplement established strategies, represented by supervised learning models, by balancing between the exploration and exploitation of new and known knowledge. In our application to dynamic allocation, we employ a multi-armed bandit, one of the classical reinforcement learning problems, to balance the rewards provided by a set of random forest driven strategies. In particular, instead of a multi-armed bandit balancing between investing in securities or funds directly, our approach treats each underlying signal from a given model as an independent strategy that the agent can act on.

### 1.2 Main objective
Our goal with this project is to construct a framework to better allocate financial assets in a dynamical way through time. From our experience in real-world practice, this is typically accomplished by taking two separate actions: first by identifying one or more  investment strategies/rules that can perform that allocation with a edge (over the alternative of holding an allocation), and secondly, by choosing which strategy to implement at any point in time: given they might be competing, and its usual to assume that no strategy is perfect and expected to perform well in all market conditions. Thus, we construct a practical data-driven framework to tackle this common investment problem.

## 2. Conceptual Framework

### 2.1 General Framework 
We picture our setting as consisting of a portfolio manager in charge of the allocation of a balanced portfolio (i.e., investing in equity and fixed income securities), where they have to choose between the recommendations of different investment strategies on a weekly basis. Different strategies can suggest competing allocations which can yield varying results through time, as is typical in the financial practice. We model the portfolio manager as a reinforcement learning agent, in particular a multi-armed bandit, with an action space consisting of recommendations based on supervised learning models. 

In this setting, the main task of the reinforcement learning agent is to exploit the strategy with the best estimated performance, while balancing the component of exploration that permits the agent to deviate from the estimated optimal action. Finding this balance between exploration and exploitation allows the agent to adapt to the distributional conditions of the market that impact the performance of the underlying strategies. The main task of the supervised models is to outperform the market, although given the nature of financial markets, their out-of-sample performance will always be called into question. Our expectation is that the reinforcement learning agent adds value to these strategies by keeping in check possible drifts in performance of the underlying models as the markets evolve over time.

The goal of employing a reinforcement learning approach is for the agent to gradually identify the best performing strategies and adapt their selection of actions to reflect changing performance of the underlying models. Timing the optimal switch between strategies is another component the exploratory factor can have an advantage over a rigid model that explicitly follows the estimated optimum. In a financial context, exploring alternative strategies can be viewed as the agent taking risks by randomly selecting a strategy to observe its yield. The agent builds a history of rewards for each arm and based on the given policy acts upon those rewards to determine the next action the agent should take.

### 2.2 A note on Exploration vs Exploitation
One particularly important point to note is that, in a traditional sense, exploration and exploitation is not relevant in our financial setting. This stems from the fact that we do not need to act upon a financial asset or investment strategy to obtain a reward from it, this will happen regardless with the passage of time, as markets are independent of the actions taken by any single agent. As we always observe counterfactual rewards, we can always update reward estimates. This continuous updating of rewards is a key modification we implement on our reinforced models. It is so important that might even call into question whether they really should be considered reinforcement learning models (as the interaction between the agent and the environment is not meaningfully needed), however, there is one important way in which we have a trade off between exploration and exploitation where we think reinforcement learning models can be of value, and that is through time: underlying distributional market conditions are always changing, and properties of the rewards of any strategy might not be stationary over time, as such, there is value in exploring the time-changing nature of rewards versus exploiting known past rewards properties. 

### 2.3 Reinforcement Learning Framework

**Agents and Policies**
The reinforcement learning agent in our case is a Multi-Armed Bandit (MAB). We choose this model for its simplicity and natural fit to our action space and adaptability in terms of reward functions. Multi-armed bandits very clearly identify the exploration versus exploitation trade-off that is studied within the domain of reinforcement learning. If the multi-armed bandit suggests that the agent has an edge by exploring, there is merit to implementing more advanced reinforcement learning methods to solve the problem. We approached our problem using different policies, which were also selected due to their tractability and interpretability. We relied on several variations of epsilon-greedy strategies.

- Variation on Exploration Intensity: we used several parameters for epsilon, ranging from 0 to 0.40, inclusively, at 0.10 intervals. 
- Variation on Exploitation Profile: we used a parameter for decaying the epsilon parameter over time. 
- Variation on Stationarity: we constructed two non-stationary versions for the estimation of mean reward of actions by:
  - Considering only a historical buffer of past rewards to calculate mean rewards,
  - Considering an exponentially decreasing mean reward to calculate mean reward.


**Action Space**
The action space is detailed in the next section, for the time being suffice it to say they correspond to portfolio allocations as recommended by specific investment strategies based on supervised learning models.

**Rewards**
Our MAB aims to maximize the cumulative sum of the rewards observed through the whole experiment. We consider two distinct reward-generating functions that serve different purposes from a financial perspective: the first is constructed to measure Profit and Loss (P&L), the second one measures risk-adjusted profit and loss (Sharpe ratio). These are common financial objectives when constructing portfolios or analyzing strategies. 
- Portfolio Profit and Loss: at each time step it calculates the return of the recommended portfolio.
- Portfolio Risk-Adjusted Profit and Loss: at each time step it calculates the return of the recommended portfolio, divided by the expected ex-ante standard deviation of the recommended portfolio. This standard deviation is calculated by assuming the historical covariances between the equity and fixed income stay the same.

### 2.4 Action Space: Strategies and Supervised Models
As previously discussed, the action space of our reinforcement learning agent consists of several investment strategies, each recommending a given portfolio allocation. The basis for these recommendations are investment signals that dictate whether the equity market will outperform or not the fixed income market in a given period. For simplicity purposes we consider three scenarios: positive, neutral, or negative. The portfolio construction then follows the signal: in a positive scenario the portfolio should be biased towards equities, and in the case of a negative signal, the portfolio should be biased towards fixed income. The portfolio will maintain its initial allocation (a given starting parameter) in the neutral case. 

In order to obtain the investment signals, we construct two distinct supervised models that follow a common framework but differ in specific implementation. We used random forest classifiers given their versatility in the use of features, in particular little pre-processing needed, and their robustness to outliers and non-linear data (Kho, 2018). Both classifiers are based on different feature sets and trained on similar labels but with differing frequency. We settled on using mostly default parameters for both models given the small number of features available.

### 2.5 Model Strategy 1
**Features**
This strategy is based on the work of Panigirtzoglou (2019), who identified a set of features that have predictive power on the short-term market direction. The set of features includes:

- Relative Valuation: The excess of S&P 500 12M forward earnings yield versus the 10-year real U.S. treasury yield.
- Economic Momentum: The change in Global Manufacturing Purchasing Managers Index (PMI) over two-month periods.
- Speculative Positioning: The net speculative positions on S&P 500 futures over U.S. treasury futures as disclosed by the Commodity Futures Trading Commission.
- Investment Flows: The trend component (as obtained by a Hodrick–Prescott filter with lambda parameter of 100) of the difference in cumulative flows between equity and fixed income (as measured by ICI for mutual funds and by Bloomberg for ETFs).
- Price Momentum: The absolute six-month return of the S&P 500.
- Price Reversal: The absolute one-month return of the S&P 500.

These features have frequencies ranging from daily to monthly updates and are reduced to a common frequency, using the last available measurement for a given prediction at a given period. These features are not used directly, but rather a rolling z-score transformation is applied to each feature to account for outliers in recent history.

**Labels**
Label construction was an essential element in our experiment design. We aim to identify next month’s excess returns above (and below) a 3.0% threshold. But we do so carefully: we want to identify months in which the first occurrence breaches the threshold, even though by the end of the month the excess return is within bounds. This way we serve two purposes. First, we mimic an important real-world consideration, an asset manager does not necessarily care what happens between two exact dates, but what happens within a month. If the markets fall and then recover, we need to ensure this event is identified as it presents an opportunity, even when by the end of the month the performance was flat. Secondly, this way of defining the labels almost alleviates label imbalance issues that would arise if we just defined the labels on a fixed date window (see **Figure 1**).

<p align="center">
  <img src="https://github.com/vbUmich/Multi-armed-Portfolio-Bandit/blob/main/docs/images/model_1_freq_counts.png">
</p>
<p align="center">
  <b>Figure 1</b>
</p>

### 2.6 Model Strategy 2
**Features**
This strategy is based on the work of MAN AHL’s Target Risk Strategies (Korgaonkar, 2022), who implemented a set of rules that guide the level of risk of balanced portfolios they manage, which are similar in nature to the portfolios we are considering in this exercise. The set of features includes:

- Intraday Correlation: daily one-minute intraday correlation between the S&P 500 as proxied by the SPY U.S. ETF and the 10-year U.S. treasuries as proxied by the IEF U.S. ETF. We use the rolling maximum in a given week. 
- Equity Volatility: as measured by the VIX S&P option index. We use both the level and the weekly change on the level.
- Fixed Income Volatility: measured by the MOVE option index. We use both the level and the weekly change on the level.
- Equity Momentum: measured by the six-month absolute return of the SPY U.S. ETF changed. We use a one-year rolling z-scaled transformation.
- Bond Momentum: measured by the six-month total return of the U.S. 10-year constant maturity index. We use a one-year rolling z-scaled transformation.

**Labels**
Given the entirety of the feature set has at least daily frequency, we construct the prediction labels similar to the other strategy, but with a one-week horizon and a 1.5% overperformance threshold. Similarity to the previous case, there is no large concern regarding label imbalance (see **Figure 2**)

<p align="center">
  <img src="https://github.com/vbUmich/Multi-armed-Portfolio-Bandit/blob/main/docs/images/model_2_freq_counts.png">
</p>
<p align="center">
  <b>Figure 2</b>
</p>

### 2.7 Baseline Strategies
The two following strategies are constructed for baseline comparison purposes. There was an a priori expectation that these strategies would underperform compared to our supervised learning models. Knowing this, we expected that our multi-armed bandit would select these strategies less often than those generated by our supervised learning models. 

**Static Allocation**: The first baseline strategy we constructed for comparison purposes follows a single allocation determined at the start of the experiment, typically representing the initial allocation of the portfolio. For example, if the initial target allocation is set to 60% equity and 40% fixed income, the static allocation will maintain that asset mix throughout the entire testing period. This follows a long-term passive approach to managing a portfolio, without rebalancing and no transaction costs.

**Naive Momentum**: The second baseline strategy represents a non-trading condition that continues with the previously selected condition. If the allocation were 50% equity and 50% fixed income in the prior period, the momentum baseline strategy would maintain that allocation in the current period. At the first date of the testing period, this baseline strategy maintains the initial target allocation of the portfolio.

### 2.8 Overarching Assumptions
Since our multi-armed bandit ran in a closed system, assumptions surrounding the effects of changing a portfolio’s allocation must be made. When trading in a market, any trade contributes to the aggregate pool of signals that drive prices up or down for a given security. When evaluating our bandit, we assume that our actions, buying or selling shares of a fund, are negligible in that the volume that is traded does not affect the true price of the stock. In other words, the actions taken by our multi-armed bandit do not change the state of the market. 


### 2.9 Evaluation of The Models
In alignment with the reward functions constructed, the most important evaluation metrics of the success of our reinforcement learning model are the test period Profit and Loss (P&L) of the resulting actions and its test period Sharpe Ratio. We compared these metrics first to the previously mentioned baselines and then between different configurations of the MABs. 

We also evaluated the performance of the supervised models, but only as an intermediate step to our analysis and not as an end-objective of our project. The main measure is average out-of-sample accuracy of the random forest classifiers. 

For the purposes of evaluation, we used a test period that spans from July 1st, 2016, until the end of December 2020, and represents about 35% of our dataset we used to develop our supervised learning models. We wanted to observe the performance of each model in different financial environments, and this period spans at least two bull markets in 2016 - 2017 and Q2 2020 - Q4 2020, and two notable bear markets in 2018 and 2020.

## 3. Data 
Data to train our supervised learning models were obtained from several sources and linked within the source code. We include data from the Commodity Futures Trading Commission, Wharton Research Data Services’ (WRDS) TAQ millisecond database and Bloomberg Terminal. Since both WRDS and Bloomberg data require credentialed access, included samples from the datasets in our repository. In all cases historical data is available from at least 2010 until the end of 2021, with varying frequency from intraday to monthly. When needed, data was resampled to match the required frequency of the models, by filling forward the last available data for a given feature.

To construct the portfolio, we decided to incorporate two funds, one equity fund and another fixed income fund to create a simple balanced portfolio where the allocation between the two funds can be adjusted based on signals from our underlying strategies. The equity-centric fund was the iShares S&P 500 Index Fund (Class K), and the fixed income fund was the iShares 7-10 Year Treasury Bond ETF, both funds sponsored by BlackRock, an American multinational asset manager. We retrieved the daily returns and net asset values of each of the selected funds through a mutual fund dataset maintained by the Center for Research in Security Prices (CRSP) and accessed through the Wharton Research Data Services (WRDS). Data for two funds were used in our testing period between July 2016 and December 2020 to make up our portfolios based on the allocations determined by our strategies. Daily return data for each fund was converted to weekly returns to align with our weekly signals from our models.

## 4. Results
Here we present the results of our models as pertaining to the testing set period that ranges from the second half of 2016 until the end of 2020.

### 4.1 Supervised Models
**Out of Sample Evaluation**
Our aim in analyzing the performance of the supervised models, is first to check whether they are achieving reasonable performance outside the training set, and secondly to form a prior into the expected behavior of the reinforcement learning agent. 

With respect to the accuracy, the supervised models perform minimally well as evidenced by the metrics on **Table 1**. Both models are achieving accuracies in excess of a naive threshold (0.33, random classification) for a balanced three-way classification, and in a financial setting is a respectable accuracy in classifying market direction. This accuracy is not however uniform with respect to the classes, both models suffer in identifying the -1 labels (falling equity markets).

As expected, general accuracy falls greatly from the training period to the testing period, but the effect is much more marked for Model 1, which may be due to overfitting. We can speculate that Model 1 is more prone to such problems given less features at its disposal.

Given the better out-of-sample performance of Model 2, we can speculate that the reinforcement learning agent would tend to have a preference for this strategy over Model 1.

<p align="center">
  <img src="https://github.com/vbUmich/Multi-armed-Portfolio-Bandit/blob/main/docs/images/supervised_confusion_matrices.PNG">
</p>
<p align="center">
  <b>Table 1 - Supervised Model Confusion Matrices and Aggregated Metrics</b>
</p>

**Qualitative Evaluation During COVID**
It is interesting to discuss the behavior of the models during 2020, especially after the start of the COVID pandemic. As evidenced by **Figures 3 & 4**, both models failed to adequately avoid the most important drawdowns during the pandemic, however Model 1 was positive before the fall whereas Model 2 was neutral. It is still interesting to note how both models noticed the high volatility, as the neutral probability was significantly reduced. Model 1 seems to have been much more reactive during this period, and quickly adopted a positive stance that as the market recovered ceded ground to the preponderance of neutral and negative probabilities. Model 2 on the other hand stayed positive for quite some time, only ceding ground in the last quarter of the year. During the few months as the market recovered, Model 2 appeared to have had the advantage.

<p align="center">
  <img src="https://github.com/vbUmich/Multi-armed-Portfolio-Bandit/blob/main/docs/images/model_1_label_evolution.png">
</p>
<p align="center">
  <b>Figure 3</b>
</p>

<p align="center">
  <img src="https://github.com/vbUmich/Multi-armed-Portfolio-Bandit/blob/main/docs/images/model_2_label_evolution.png">
</p>
<p align="center">
  <b>Figure 4</b>
</p>

### 4.2 Reinforcement Learning Model

**Main Findings**

We found it is no easy task to add value from a P&L perspective through the use of a MAB selecting strategies. One way in which it appears to add value, is through quickly discarding strategies that tend to perform worse. This is evident by inspecting **Figure 5**, where we can notice how strategies ‘static_60/40’ and ‘naive momentum’ were not frequently selected. What was much more challenging, is in adding value by switching between remaining strategies that appear to have an edge such as strategies ‘supervised 1’, ‘supervised 2’ and ‘perfect_foresight_20’. The MAB strategy ended up being suboptimal to just sticking with each of these best performing strategies, as it correctly identified the best performing strategy, but switched in particularly bad moments only to reverse course later on.

<p align="center">
  <img src="https://github.com/vbUmich/Multi-armed-Portfolio-Bandit/blob/main/docs/images/eg_0_rel_returns_action_counts_pl.png">
</p>
<p align="center">
  <b>Figure 5</b>
</p>

As can be appreciated from **Table 2**, the non-stationary version of the P&L reward function appears to work better, in particular for short buffer periods, both for P&L and Sharpe ratio metrics. From **Figure 6** we can appreciate an important qualitative change in how the MAB selects strategies, as no single one dominates as strongly as in the other settings. In general, this is a remarkably interesting result from a financial perspective. Non-stationary Greedy MABs can be interpreted as what in a financial setting would be called a ‘time series momentum’ policy, which in essence prioritizes strategies or assets that have performed best in a recent rolling window period. There is a large strand of literature (see for example Asness et al, 2013) that documents this phenomenon, and actually finds that shorter window periods tend to work best, although at the cost of typically increased costs (which we do not analyze here). 

<p align="center">
  <img src="https://github.com/vbUmich/Multi-armed-Portfolio-Bandit/blob/main/docs/images/policy_comparison_pl.PNG">
</p>
<p align="center">
  <b>Table 2</b>
</p>

<p align="center">
  <img src="https://github.com/vbUmich/Multi-armed-Portfolio-Bandit/blob/main/docs/images/nseg_0_rel_return_action_counts_pl.png">
</p>
<p align="center">
  <b>Figure 6</b>
</p>

Sharpe-based reward functions appear to work best in selecting superior-performing strategies, as can be seen from **Figure 7**, they quickly converge to Supervised Model 1, without deviating from it. This makes some intuitive sense, as Sharpe ratios are commonly used in the industry to identify superior strategies with more precision (through risk-adjustment). It is unsettling however, to notice that when analyzing non-stationary versions of the MAB with Sharpe-based rewards, the performance suffers strongly (see **Figure 8**), and even more so with bigger buffers (see **Table 3**). This calls into question the validity of this whole configuration, as it could be just due to a ‘lucky start’ in the non-stationary formulation, given the strong divergence in results, and the solid literature supporting momentum-like policies discussed previously.

<p align="center">
  <img src="https://github.com/vbUmich/Multi-armed-Portfolio-Bandit/blob/main/docs/images/eg_0_rel_returns_action_counts_sharpe.png">
</p>
<p align="center">
  <b>Figure 7</b>
</p>

<p align="center">
  <img src="https://github.com/vbUmich/Multi-armed-Portfolio-Bandit/blob/main/docs/images/nseg_0_rel_return_action_counts_sharpe.png">
</p>
<p align="center">
  <b>Figure 8</b>
</p>

<p align="center">
  <img src="https://github.com/vbUmich/Multi-armed-Portfolio-Bandit/blob/main/docs/images/policy_comparison_sharpe.PNG">
</p>
<p align="center">
  <b>Table 3</b>
</p>

Finally, neither the epsilon parameter nor its decay plays a relevant role in our results. This was expected, as we are using a modified version of the MAB algorithm, one in which at every time step the mean rewards are updated, eliminating the need to randomly explore alternatives to increase their reward estimates. 

### 4.3 Robustness Checks
As a way to assess the robustness of our results, we tested our multi-armed bandit on a set of strategies that have a certain degree of accurate foresight. The idea is that the multi-armed bandit should be able to identify the investment strategy that on average produces higher rewards on the market. 

In order to proxy for this, we created multiple foresight strategies, with a 50%, 40%, 30%, 20%, and 10% chance to sample from the optimal strategy at every period. Since our signal only represented three states, a strategy that had a 30% perfect foresight in fact can have anywhere between 33.33% - 63.33% of all dates having the optimal strategy.

Our results indicate that our multi-armed bandit is effective in identifying the strategy with the highest degree of sampling from the perfect strategy, in its various forms. There is some loss to performance when switching between a stationary policy to a non-stationary policy, although this can be expected as the degree of foresight applied is in effect stationary. 

## 5. Discussion

As we reflect on our findings, our modified multi-armed bandit manages multiple underlying strategies, adapting to shifting market conditions as the signals from one strategy become optimal over another. Our work also highlights that the traditional paradigm of cross-sectional exploration versus exploitation is not suitable for our financial setting. Since financial datasets report on all publicly traded securities at every period, the counterfactual of not selecting an action is observed. The rewards from a given strategy can be calculated retroactively despite not taking the action. Instead, we have to consider exploration as having value in a time-series sense of investigating changing distributional properties versus exploitation of estimated ones. Traditional exploration of exploitation might be relevant when rebalancing the portfolio affects the price of the underlying securities. In this case, exploration plays a role in estimating the opposing market’s value function over a period of time.

On average, any multi-armed bandit that had a degree of exploration in our experiments underperformed relative to agents that strictly exploited the estimated best action at every time step. This observation was prevalent across the four policies we investigated, but there did exist cases that were evident outliers where the agent did explore in key periods into the optimal allocation to maximize the rewards. Since our work focused on a limited range of allocations, the success of these outliers would be further minimized when increasing the set of possible allocations to a continuous scale. While the multi-armed bandits did approach the optimal strategy, a higher rate of exploration tended to deviate the agent further from optimum.

More importantly, we investigated stationary and non-stationary bandits to observe whether a shorter-term outlook to rewards yields higher returns. In the case of looking at P&L returns, the non-stationary bandit outperformed the stationary bandit at lower reward windows, suggesting that recent strategy performance is indicative of superior performance to follow. Interestingly, when considering the Sharpe-based returns, the bandit with a stationary policy quickly identifies the best-performing model to outperform the non-stationary bandit which appears to frequently select sub-optimal arms.

While our findings suggest that a multi-armed bandit approach to portfolio	management has the benefit of balancing between competing investment strategies, these experiments were done in a closed environment that did not directly interact with the market and did not take into account transaction costs. As previously noted, our trades are not affecting the price of our underlying funds and our results in our paper trading environment may not directly translate in a live setting. This work is not financial advice and readers should consult with a financial advisor prior to making any financial decisions.

## 6. Future Work
Within the domain of finance, no experiment is ever complete without validating a proposed model in a live trading setting. Our multi-armed bandit is not interacting with the environment and not receiving a true representation of market feedback. We hope to continue our work and implement our multi-armed bandit as an overlay to our underlying supervised strategies. Future iterations of this work should also embody a more robust construction of the portfolio by adding a wider selection of securities. Signals from the supervised learning models should also be adapted to point to a more diverse and dynamic set of portfolio allocations. Lastly, future work can consider the effect of adding additional competitive arms to the multi-armed bandit to cumulative returns.

## 7. Statement of Work
Both Rodrigo Erices and Vlad Bardalez contributed to all aspects of the project.

## 8. References
Asness, C. et al. (2013). Value and Momentum Everywhere. Journal of Finance. 68(3). https://pages.stern.nyu.edu/~lpederse/papers/ValMomEverywhere.pdf

Galbraith (2016) Python library for Multi-Armed Bandits [Source code]. https://github.com/bgalbraith/bandits.

Kho, J. (accessed online 2018), Why Random Forest is My Favorite Machine Learning Model. Towards Data Science. https://towardsdatascience.com/why-random-forest-is-my-favorite-machine-learning-model-b97651fa3706

Korgaonkar, R. et al (accessed online 2022), TargetRisk: Aiming to Protect and Grow Your Money. MAN AHL’s Website. https://www.man.com/ahl-targetrisk

Panigirtzoglou, N. et al. (2019). Flows and Liquidity: Dollar liquidity tightening. J.P. Morgan Markets. April 26th.
