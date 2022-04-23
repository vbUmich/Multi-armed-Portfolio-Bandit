# Multi-armed Portfolio Bandit

## 1. Introduction

Active asset managers are commonly faced with the dilemma of selecting competing investment strategies as market conditions evolve to favor one strategy over a set of others. Reinforcement learning offers a solution to supplement established strategies by balancing  between the exploration and exploitation of new and known knowledge. A reinforcement learning agent can take actions within an environment and maximize a defined cumulative reward. In our application to optimal portfolio execution, we employ a multi-armed bandit, one of the classical reinforcement learning problems, to balance reward maximization of a portfolio. Instead of a multi-armed bandit balancing between investing in securities or funds directly, our approach treats each underlying signal from a given model as an independent strategy that the agent can act on. 

## 2. Conceptual Framework

### 2.1 General Framework 
Our setting consists of a portfolio manager in charge of the allocation of a balanced portfolio (i.e. investing in equities and fixed income securities), where he has to choose between the recommendations of different investment strategies every week. Different strategies can suggest different competing allocations, and might be performing differently through time, as is typical in real-world practice. 

We model the portfolio manager as a reinforcement learning agent (in particular a multi-armed bandit agent), with an action space consisting of recommendations based on supervised learning models (in particular random forest classifiers of short-term market direction). 

In this setting, the main task of the RL agent is to perform exploitation of the estimated best performing strategy, while balancing the exploration of possible changing underlying distributional conditions of the market that impact the performance of the strategies. The main task of the supervised models is to beat the market as best as they can, although given the nature of financial markets, their out-of-sample performance is always called into question. Our expectation is that the RL agent adds value by keeping in check possible drifts in performance of the supervised models in a real-world setting. 

### 2.2 Reinforcement Learning Framework

**Agents and Policies**

The reinforcement learning agent in our case is a Multi-Armed Bandit (MAB) which we solved by using different policies. We relied on several variations of epsilon-Greedy strategies.

- Variation on Exploration Intensity: we used several parameters for epsilon, ranging from... 
- Variation on Exploitation Profile: we used a parameter for decaying the epsilon parameter over time. 
- Variation on Stationariety: we constructed two non-stationary versions for the estimation of mean reward of actions by 
  - Considering only a historical buffer of past rewards to calculate mean rewards,
  - Considering an exponentially decreasing mean reward to calculate mean reward.

It is extremely important to comment on a relevant tweak to a normal MAB problem. In a general financial setting, involving asset prices that evolve independently of the actions of the agent, each time our agent takes an action, he observes the rewards associated with that action, but also observes the counterfactual rewards of the other actions, as such, it is relevant to update the estimates not only of the actioned strategy, but also of the others. In this setting, as discussed previously, there is no exploration-exploitation dilemma cross-sectionally versus the strategies, but instead this happens over time, as it might be worthwhile to explore different strategies over time as their underlying distributional properties might have changed. 

**Action Space**

The action space is detailed in the next section, for the time being suffice it to say they correspond to portfolio allocations as recommended by specific investment strategies based on supervised learning models.

**Rewards**

Our MAB aims to maximize the cumulative sum of the rewards observed through the whole experiment. We consider two distinct reward-generating functions that serve different purposes from a financial perspective: the first is constructed to measure profit and loss, the second one measures risk-adjusted profit and loss (Sharpe ratio).

Portfolio Profit and Loss: at each time step it calculates the return of the recommended portfolio.
Portfolio Risk-Adjusted Profit and Loss: at each time step it calculates the return of the recommended portfolio, divided by the expected ex-ante standard deviation of the recommended portfolio. This standard deviation is calculated by assuming the historical covariances between the equity and fixed income stay the same.

<p align="center">
  <img src="https://github.com/vbUmich/Multi-armed-Portfolio-Bandit/blob/main/docs/images/model_1_freq_counts.png">
</p>
<p align="center">
  <b>Figure 1</b>
</p>

<p align="center">
  <img src="https://github.com/vbUmich/Multi-armed-Portfolio-Bandit/blob/main/docs/images/model_2_freq_counts.png">
</p>
<p align="center">
  <b>Figure 2</b>
</p>

<p align="center">
  <img src="https://github.com/vbUmich/Multi-armed-Portfolio-Bandit/blob/main/docs/images/supervised_confusion_matrices.PNG">
</p>
<p align="center">
  <b>Table 1 - Supervised Model Confusion Matrices and Aggregated Metrics</b>
</p>

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

<p align="center">
  <img src="https://github.com/vbUmich/Multi-armed-Portfolio-Bandit/blob/main/docs/images/policy_comparison_pl.PNG">
</p>
<p align="center">
  <b>Table 2</b>
</p>

<p align="center">
  <img src="https://github.com/vbUmich/Multi-armed-Portfolio-Bandit/blob/main/docs/images/eg_0_rel_returns_action_counts_pl.png">
</p>
<p align="center">
  <b>Figure 5</b>
</p>

<p align="center">
  <img src="https://github.com/vbUmich/Multi-armed-Portfolio-Bandit/blob/main/docs/images/nseg_0_rel_return_action_counts_pl.png">
</p>
<p align="center">
  <b>Figure 6</b>
</p>

<p align="center">
  <img src="https://github.com/vbUmich/Multi-armed-Portfolio-Bandit/blob/main/docs/images/policy_comparison_sharpe.PNG">
</p>
<p align="center">
  <b>Table 3</b>
</p>

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

***
***
***
You can use the [editor on GitHub](https://github.com/vbUmich/Multi-armed_Portfolio_Bandit/edit/main/docs/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/vbUmich/Multi-armed_Portfolio_Bandit/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
