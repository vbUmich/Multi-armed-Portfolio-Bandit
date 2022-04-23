# Forest Bandits: An Approach to Dynamic Portfolio Allocation

## Blog Post 

The accompanying blog post for this repository can be found [here](https://vbumich.github.io/Multi-armed-Portfolio-Bandit/).


## Installing packages with pip:

The following command will install all packages listed in `requirements.txt`.

```
pip install -r requirements.txt
```

## Understanding the Structure of the Code

There are several Jupyter notebooks that comprise the project. However, the main analysis is stored in just one notebook. The other notebooks are there to provide data inputs to be used in the main file.

### Main Code: 'Multi-armed Portfolio Bandit.ipynb'
This notebook is the core of our Multi-armed Bandit (MAB) analysis. The notebooks is seperated into different sections five sections:

- Setup - Set the correct file path
- Class Structure - The Multi-armed Portfolio Bandit Framework
- Data Preparation - Data preprocessing and final dataframe creation
- Model Analysis - Comparing performance of underlying supervised models
- Multi-armed Bandit Experiment - Conducting the MAB experiments with different configuarionts

### Supporting Code

#### 'SupervisedModel1.ipynb'
The objective of this code is to generate the sequence of signals (-1, 0, 1) that represent an investment strategy. Those signals are indicative of a recommendation to either decrease, maintain or increase portfolio allocations to equity.
The signals are the output classification of a random forest model that is trained to classify next-period S&P returns above, between or below a certain threshold.
The inspiration of this model, and the features utilized come from the work of Nikolaos Panigirtzoglou (2019) through his recurring publications 'Flows and Liqudity' for JPMorgan.

#### 'SupervisedModel2.ipynb'
The objective of this code is to generate the sequence of signals (-1, 0, 1) that represent an investment strategy. Those signals are indicative of a recommendation to either decrease, maintain or increase portfolio allocations to equity.
The signals are the output classification of a random forest model that is trained to classify next-period S&P returns above, between or below a certain threshold.
The inspiration of this model, and the features utilized come from the work of the hedge fund AHL of MAN Group. More information can be found on the webpage of their TargetRisk strategies.https://www.man.com/ahl-targetrisk

#### 'ForesightStrategy.ipynb'
The purpose of this notebook is to create signals that predict the optimal allocation a determined percent of the time. These generated signals will allow us to test whether the mutli-armed bandit is correctly recognizing and adapting to the best strategy.

#### 'CRSP_FundSelection.ipynb' 
The purpose of this notebook was to explore the Center for Research in Security Prices (CRSP) mutual fund dataset accessed through Wharton Research Data Services (WRDS) and select funds that were used in our portfolio.'

#### 'GetWeeklyFundReturns.ipynb'
The purpose of this notebook is to create a weekly returns CSV file for all the funds in our portfolio that can be easily used in our multi-armed bandit

#### 'IntradayData.ipynb'
The purpose of this notebook is to obtain the data needed to calculate the intraday-correlation feature for the Supervised Model 2. This is one of the trickiest features to obtain, as common sources only have short histories of intraday data (Bloomberg typically houses about 6months worth of intraday data). However, for the purpose of the model, several years are needed. 
We source the intraday data from the TAQ database of Wharton Research Data Services (WRDS), and the code requires a login credentials.
The code proceeds in two steps. The first one downloads to separate csv files the relevant intraday trades for several years (one at a time, given query size) while at the same time resampling it to a 1-minute frequency. The second step loads all of these csv files, merges them together and calculates a continues time series of mean intraday correlation, which is outputed as a separate csv file to be used as a feature.

## Running the Code

### To run the main analysis:
1. The main code runs on its own. It relyies on already stored inputs in the repository.

### To recalculate inputs:

#### Supervised Models
1. Run 'IntradayData.ipynb'. This file outputs 'intraday.csv' which is required by 'SupervisedModel2.ipynb'
2. Run 'SupervisedModel1.ipynb' - This file outputs 'supervisedmodel1.csv' which is required by the main model
3. Run 'SupervisedModel2.ipynb' - This file outputs 'supervisedmodel1.csv' which is required by the main model

#### Other Inputs for The Main File
1. Run 'GetWeeklyFundReturns.ipynb' - This file outputs 'selectedFundWeeklyReturns.csv' which is used by the main model and the next file
2. Run 'ForesightStrategy.ipynb' - This file outputs 'foresight_20_signal.csv' which is used by the main model

### Supplementary Code
1. Run 'CRSP_FundSelection.ipynb' - This file does not generate outputs, but contains the analysis to select the funds that were used in the rest of the project. 
5. Run 'ForesightStrategy.ipynb' - This file outputs 'supervisedmodel1.csv' which is required by the main model

***

Made by Rodrigo Erices and Vlad Bardalez
