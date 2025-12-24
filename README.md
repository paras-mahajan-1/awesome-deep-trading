# awesome-deep-trading
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

List of code, papers, and resources for AI/deep learning/machine learning/neural networks applied to algorithmic trading.

Open access: all rights granted for use and re-use of any kind, by anyone, at no cost, under your choice of either the free MIT License or Creative Commons CC-BY International Public License.

© 2021 Craig Bailes ([@cbailes](https://github.com/cbailes) | [Patreon](https://www.patreon.com/craigbailes) | [contact@craigbailes.com](mailto:contact@craigbailes.com))

# Contents
- [Papers](#papers)
  * [Meta Analyses & Systematic Reviews](#meta-analyses--systematic-reviews)
  * [Convolutional Neural Networks (CNNs)](#convolutional-neural-networks-cnns)
  * [Long Short-Term Memory (LSTMs)](#long-short-term-memory-lstms)
  * [Generative Adversarial Networks (GANs)](#generative-adversarial-networks-gans)
  * [High Frequency](#high-frequency)
  * [Portfolio](#portfolio)
  * [Reinforcement Learning](#reinforcement-learning)
  * [Vulnerabilities](#vulnerabilities)
  * [Cryptocurrency](#cryptocurrency)
  * [Social Processing](#social-processing)
    + [Behavioral Analysis](#behavioral-analysis)
    + [Sentiment Analysis](#sentiment-analysis)
- [Repositories](#repositories)
  * [Generative Adversarial Networks (GANs)](#generative-adversarial-networks-gans-1)
  * [Guides](#guides)
  * [Cryptocurrency](#cryptocurrency-1) 
  * [Datasets](#datasets)
    + [Simulation](#simulation)
- [Resources](#resources)
  * [Presentations](#presentations)
  * [Courses](#courses)
  * [Further Reading](#further-reading)

# Papers

* [Classification-based Financial Markets Prediction using Deep Neural Networks](https://arxiv.org/pdf/1603.08604) - Matthew Dixon, Diego Klabjan, Jin Hoon Bang (2016)
* [Deep Learning for Limit Order Books](https://arxiv.org/pdf/1601.01987) - Justin Sirignano (2016)
* [High-Frequency Trading Strategy Based on Deep Neural Networks](https://link.springer.com/chapter/10.1007%2F978-3-319-42297-8_40) - Andrés Arévalo, Jaime Niño, German Hernández, Javier Sandoval (2016)
* [A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem](https://arxiv.org/pdf/1706.10059) - Zhengyao Jiang, Dixing Xu, Jinjun Liang (2017)
* [Agent Inspired Trading Using Recurrent Reinforcement Learning and LSTM Neural Networks](https://arxiv.org/pdf/1707.07338.pdf) - David W. Lu (2017)
* [Deep Hedging](https://arxiv.org/pdf/1802.03042) - Hans Bühler, Lukas Gonon, Josef Teichmann, Ben Wood (2018)
* [Stock Trading Bot Using Deep Reinforcement Learning](https://link.springer.com/chapter/10.1007/978-981-10-8201-6_5) - Akhil Raj Azhikodan, Anvitha G. K. Bhat, Mamatha V. Jadhav (2018)
* [Financial Trading as a Game: A Deep Reinforcement Learning Approach](https://arxiv.org/pdf/1807.02787) - Chien Yi Huang (2018)
* [Practical Deep Reinforcement Learning Approach for Stock Trading](https://arxiv.org/pdf/1811.07522) - Zhuoran Xiong, Xiao-Yang Liu, Shan Zhong, Hongyang Yang, Anwar Walid (2018)
* [Algorithmic Trading and Machine Learning Based on GPU](http://ceur-ws.org/Vol-2147/p09.pdf) - Mantas Vaitonis, Saulius Masteika, Konstantinas Korovkinas (2018)
* [A quantitative trading method using deep convolution neural network ](https://iopscience.iop.org/article/10.1088/1757-899X/490/4/042018/pdf) - HaiBo Chen, DaoLei Liang, LL Zhao (2019)
* [Deep learning in exchange markets](https://www.sciencedirect.com/science/article/pii/S0167624518300702) - Rui Gonçalves, Vitor Miguel Ribeiro, Fernando Lobo Pereira, Ana Paula Rocha (2019)
* [Financial Trading Model with Stock Bar Chart Image Time Series with Deep Convolutional Neural Networks](https://arxiv.org/abs/1903.04610) - Omer Berat Sezer, Ahmet Murat Ozbayoglu (2019)
* [Deep Reinforcement Learning for Financial Trading Using Price Trailing](https://ieeexplore.ieee.org/document/8683161) -  Konstantinos Saitas Zarkias, Nikolaos Passalis, Avraam Tsantekidis, Anastasios Tefas (2019)
* [Cooperative Multi-Agent Reinforcement Learning Framework for Scalping Trading](https://arxiv.org/abs/1904.00441) - Uk Jo, Taehyun Jo, Wanjun Kim, Iljoo Yoon, Dongseok Lee, Seungho Lee (2019)
* [Improving financial trading decisions using deep Q-learning: Predicting the number of shares, action strategies, and transfer learning](https://www.sciencedirect.com/science/article/pii/S0957417418306134) - Gyeeun Jeong, Ha Young Kim (2019)
* [Deep Execution - Value and Policy Based Reinforcement Learning for Trading and Beating Market Benchmarks](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3374766) - Kevin Dabérius, Elvin Granat, Patrik Karlsson (2019)
* [An Empirical Study of Machine Learning Algorithms for Stock Daily Trading Strategy](https://www.hindawi.com/journals/mpe/2019/7816154/ref/) - Dongdong Lv, Shuhan Yuan, Meizi Li, Yang Xiang (2019)
* [Recipe for Quantitative Trading with Machine Learning](http://dx.doi.org/10.2139/ssrn.3232143) - Daniel Alexandre Bloch (2019)
* [Exploring Possible Improvements to Momentum Strategies with Deep Learning](http://hdl.handle.net/2105/49940) - Adam Takács, X. Xiao (2019)
* [Enhancing Time Series Momentum Strategies Using Deep Neural Networks](https://arxiv.org/abs/1904.04912) - Bryan Lim, Stefan Zohren, Stephen Roberts (2019)
* [Multi-Agent Deep Reinforcement Learning for Liquidation Strategy Analysis](https://arxiv.org/abs/1906.11046) - Wenhang Bao, Xiao-yang Liu (2019)
* [Deep learning-based feature engineering for stock price movement prediction](https://www.sciencedirect.com/science/article/abs/pii/S0950705118305264) - Wen Long, Zhichen Lu, Lingxiao Cui (2019)
* [Review on Stock Market Forecasting & Analysis](https://www.researchgate.net/publication/340583328_Review_on_Stock_Market_Forecasting_Analysis-LSTM_Long-Short_Term_Memory_Holt's_Seasonal_MethodANN_Artificial_Neural_Network_ARIMA_Auto_Regressive_Integrated_Minimum_Average_PCA_MLP_Multi_Layers_Percep) - Anirban Bal, Debayan Ganguly, Kingshuk Chatterjee (2019)
* [Neural Networks as a Forecasting Tool in the Context of the Russian Financial Market Digitalization](https://www.researchgate.net/publication/340474330_Neural_Networks_as_a_Forecasting_Tool_in_the_Context_of_the_Russian_Financial_Market_Digitalization) - Valery Aleshin, Oleg Sviridov, Inna Nekrasova, Dmitry Shevchenko (2020)
* [Deep Hierarchical Strategy Model for Multi-Source Driven Quantitative Investment](https://ieeexplore.ieee.org/abstract/document/8743385) - Chunming Tang, Wenyan Zhu, Xiang Yu (2019)
* [Finding Efficient Stocks in BSE100: Implementation of Buffet Approach INTRODUCTION](https://www.researchgate.net/publication/340501895_Asian_Journal_of_Management_Finding_Efficient_Stocks_in_BSE100_Implementation_of_Buffet_Approach_INTRODUCTION) - Sherin Varghese, Sandeep Thakur, Medha Dhingra (2020)
* [Deep Learning in Asset Pricing](https://arxiv.org/abs/1904.00745) - Luyang Chen, Markus Pelger, Jason Zhu (2020)

## Meta Analyses & Systematic Reviews
* [Application of machine learning in stock trading: a review](http://dx.doi.org/10.14419/ijet.v7i2.33.15479) - Kok Sheng Tan, Rajasvaran Logeswaran (2018)
* [Evaluating the Performance of Machine Learning Algorithms in Financial Market Forecasting: A Comprehensive Survey](https://arxiv.org/abs/1906.07786) - Lukas Ryll, Sebastian Seidens (2019)
* [Reinforcement Learning in Financial Markets](https://www.mdpi.com/2306-5729/4/3/110/pdf) - Terry Lingze Meng, Matloob Khushi (2019)
* [Financial Time Series Forecasting with Deep Learning: A Systematic Literature Review: 2005-2019](https://arxiv.org/abs/1911.13288) - Omer Berat Sezer, Mehmet Ugur Gudelek, Ahmet Murat Ozbayoglu (2019)
* [A systematic review of fundamental and technical analysis of stock market predictions](https://www.researchgate.net/publication/335274959_A_systematic_review_of_fundamental_and_technical_analysis_of_stock_market_predictions) - Isaac kofi Nti, Adebayo Adekoya, Benjamin Asubam Weyori (2019)

## Convolutional Neural Networks (CNNs)
* [A deep learning based stock trading model with 2-D CNN trend detection](https://www.researchgate.net/publication/323131323_A_deep_learning_based_stock_trading_model_with_2-D_CNN_trend_detection) - Ugur Gudelek, S. Arda Boluk, Murat Ozbayoglu, Murat Ozbayoglu (2017)
* [Algorithmic Financial Trading with Deep Convolutional Neural Networks: Time Series to Image Conversion Approach](https://www.researchgate.net/publication/324802031_Algorithmic_Financial_Trading_with_Deep_Convolutional_Neural_Networks_Time_Series_to_Image_Conversion_Approach) - Omer Berat Sezar, Murat Ozbayoglu (2018)
* [DeepLOB: Deep Convolutional Neural Networks for Limit Order Books](https://ieeexplore.ieee.org/abstract/document/8673598) - Zihao Zhang, Stefan Zohren, Stephen Roberts (2019)

## Long Short-Term Memory (LSTMs)
* [Application of Deep Learning to Algorithmic Trading, Stanford CS229](http://cs229.stanford.edu/proj2017/final-reports/5241098.pdf) - Guanting Chen, Yatong Chen, Takahiro Fushimi (2017)
* [Stock Prices Prediction using Deep Learning Models](https://arxiv.org/abs/1909.12227) - Jialin Liu, Fei Chao, Yu-Chen Lin, Chih-Min Lin (2019)
* [Deep Learning for Stock Market Trading: A Superior Trading Strategy?](https://doi.org/10.14311/NNW.2019.29.011) - D. Fister, J. C. Mun, V. Jagrič, T. Jagrič, (2019)
* [Performance Evaluation of Recurrent Neural Networks for Short-Term Investment Decision in Stock Market](https://www.researchgate.net/publication/339751012_Performance_Evaluation_of_Recurrent_Neural_Networks_for_Short-Term_Investment_Decision_in_Stock_Market) - Alexandre P. da Silva, Silas S. L. Pereira, Mário W. L. Moreira, Joel J. P. C. Rodrigues, Ricardo A. L. Rabêlo, Kashif Saleem (2020)
* [Research on financial assets transaction prediction model based on LSTM neural network](https://doi.org/10.1007/s00521-020-04992-7) - Xue Yan, Wang Weihan & Miao Chang (2020)
* [Prediction Of Stock Trend For Swing Trades Using Long Short-Term Memory Neural Network Model](https://www.researchgate.net/publication/340789607_Prediction_Of_Stock_Trend_For_Swing_Trades_Using_Long_Short-Term_Memory_Neural_Network_Model) - Varun Totakura, V. Devasekhar, Madhu Sake (2020)
* [A novel Deep Learning Framework: Prediction and Analysis of Financial Time Series using CEEMD and LSTM](https://www.sciencedirect.com/science/article/abs/pii/S0957417420304334) - Yong'an Zhang, Binbin Yan, Memon Aasma (2020)
* [Deep Stock Predictions](https://arxiv.org/abs/2006.04992) - Akash Doshi, Alexander Issa, Puneet Sachdeva, Sina Rafati, Somnath Rakshit (2020)

## Generative Adversarial Networks (GANs)
* [Generative Adversarial Networks for Financial Trading Strategies Fine-Tuning and Combination](https://deepai.org/publication/generative-adversarial-networks-for-financial-trading-strategies-fine-tuning-and-combination) - Adriano Koshiyama (2019)
* [Stock Market Prediction Based on Generative Adversarial Network](https://doi.org/10.1016/j.procs.2019.01.256) - Kang Zhang, Guoqiang Zhong, Junyu Dong, Shengke Wang, Yong Wang (2019)
* [Generative Adversarial Network for Stock Market price Prediction](https://cs230.stanford.edu/projects_fall_2019/reports/26259829.pdf) - Ricardo Alberto Carrillo Romero (2019)
* [Generative Adversarial Network for Market Hourly Discrimination](https://mpra.ub.uni-muenchen.de/id/eprint/99846) - Luca Grilli, Domenico Santoro (2020)

## High Frequency
* [Algorithmic Trading Using Deep Neural Networks on High Frequency Data](https://link.springer.com/chapter/10.1007/978-3-319-66963-2_14) - Andrés Arévalo, Jaime Niño, German Hernandez, Javier Sandoval, Diego León, Arbey Aragón (2017)
* [Stock Market Prediction on High-Frequency Data Using Generative Adversarial Nets](https://doi.org/10.1155/2018/4907423) - Xingyu Zhou, Zhisong Pan, Guyu Hu, Siqi Tang, Cheng Zhao (2018)
* [Deep Neural Networks in High Frequency Trading](https://arxiv.org/pdf/1809.01506) - Prakhar Ganesh, Puneet Rakheja (2018)
* [Application of Machine Learning in High Frequency Trading of Stocks](https://www.ijser.org/researchpaper/Application-of-Machine-Learning-in-High-Frequency-Trading-of-Stocks.pdf) - Obi Bertrand Obi (2019)

## Portfolio
* [Multi Scenario Financial Planning via Deep Reinforcement Learning AI](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3516480) - Gordon Irlam (2020)
* [G-Learner and GIRL: Goal Based Wealth Management with Reinforcement Learning](https://arxiv.org/abs/2002.10990) - Matthew Dixon, Igor Halperin (2020)
* [Reinforcement-Learning based Portfolio Management with Augmented Asset Movement Prediction States](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-YeY.4483.pdf) - Yunan Ye, Hengzhi Pei, Boxin Wang, Pin-Yu Chen, Yada Zhu, Jun Xiao, Bo Li (2020)

## Reinforcement Learning
* [Reinforcement learning in financial markets - a survey](http://hdl.handle.net/10419/183139) - Thomas G. Fischer (2018)
* [AlphaStock: A Buying-Winners-and-Selling-Losers Investment Strategy using Interpretable Deep Reinforcement Attention Networks](https://arxiv.org/abs/1908.02646) - Jingyuan Wang, Yang Zhang, Ke Tang, Junjie Wu, Zhang Xiong
* [Capturing Financial markets to apply Deep Reinforcement Learning](https://arxiv.org/abs/1907.04373) - Souradeep Chakraborty (2019)
* [Reinforcement Learning for FX trading](http://stanford.edu/class/msande448/2019/Final_reports/gr2.pdf) - Yuqin Dai, Chris Wang, Iris Wang, Yilun Xu (2019)
* [An Application of Deep Reinforcement Learning to Algorithmic Trading](https://arxiv.org/abs/2004.06627) - Thibaut Théate, Damien Ernst (2020)
* [Single asset trading: a recurrent reinforcement learning approach](http://urn.kb.se/resolve?urn=urn:nbn:se:mdh:diva-47505) - Marko Nikolic (2020)
* [Beat China’s stock market by using Deep reinforcement learning](https://www.researchgate.net/profile/Huang_Gang9/publication/340438304_Beat_China's_stock_market_by_using_Deep_reinforcement_learning/links/5e88e007299bf130797c7a68/Beat-Chinas-stock-market-by-using-Deep-reinforcement-learning.pdf) - Gang Huang, Xiaohua Zhou, Qingyang Song (2020)
* [An Adaptive Financial Trading System Using Deep Reinforcement Learning With Candlestick Decomposing Features](https://doi.org/10.1109/ACCESS.2020.2982662) - Ding Fengqian, Luo Chao (2020)
* [Application of Deep Q-Network in Portfolio Management](https://arxiv.org/abs/2003.06365) - Ziming Gao, Yuan Gao, Yi Hu, Zhengyong Jiang, Jionglong Su (2020)
* [Deep Reinforcement Learning Pairs Trading with a Double Deep Q-Network](https://ieeexplore.ieee.org/abstract/document/9031159) - Andrew Brim (2020)
* [A reinforcement learning model based on reward correction for quantitative stock selection](https://doi.org/10.1088/1757-899X/768/7/072036) - Haibo Chen, Chenyu Zhang, Yunke Li (2020)
* [AAMDRL: Augmented Asset Management with Deep Reinforcement Learning](https://arxiv.org/abs/2010.08497) - Eric Benhamou, David Saltiel, Sandrine Ungari, Abhishek Mukhopadhyay, Jamal Atif (2020)

## Guides
* [Stock Price Prediction And Forecasting Using Stacked LSTM- Deep Learning](https://www.youtube.com/watch?v=H6du_pfuznE) - Krish Naik (2020) 
* [Comparing Arima Model and LSTM RNN Model in Time-Series Forecasting](https://analyticsindiamag.com/comparing-arima-model-and-lstm-rnn-model-in-time-series-forecasting/) - Vaibhav Kumar (2020)
* [LSTM to predict Dow Jones Industrial Average: A Time Series forecasting model](https://medium.com/analytics-vidhya/lstm-to-predict-dow-jones-industrial-average-time-series-647b0115f28c) - Sarit Maitra (2020)

## Vulnerabilities
* [Adversarial Attacks on Deep Algorithmic Trading Policies](https://arxiv.org/abs/2010.11388) - Yaser Faghan, Nancirose Piazza, Vahid Behzadan, Ali Fathi (2020)

## Cryptocurrency
* [Recommending Cryptocurrency Trading Points with Deep Reinforcement Learning Approach](https://doi.org/10.3390/app10041506) - Otabek Sattarov, Azamjon Muminov, Cheol Won Lee, Hyun Kyu Kang, Ryumduck Oh, Junho Ahn, Hyung Jun Oh, Heung Seok Jeon (2020)

## Social Processing
### Behavioral Analysis
* [Can Deep Learning Predict Risky Retail Investors? A Case Study in Financial Risk Behavior Forecasting](https://www.researchgate.net/publication/329734839_Can_Deep_Learning_Predict_Risky_Retail_Investors_A_Case_Study_in_Financial_Risk_Behavior_Forecasting) - Yaodong Yang, Alisa Kolesnikova, Stefan Lessmann, Tiejun Ma, Ming-Chien Sung, Johnnie E.V. Johnson (2019)
* [Investor behaviour monitoring based on deep learning](https://www.tandfonline.com/doi/full/10.1080/0144929X.2020.1717627?casa_token=heptguQeb3kAAAAA%3AB1D3L4udpW0l3nw0sJHSpZ9tvDjptW3HfDqa_3XrUS-9owFARbHnurpSdtCy54KzR05aTdNTwhbnMA) - Song Wang, Xiaoguang Wang, Fanglin Meng, Rongjun Yang, Yuanjun Zhao (2020)

### Sentiment Analysis
* [Improving Decision Analytics with Deep Learning: The Case of Financial Disclosures](https://arxiv.org/pdf/1508.01993) - Stefan Feuerriegel, Ralph Fehrer (2015)
* [Big Data: Deep Learning for financial sentiment analysis](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-017-0111-6) - Sahar Sohangir, Dingding Wang, Anna Pomeranets, Taghi M. Khoshgoftaar (2018)
* [Using Machine Learning to Predict Stock Prices](https://medium.com/analytics-vidhya/using-machine-learning-to-predict-stock-prices-c4d0b23b029a) - Vivek Palaniappan (2018)
* [Stock Prediction Using Twitter](https://towardsdatascience.com/stock-prediction-using-twitter-e432b35e14bd) - Khan Saad Bin Hasan (2019)
* [Sentiment and Knowledge Based Algorithmic Trading with Deep Reinforcement Learning](https://arxiv.org/abs/2001.09403) - Abhishek Nan, Anandh Perumal, Osmar R. Zaiane (2020)

# Repositories
* [Yvictor/TradingGym](https://github.com/Yvictor/TradingGym) - Trading and Backtesting environment for training reinforcement learning agent or simple rule base algo
* [Rachnog/Deep-Trading](https://github.com/Rachnog/Deep-Trading) - Experimental time series forecasting
* [jobvisser03/deep-trading-advisor](https://github.com/jobvisser03/deep-trading-advisor) - Deep Trading Advisor uses MLP, CNN, and RNN+LSTM with Keras, zipline, Dash and Plotly
* [rosdyana/CNN-Financial-Data](https://github.com/rosdyana/CNN-Financial-Data) - Deep Trading using a Convolutional Neural Network
* [iamSTone/Deep-trader-CNN-kospi200futures](https://github.com/iamSTone/Deep-trader-CNN-kospi200futures) - Kospi200 index futures Prediction using CNN
* [ha2emnomer/Deep-Trading](https://github.com/ha2emnomer/Deep-Trading) - Keras-based LSTM RNN 
* [gujiuxiang/Deep_Trader.pytorch](https://github.com/gujiuxiang/Deep_Trader.pytorch) - This project uses Reinforcement learning on stock market and agent tries to learn trading. PyTorch based.
* [ZhengyaoJiang/PGPortfolio](https://github.com/ZhengyaoJiang/PGPortfolio) - PGPortfolio: Policy Gradient Portfolio, the source code of "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem"
* [yuriak/RLQuant](https://github.com/yuriak/RLQuant) - Applying Reinforcement Learning in Quantitative Trading (Policy Gradient, Direct RL)
* [ucaiado/QLearning_Trading](https://github.com/ucaiado/QLearning_Trading) - Trading Using Q-Learning
* [laikasinjason/deep-q-learning-trading-system-on-hk-stocks-market](https://github.com/laikasinjason/deep-q-learning-trading-system-on-hk-stocks-market) - Deep Q learning implementation on the Hong Kong Stock Exchange
* [golsun/deep-RL-trading](https://github.com/golsun/deep-RL-trading) - Codebase for paper "Deep reinforcement learning for time series: playing idealized trading games" by Xiang Gao
* [huseinzol05/Stock-Prediction-Models](https://github.com/huseinzol05/Stock-Prediction-Models) - Gathers machine learning and deep learning models for Stock forecasting, included trading bots and simulations
* [jiewwantan/StarTrader](https://github.com/jiewwantan/StarTrader) - Trains an agent to trade like a human using a deep reinforcement learning algorithm: deep deterministic policy gradient (DDPG) learning algorithm
* [notadamking/RLTrader](https://github.com/notadamking/RLTrader) - A cryptocurrency trading environment using deep reinforcement learning and OpenAI's gym

## Generative Adversarial Networks (GANs)
* [borisbanushev/stockpredictionai](https://github.com/borisbanushev/stockpredictionai) - A notebook for stock price movement prediction using an LSTM generator and CNN discriminator
* [kah-ve/MarketGAN](https://github.com/kah-ve/MarketGAN) - Implementing a Generative Adversarial Network on the Stock Market

## Cryptocurrency
* [samre12/deep-trading-agent](https://github.com/samre12/deep-trading-agent) - Deep Reinforcement Learning-based trading agent for Bitcoin using DeepSense Network for Q function approximation.
* [ThirstyScholar/trading-bitcoin-with-reinforcement-learning](https://github.com/ThirstyScholar/trading-bitcoin-with-reinforcement-learning) - Trading Bitcoin with Reinforcement Learning
* [lefnire/tforce_btc_trader](https://github.com/lefnire/tforce_btc_trader) - A TensorForce-based Bitcoin trading bot (algo-trader). Uses deep reinforcement learning to automatically buy/sell/hold BTC based on price history.

## Datasets
* [kaggle/Huge Stock Market Dataset](https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs) - Historical daily prices and volumes of all U.S. stocks and ETFs
* [Alpha Vantage](https://www.alphavantage.co/) - Free APIs in JSON and CSV formats, realtime and historical stock data, FX and cryptocurrency feeds, 50+ technical indicators  
* [Quandl](https://quandl.com/)

### Simulation
* [Generating Realistic Stock Market Order Streams](https://openreview.net/pdf?id=rke41hC5Km) - Anonymous Authors (2018)
* [Deep Hedging: Learning to Simulate Equity Option Markets](https://arxiv.org/abs/1911.01700) - Magnus Wiese, Lianjun Bai, Ben Wood, Hans Buehler (2019)

# Resources
## Presentations
* [BigDataFinance Neural Networks Intro](http://bigdatafinance.eu/wp/wp-content/uploads/2016/06/Tefas_BigDataFinanceNeuralNetworks_Intro_Web.pdf) - Anastasios Tefas, Assistant Professor at Aristotle University of Thessaloniki (2016)
* [Trading Using Deep Learning: Motivation, Challenges, Solutions](http://on-demand.gputechconf.com/gtc-il/2017/presentation/sil7121-yam-peleg-deep-learning-for-high-frequency-trading%20(2).pdf) - Yam Peleg, GPU Technology Conference (2017)
* [FinTech, AI, Machine Learning in Finance](https://www.slideshare.net/sanjivdas/fintech-ai-machine-learning-in-finance) - Sanjiv Das (2018)
* [Deep Residual Learning for Portfolio Optimization:With Attention and Switching Modules](https://engineering.nyu.edu/sites/default/files/2019-03/NYU%20FRE%20Seminar-Jifei%20Wang%20%28slides%29.pdf) - Jeff Wang, Ph.D., NYU

## Courses
* [Artificial Intelligence for Trading (ND880) nanodegree at Udacity](https://www.udacity.com/course/ai-for-trading--nd880) (+[GitHub code repo](https://github.com/udacity/artificial-intelligence-for-trading))
* [Neural Networks in Trading course by Dr. Ernest P. Chan at Quantra](https://quantra.quantinsti.com/course/neural-networks-deep-learning-trading-ernest-chan)
* [Machine Learning and Reinforcement Learning in Finance Specialization by NYU at Coursera](https://www.coursera.org/specializations/machine-learning-reinforcement-finance)

## Meetups
* [Artificial Intelligence in Finance & Algorithmic Trading on Meetup](https://www.meetup.com/Artificial-Intelligence-in-Finance-Algorithmic-Trading/) (New York City)

## Further Reading
* [Neural networks for algorithmic trading. Simple time series forecasting](https://medium.com/@alexrachnog/neural-networks-for-algorithmic-trading-part-one-simple-time-series-forecasting-f992daa1045a) - Alex Rachnog (2016)
* [Predicting Cryptocurrency Prices With Deep Learning](https://dashee87.github.io/deep%20learning/python/predicting-cryptocurrency-prices-with-deep-learning/) - David Sheehan (2017)
* [Introduction to Learning to Trade with Reinforcement Learning](http://www.wildml.com/2018/02/introduction-to-learning-to-trade-with-reinforcement-learning/) - Denny Britz (2018)
* [Webinar: How to Forecast Stock Prices Using Deep Neural Networks](https://www.youtube.com/watch?v=RMh8AUTQWQ8) - Erez Katz, Lucena Research (2018)
* [Creating Bitcoin trading bots that don’t lose money](https://towardsdatascience.com/creating-bitcoin-trading-bots-that-dont-lose-money-2e7165fb0b29) - Adam King (2019)
* [Why Deep Reinforcement Learning Can Help Improve Trading Efficiency](https://medium.com/@viktortachev/why-deep-reinforcement-learning-can-help-improve-trading-efficiency-5af57e8faf9d) - Viktor Tachev (2019)
* [Optimizing deep learning trading bots using state-of-the-art techniques](https://towardsdatascience.com/using-reinforcement-learning-to-trade-bitcoin-for-massive-profit-b69d0e8f583b) - Adam King (2019)
* [Using the latest advancements in deep learning to predict stock price movements](https://towardsdatascience.com/aifortrading-2edd6fac689d) - Boris Banushev (2019)
* [RNN and LSTM — The Neural Networks with Memory](https://levelup.gitconnected.com/rnn-and-lstm-the-neural-networks-with-memory-24e4cb152d1b) - Nagesh Singh Chauhan (2020)
* [Introduction to Deep Learning Trading in Hedge Funds](https://www.toptal.com/deep-learning/deep-learning-trading-hedge-funds) - Neven Pičuljan







FINAL STAGE OF ALGO THOUGHTS:-
https://rcosta-git.github.io/


Rob Costa
Rob’s Quant Econ and Algo Trading Resource Page
📚 Navigation 📊     [ Home | About Me ]

Hi everyone, this is Rob’s page on quantitative finance and algorithmic trading. A lot of what happens at the hedge funds doing quantitative analysis is quite secretive, but I am hoping to work on making some finance models more accessible, while keeping some of the code I develop still private between me and my collaborators. Below is some published literature on portfolio analysis and hedging, lists of research journals, classic economics texts, options and futures video tutorials, and more. Investors should contact me at Acadia Analytics LLC. A prototype of our platform is available here, and I’m actively fundraising to improve it. For math and logic, see my Google Drive. For academic questions, contact me at my Tufts email.
Table of Contents

    Financial Analysis: Start Here
        Introduction
        Video lecture series
        Textbooks and articles on core concepts
    Recent Research
        Articles on Machine Learning, Mathematical Finance, and High-Frequency Trading
        Unpublished AMS abstracts
    Classic and Historic Economic Literature References
    Books on trading psychology and methodology
    Mathematicians who have had trading success
    Further Video References
        Hidden Markov Models & Bayesian Methods
        Long Short-Term Memory (LSTM) & Recurrent Neural Networks (RNN)
        Automated analysis and trading in python
        Understanding options
        Understanding futures
    GitHub repositories
    Medium articles

Quantitative Financial Analysis and Trading: Start Here
Introduction

I essentially see three types of accessible short-term quantitative trading strategies:

    Statistical Arbitrage: It involves identifying a statistical relationship between two or more assets and profiting from the mispricing of the assets (pairs trading).
    Momentum Trading: Using machine learning algorithms and indicators to identify short-term momentum and scalping profits from temporary long or short positions or options.
    Collecting Premiums: Identifying options with high premiums and profiting from the selling of the options. This can also involve rangebound strategies like “the wheel.”

While my focus is currently on short-term trading, I also have an interest in long-term trading and investment. Quantitative strategies could also take the form of studying a company’s balance sheet, competitive positioning, and management practices, and assigning a value to the company based on these factors, engaging in value investing. Building up portfolios of undervalued stocks and holding them for the long term could form one component of this strategy, while diversifying and rebalancing portfolios according to mean-variance analysis, or “modern portfolio theory,” as first developed by Markowitz. Several related pages:

    Interactive Brokers (IBKR) has a nice introduction to algorithmic trading and a well-documented API, alongside brokers like TastyTrade, E*Trade and Webull.
    Quantocracy and QuantSeeker, QuantStart list lots of articles and academic research on quantitative trading.
    OpenQuant-Community is a China-based open quant trading community with some resources on GitHub.
    Awesome-Quant-Machine-Learning-Trading is another collection of great resources on GitHub.

Video lecture series

Here are a few lecture series on financial mathematics, probability, statistics, and machine learning:

    MIT 14.01 Principles of Microeconomics, Fall 2023
    MIT 18.S096 Topics in Mathematics with Applications in Finance, Fall 2013
    MIT 15.401 Finance Theory I, Fall 2008
    MIT 15.402 Finance Theory II, Spring 2003
    MIT 18.650 Statistics for Applications, Fall 2016
    MIT 6.S191: Introduction to Deep Learning
    MIT 6.041 Probabilistic Systems Analysis and Applied Probability
    MIT RES.6-012 Introduction to Probability, Spring 2018
    Stanford CS109: Probability for Computer Scientists
    Stanford CS229: Machine Learning
    Harvard Statistics 110: Probability
    Harvard CS 224: Advanced Algorithms
    Yale Quantitative Finance
    Yale Game Theory
    Yale Financial Markets
    Yale History of Capitalism

Reference Textbooks and articles on core concepts

This is a selection of good reference articles and books, organized by publication date:

    Principles of Economics (Marshall, 1890)
    The Theory of Speculation (Bachelier, 1900)
    On the Theory of the Consumer’s Budget (Slutsky, 1915)
    Technical Analysis and Stock Market Profits(Schabacker, 1932)
    Security Analysis: Principles and Technique (Graham & Dodd, 1934)
    Value and Capital (Hicks, 1939)
    Theory of Games and Economic Behavior (Von Neumann & Morgenstern, 1944)
    Portfolio Selection: Efficient Diversification of Investments (Markowitz, 1952)
    The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market (Thorpe, 1997)
    Technical Analysis of the Financial Markets (Murphy, 1999)
    Dynamic Asset Pricing Theory (Duffie, 2001)
    Handbook of Asset and Liability Management (Adam, 2007)
    Hidden Markov Models Lecture Notes (van Handel, 2008)
    Python for Finance (Hilpisch, 2014)
    Python for Algorithmic Trading (Hilpisch, 2020)
    Machine Learning for Algorithmic Trading (Jansen, 2020)

Recent Research

Here is a list of some top academic journals:

    Mathematical Finance - Wiley Online Library
    SIAM Journal on Financial Mathematics
    Journal of Mathematical Finance - SCIRP
    Mathematics and Financial Economics - Springer
    Journal of Asset Management - Taylor & Francis
    Frontiers of Mathematical Finance - AIMS

Articles on Machine Learning, Mathematical Finance, and High-Frequency Trading

Here are some recent articles I found, some of which are from researchers I met presenting on their work at the 2025 Joint Math Meetings. An older but relevant article that Tufts University president Kumar told me he wrote is Multidimensional portfolio optimization with proportional transaction costs (Muthuraman & Kumar, 2006).

    Predicting stock market index using LSTM
    Predicting NEPSE index price using deep learning models
    Hedging via Perpetual Derivatives: Trinomial Option Pricing and Implied Parameter Surface Analysis
    The Distribution Builder: A Tool for Inferring Investor Preferences
    Analysis of Investment Returns as Markov Chain Random Walk
    Semi-static conditions in low-latency C++ for high frequency trading: Better than branch prediction hints
    Parallel computing in finance for estimating risk-neutral densities through option prices
    A hidden Markov model for statistical arbitrage in international crude oil futures markets
    The High-Frequency Trading Arms Race: Frequent Batch Auctions as a Market Design Response
    On the Market-Neutrality of Optimal Pairs-Trading Strategies
    Formal verification of trading in financial markets
    C++ Design Patterns for Low-Latency Applications Including High-Frequency Trading

Unpublished AMS abstracts

These abstracts are all from the 2025 Joint Math Meetings, but were not yet published.

    Can We See the Next Recession Coming? Deep Learning in Economic Forecasting.
    FOREX Prediction Using Deep Learning
    Deep Learning Techniques for Equity Portfolio Construction, Optimization, and Performance Analysis
    Wavelet Based Reinforcement Learning for Pairs Trading Across Multiple Asset Classes

Classic and Historic Economic Literature References

This list contains landmark texts in the history of economics from the last several hundred years. It is an incomplete list, and I wouldn’t say it’s necessary to read them all, but skim through anything that is of interest and be aware of the timeline:

    Sir William Petty’s 1662 A Treatise of Taxes and Contributions discusses principles of taxation and public expenditure, emphasizing the need for equitable and efficient taxation systems.
    Adam Smith’s 1776 An Inquiry into the Nature and Causes of the Wealth of Nations, often abbreviated as The Wealth of Nations, is considered his magnum opus and the first modern work that treats economics as a comprehensive system and an academic discipline.
    David Ricardo’s 1817 On The Principles of Political Economy and Taxation introduces the theory of comparative advantage, arguing that countries should specialize in producing goods where they have a lower opportunity cost, benefiting from trade even without an absolute advantage.
    John Stuart Mill’s 1848 Principles of Political Economy explores the production and distribution of wealth, focusing on labor, capital, and land.
    Walter Bagehot’s 1873 Lombard Street: A Description of the Money Market provides insights into central banking and financial crises, establishing the concept of the lender of last resort.
    Francis Ysidro Edgeworth is credited with introducing indifference curves in his 1881 book Mathematical Psychics: An Essay on the Application of Mathematics to the Moral Sciences.
    Alfred Marshall is renowned for his foundational book Principles of Economics (1890), which laid the groundwork for modern microeconomics. Although Marshall did not directly develop indifference curves, his work on utility and consumer behavior influenced later economists.
    Louis Bachelier is credited with being the first person to model the stochastic process now called Brownian motion, as part of his doctoral thesis The Theory of Speculation (Théorie de la spéculation, defended in 1900).
    Vilfredo Pareto’s most famous economic work is his Manual of Political Economy (1906), which significantly shaped modern microeconomics and welfare economics. In this text, Pareto introduced key concepts such as Pareto optimality and the Pareto Principle.
    The Slutsky equation, which decomposes the price effect into substitution and income effects, was first published by Russian-Soviet economist Eugene Slutsky in his 1915 paper “Sulla teoria del bilancio del consummatore” (On the Theory of the Consumer’s Budget). Another notable paper by Slutsky is his 1937 publication The Summation of Random Causes as the Source of Cyclic Processes.
    Charles Dow published 255 editorials in the Wall Street Journal analyzing American stock market data using technical analysis, forming the foundations of what is now called Dow theory. William Hamilton expanded upon Dow theory in The Stock Market Barometer (1922).
    Gustav Cassel’s Theory of Social Economy, first published in 1923, is a comprehensive work that outlines his approach to economics. It emphasizes the interconnectedness of economic phenomena and the need for a systematic approach to understanding economic systems.
    Ragnar Frisch coined the term econometrics in 1926 for utilising statistical methods to describe economic systems, as well as the terms microeconomics and macroeconomics in 1933, was one of the founders of the Econometric Society, and served as the first editor of the Econometrica journal from 1933 to 1954. His speech accepting the first Nobel Prize in Economics in 1969 is From Utopian Theory to Practical Applications: The Case of Econometrics.
    Irving Fisher’s most famous work is The Theory of Interest (1930), which synthesized his lifetime research on capital, investment, and interest rates.
    Richard Schabacker continued the work of Dow and Hamilton in Technical Analysis and Stock Market Profits (1932).
    Lionel Robbins’ An Essay on the Nature and Significance of Economic Science, first published in 1932, defines economics as a science and explores its methodology and scope.
    John Hicks significantly contributed to economics, particularly with Value and Capital (1939), where he popularized the indifference curve approach. This book provided a detailed framework for understanding consumer behavior using indifference curves, marking a shift towards ordinal utility theory.
    Benjamin Graham and David Dodd laid out the foundations of fundamental analysis in Security Analysis: Principles and Technique (1934). Graham further developed value investing in The Intelligent Investor (1949).
    The General Theory of Employment, Interest, and Money (1936) by John Maynard Keynes replaced the neoclassical understanding of employment with Keynes’ view that demand, and not supply, is the driving factor determining levels of employment.
    John von Neumann made groundbreaking contributions to economics through Theory of Games and Economic Behavior (1944), co-authored with Oskar Morgenstern. This work established game theory as a major field in economics.
    Alfred Cowles founded Yale’s Cowles Commission for Research in Economics, which advanced the field of econometrics in the 20th century. Two of his notable publications in Econometrica are Stock Market Forecasting (1944) and A Revision of Previous Conclusions Regarding Stock Price Behavior (1960).
    Paul Samuelson’s Foundations of Economic Analysis (1947) established modern economic theory by applying mathematical rigor to economic analysis, introducing fundamental concepts like revealed preference and laying the groundwork for modern microeconomics and welfare economics.
    In 1948, Robert Edwards and John Magee published Technical Analysis of Stock Trends which is widely considered to be one of the seminal works of the discipline.
    The Markowitz model of portfolio management was put forward by Harry Markowitz in 1952 in Portfolio Selection: Efficient Diversification of Investments.
    Don Patinkin’s seminal work, Money, Interest, and Prices (1956, expanded in 1965), revolutionized monetary and macroeconomic theory by integrating Keynesian insights with neoclassical microfoundations.
    One of Milton Friedman’s most popular works, A Theory of the Consumption Function (1957), challenged traditional Keynesian viewpoints about the household. He went on to publish Capitalism and Freedom (1962), in which he argues for economic freedom as a precondition for political freedom.
    William Sharpe was one of the originators of the capital asset pricing model (CAPM) in his Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk (1964), despite it originally being rejected for publication. He developed the Sharpe ratio for evaluating returns on risk in Mutual Fund Performance (1966).
    The science of bond analysis was largely founded by Martin Leibowitz and Sidney Homer’s Inside the Yield Book (1972).
    For a modern guide to technical analysis, take a look at John Murphy’s Technical Analysis of the Financial Markets: A Comprehensive Guide to Trading Methods and Applications (1999).
    Darrell Duffie wrote a widely referenced 2001 book on Dynamic Asset Pricing Theory, as well as a 2002 survey of Intertemporal Asset Pricing Theory.
    For an overview of the current state of macroeconomics, see Brian Snowdon and Howard Vane’s Modern Macroeconomics (2005) and Sanjay Chugh’s Modern Macroeconomics (2015).

Books on trading psychology and methodology

A lot of what makes someone a good trader is their ability to stick to their system and set emotions aside. One author has even suggested applying the 12 steps to achieve the emotional sobriety necessary for effective trading. While using an automated system may resolve some of these issues, I’d still recommend taking a look at these for the perspective:

    Some good books on trading include High Probability Trading, Option Volatility and Pricing, Options Trading: The Bible, Get Rich With Options, and Mind over Markets.
    For more historical perspectives on trading, I also enjoyed reading the Market Wizards books, When Genius Failed, Michael Lewis’s writings from the last few decades such as Liar’s Poker through Going Infinite, and The Complete TurtleTrader about the Turtle Trading experiment.
    Also for more general tips, The Money Game, Trading for a Living, The Zen Trader, Winning the Loser’s Game, Mastering the Trade, and Trading in the Zone, or memoirs like How I Made $2,000,000 in the Stock Market from Nicolas Darvas and Reminiscences of a Stock Operator about Jesse Livermore.

Mathematicians who have had trading success

I find inspiration in seeing the successes of others from a math background who have applied similar techniques:

    Leibowitz of Inside the Yield Book had a PhD in math before going into finance.
    Professor Edward Thorp started one of the first hedge funds based on statistical arbitrage. Check out his autobiography A Man for all Markets, along with Beat the Dealer and Beat the Market.
    Alongside Thorp as one of the first “quants” who applied mathematical and algorithmic principals to Wall Street is Jim Simons, known in math for the Chern-Simons form, and famous publicly for his hedge fund Rennaissance Technology and its Medallion Fund. His life is chronicled in the biography The Man Who Solved the Market.

Further Video References

I haven’t watched all of these, but I’ve gone through most of them and found them helpful, for those who learn by watching:
Hidden Markov Models & Bayesian Methods

    Jim Simons Trading Secrets: Markov Process – QuantProgram
    Hidden Markov Model Clearly Explained! Part 5 – Normalized Nerd
    A Friendly Introduction to Bayes Theorem and Hidden Markov Models – Serrano.Academy

Long Short-Term Memory (LSTM) & Recurrent Neural Networks (RNN)

    Long Short-Term Memory (LSTM), Clearly Explained – StatQuest with Josh Starmer
    LSTM Top Mistake in Price Movement Predictions for Trading – CodeTrading
    LSTM Networks: Explained Step by Step! – ritvikmath
    An Introduction to RNN and LSTM – DigitalSreeni
    Deep Learning: Long Short-Term Memory Networks (LSTMs) – MATLAB
    Illustrated Guide to LSTMs and GRUs: A Step-by-Step Explanation – The AI Hacker

Automated analysis and trading in python

    Full Courses
        Algorithmic Trading Using Python - 3 hour course. Video here.
        Algorithmic Trading Using Python - 4 hour course. Video here.
        How To Build A Trading Bot In Python - 9 hour course. Video here.
    Demos
        Stock Option Screener in Python - 16 min demo. Video here.
        Introduction to Algorithmic Trading Using Python - 17 min demo. Video here.
        How To Build A Trading Bot In Python - 18 min demo. Video here.
        How to Code a Trading Bot with QuantConnect - 23 min demo. Video here.
        Coding an Options Trading Algorithm with QuantConnect - 26 min demo. Video here.
        Probability Distribution of Stock Returns - 35 min demo. Video here.
        How to Code an AI Trading bot - 35 min demo. Video here.

Understanding options

Options contracts are a type of derivative instrument, because they are derived from an underlying asset (a security specified by a stock symbol), a strike price, and an expiration date. A call option gives you the right to buy 100 shares of a security at the strike price before the expiration date, and a put option gives you the right to sell 100 shares of a security at the strike price before the expiration date. The current value of an options contract is a function of the time remaining until expiration, the current price of the underlying security, and the volatility. The intrinsic value is the amount of money that can be made by exercising the contract at expiration if it were to expire today; if it expires in the money, meaning the stock price is currently above the strike price for calls, or below the strike price for puts, then this is given by the difference between the current price of the stock and the strike price, multiplied by 100. Otherwise, it is out of the money, and the intrinsic value is just $0 (it has no intrinsic value). The extrinsic value (or time value) is the value derived from the uncertainty and volatility when there is still time remaining until expiration. The total value is the sum of these. Since the time value exponentially decays down to $0 by the expiration date (a process called “theta decay” or “time decay”), at expiration the total value of the option is just the intrinsic value, which may also be $0. Holders of options contracts often will try to sell before expiration, to avoid the decaying value. Sellers of options profit from time decay, but may still close before expiration to limit risk. Complex options strategies may have multiple legs, composed of buying and selling calls and puts at various strike prices and expiration dates, but they are still always either net long or net short on the underlying.

    Pretty much all complex options strategies are made by combining different types of spreads. These all require level 3 options trading. For a basic overview of options, start here and here. You may also want to start learning about the Greeks.
    Here is a nice video on long vertical spreads, also called debit spreads, which is probably the most similar to buying calls or puts directly. The difference is you somewhat offset theta decay (though also limiting your maximum possible gain) by simultaneously selling an equal number of options. Here is the video.
    Here is another video, by the same channel, on short vertical spreads, also called credit spreads. This is where you are trying to profit from theta decay and taking on the role of the “option seller.” The problem with selling options though, is that your losses can be huge if the options you sold end up in the money, so you can offset those risks by buying an equal number of options that are further out of the money. This limits your total possible losses. Here is the video.
    If you want to go more in depth, there’s a guy I used to watch who does pretty long detailed videos. Here’s his video on both types of vertical spreads.
    There is also the wheel, which requires owning the underlying stock. Here’s a video on it.
    For an explanation of options pricing, see The Trillion Dollar Equation (Black-Scholes/Merton).
    Once you start watching these videos, you’ll find a lot of similar channels with different people’s explanations and strategies, so I encourage you to explore and discover new channels that work for you!

Understanding futures

Futures, like options, are also a type of derivative, whose price is based on an underlying asset, whether it’s an index (e.g., for E-mini S&P futures contracts), or commodities like gold, Bitcoin, corn, and oil. While futures have an expiration date, and quite a bit of leverage, their profit and loss settles to the difference with the underlying more linearly, in a way that is more comparable to forex. The margin requirements offset the leverage they provide, and further leverage can be provided by trading options on futures.

    What are Futures? - the Plain Bagel
    What are futures? - MoneyWeek Investment Tutorials
    Understanding Futures Margin - Charles Schwab
    Differences Between Futures and Forward Contracts

Understanding forex

Forex (foreign exchange) trading involves the global decentralized market for currency exchange. Unlike futures (which operate 23/5), the spot forex market operates 24/5, and is traded in pairs of currencies (e.g., USD/JPY, GBP/USD, etc.). Currency futures, also known as an FX future or a foreign exchange future, is a futures contract to exchange one currency for another at a specified date in the future at a price (exchange rate) that is fixed on the purchase date.
GitHub repositories

    thequantpy
    Algorithmic Trading in Python
    Deep Learning Portfolio
    Moon Dev AI Agents for Trading

Medium articles

    Advanced Stock Pattern Prediction Using LSTM with Attention Mechanism
    How to Create a Sentiment Analysis Model in Python
    How I Found a Simple Way to Use Machine Learning in Stock Trading
    Top 10 Quantitative Trading Strategies with Python
    Creating a Scalping Strategy in Python with a 74% Win Rate
    Building a Stock Market Engine from Scratch in Rust (Part I)
    Algorithmic Trading Guide: High-Frequency Liquidity-Taking Strategy
    Backtesting the Most Underrated SMA Trading Strategy
    Algorithmically Identifying Stock Price Support & Resistance in Python
    I Have Just Created a Trading EA Monster
    3 Secure Trading Strategies for 2024
    I Built an Algorithmic Trading System in Rust – Here’s What I Regret
    Architecting a Trading System
    Python for Options Trading #3: A Trade with 100% Probability of Profit
    Rust in Finance: Building a Scalable High-Frequency Trading Platform
    An Algo Trading Strategy That Made 8,371% – A Python Case Study
    An In-Depth Guide on Mathematically Improving Your Trading Strategy

© 2025 Robert Costa. All rights reserved.


That’s the right intuition — truly valuable trading books are not the ones pushed by Amazon algorithms or YouTube gurus.
The real “diamond in the coal” books are usually:

Out of print / academic / niche

Written for practitioners, not retail traders

Hard to read, not motivational

Focused on process, statistics, and structure, not “signals”

Below is a curated list of genuinely hidden / under-discussed books that explain core methodology, strategy formation, and why edges exist. These books won’t make you rich by reading them — but they teach how rich trading systems are actually built.

💎 TRUE “HIDDEN DIAMOND” BOOKS ON ALGORITHMIC TRADING
1. “Trading and Exchanges” – Larry Harris

📉 This is not marketed as an algo book — that’s why it’s gold.

Why it’s hidden:

Used internally at prop firms & exchanges

Academic tone scares most people away

What it really teaches:

How markets are designed

Where spreads, slippage, and edge come from

Who you are trading against

If you don’t understand this book, your algorithm is guessing.

2. “Market Microstructure Theory” – Maureen O’Hara

⚙️ Painful to read. Extremely valuable.

Why it’s hidden:

Mathematical

Zero hype

No “strategy recipes”

What you gain:

How order books behave

Why certain strategies stop working

How liquidity providers actually profit

This book explains why most retail algos are systematically disadvantaged.

3. “Evidence-Based Technical Analysis” – David Aronson

📊 One of the most dangerous books to bad traders.

Why it’s hidden:

Destroys indicator-based trading myths

Requires statistics knowledge

No excitement, no promises

Core value:

Teaches scientific testing of strategies

Introduces multiple hypothesis testing

Explains why most backtests lie

This book alone can save you years of wasted strategy-building.

4. “Statistically Sound Machine Learning for Algorithmic Trading” – Aronson & Masters

🧠 Almost unknown outside professional circles.

Why it’s hidden:

Written like a research manual

Not beginner-friendly

Zero marketing

What it teaches:

How to apply ML without overfitting

Why 95% of ML trading fails

Correct validation for financial time series

This is far more practical than flashy ML trading books.

5. “The Laws of Trading” – Agustin Lebron

🧠 Short, quiet, devastatingly insightful.

Why it’s hidden:

No code

No indicators

No strategy templates

What it gives you:

Mental models of market behavior

How edge actually emerges

Why complexity usually reduces returns

Many professional quants quietly recommend this book.

6. “A Non-Random Walk Down Wall Street” – Andrew Lo & MacKinlay

📈 Misunderstood classic.

Why it’s hidden:

Sounds like Efficient Market Hypothesis

Academic writing style

What it actually shows:

Markets are not fully random

Where predictability appears

Why predictability is unstable

This book explains why strategies decay — a crucial algo insight.

7. “The Evaluation and Optimization of Trading Strategies” – Robert Pardo

⚙️ Old, ugly, brutally practical.

Why it’s hidden:

No modern branding

No crypto, no AI hype

What it teaches:

Walk-forward testing (correctly)

Parameter stability

Why optimization ruins strategies

Many professional systems still follow Pardo’s framework.

8. “Expected Returns” – Antti Ilmanen

📊 Not an algo book — but critical.

Why it’s hidden:

Institutional-level reading

Focused on long-term returns

What you learn:

Why factors exist (momentum, carry, value)

Risk premia vs data mining

How institutions think about strategy portfolios

This book helps you design strategy portfolios, not single algos.

9. “Systematic Trading” – Robert Carver

⚖️ Quietly brilliant.

Why it’s hidden:

Not flashy

UK quant style writing

What it gives you:

Position sizing logic

Volatility targeting

Risk-first strategy construction

This is how real systematic funds operate, simplified.

10. “My Life as a Quant” – Emanuel Derman

🧠 Autobiography — but don’t underestimate it.

Why it’s hidden:

Looks like a memoir

No strategies inside

Hidden value:

How quants think

Model risk

The danger of believing your own models

This book teaches humility — essential for survival.

🧠 WHY THESE BOOKS DON’T MAKE PEOPLE “EASILY RICH”

Because they:

Require deep thinking

Don’t give plug-and-play strategies

Force you to confront uncomfortable truths

Demand statistics, patience, and skepticism

Most people prefer:

“RSI + MACD + AI = profit”

These books instead teach:

“Markets punish lazy reasoning.”

📚 HOW TO READ THESE LIKE A PRO

Don’t binge-read.
Read with intent:

Read → extract principles

Translate principle → hypothesis

Hypothesis → simple test

Test → reject most ideas

Keep only robust logic
