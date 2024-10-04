# Analysis of Social Response after a Natural Calamity using Text Mining and Sentiment Analysis

## Project Overview

This project focuses on analyzing public reactions following a natural calamity through text mining and sentiment analysis. The study examines YouTube comments and news articles about the Turkey-Syria earthquake of February 2023. The objective is to extract and interpret sentiments, emotions, and patterns in public responses during disaster events to aid disaster management and decision-making processes.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Methodology](#methodology)
  - [Sentiment Analysis](#sentiment-analysis)
  - [K-Means Clustering](#k-means-clustering)
  - [Association Rule Mining](#association-rule-mining)
  - [Emotion Detection](#emotion-detection)
- [Results](#results)
- [Conclusion](#conclusion)
- [Limitations](#limitations)
- [Future Scope](#future-scope)
- [References](#references)

## Data Collection

The dataset consists of:
- **YouTube Comments**: Extracted using the YouTube Data API (v3). A total of 400 comments related to the earthquake were gathered.
- **News Articles**: Around 200 news articles were collected using the Guardian News API and World News API.

## Data Preprocessing

Text preprocessing was performed using the Natural Language Toolkit (NLTK) to:
- Remove stop words, special characters, and whitespace.
- Clean and prepare the text for further analysis by eliminating non-informative data.

Feature extraction was done using **TF-IDF** (Term Frequency-Inverse Document Frequency) to convert the text data into numerical form suitable for machine learning algorithms.

## Methodology

### Sentiment Analysis

Sentiment analysis was performed using the **TextBlob** library, categorizing the text into:
- Positive
- Negative
- Neutral

### K-Means Clustering

Unsupervised learning was employed using the **K-Means algorithm** to group similar text data. The optimal number of clusters was determined using the **Elbow method**. Dimensionality reduction was performed using **Principal Component Analysis (PCA)**.

### Association Rule Mining

**Association Rule Mining (ARM)** was used to find patterns in the text. The **Apriori algorithm** was applied with a minimum support level of 7% to detect frequent itemsets.

### Emotion Detection

Emotion detection was carried out using the **NRCLex** library, which identified five key emotions:
- Fear
- Anger
- Sadness
- Disgust
- Joy

## Results

The project revealed the following key insights:
- Sentiment distribution in news articles and YouTube comments, where most articles were neutral, and YouTube comments displayed a higher percentage of emotional content.
- Clustering results showed distinct groups of content, providing a deeper understanding of public reactions.
- Strong associations between specific words in news articles, such as "people" and "Turkey," were identified.

## Conclusion

The project successfully demonstrated how sentiment analysis, clustering, association rule mining, and emotion detection techniques can be used to extract valuable insights from unstructured textual data during disasters. These findings can assist disaster management teams in better understanding public reactions and needs during emergency situations.

## Limitations

- The dataset was limited to English-language text, and non-English comments written in English alphabets were not processed effectively.
- A larger dataset could provide more robust results, but data collection was restricted due to API limitations.

## Future Scope

- Expand language support for non-English text.
- Explore multimodal sentiment analysis by integrating images, videos, and audio.
- Implement transformer-based architectures (e.g., BERT) for more accurate sentiment detection.

## References

- [TextBlob Sentiment Analysis](https://textblob.readthedocs.io/en/dev/) - Official documentation for the TextBlob library used for sentiment analysis.
- [NRCLex Emotion Detection](https://pypi.org/project/NRCLex/) - Python package for emotion detection based on NRC's affect lexicon.
- [K-Means Clustering in Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) - Scikit-learn documentation for K-Means clustering.
- [Apriori Algorithm in MLxtend](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/) - MLxtend library documentation for the Apriori algorithm used in association rule mining.
- [Sentiment Analysis Using Python](https://www.analyticsvidhya.com/blog/2022/07/sentiment-analysis-using-python/) - Tutorial on sentiment analysis using Python, useful for understanding TextBlob and similar libraries.
- [YouTube Data API](https://developers.google.com/youtube/v3) - YouTube Data API documentation for extracting comments and other information.
- [Natural Language Toolkit (NLTK)](https://www.nltk.org/) - Official website for the NLTK library, which was used for text preprocessing in the project.
- [Principal Component Analysis (PCA) in Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) - Scikit-learn documentation for PCA, used for dimensionality reduction.
- [MLxtend Library](http://rasbt.github.io/mlxtend/) - Official website for the MLxtend library, offering various machine learning extensions.
