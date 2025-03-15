# Financial News Analysis and Ranking System

## Overview
This project is an end-to-end MLOps pipeline for analyzing financial news, ranking articles based on their investment impact, and deploying the application using AWS. The system integrates multiple components, including data preprocessing, sentiment analysis, ranking algorithms, and a Streamlit-based dashboard for visualization.

## Features
- **News Aggregation**: Fetches financial news articles from NewsAPI.
- **Sentiment Analysis**: Predicts sentiment using a fine-tuned model.
- **Ranking System**: Ranks articles based on Reddit engagement and source credibility.
- **Data Preprocessing**: Tokenization, stopword removal, and lemmatization.
- **Deployment**: Hosted on AWS with a live demo at [Financial News Dashboard](http://3.84.211.98:8501/).

## Technologies Used
- **Python**
- **PRAW (Python Reddit API Wrapper)**
- **NLTK for text preprocessing**
- **Streamlit for dashboard**
- **AWS (S3, EC2, Lambda) for MLOps**
- **NewsAPI for news fetching**

## Project Structure
```
.
├── Model
│   ├── model.py                # Reddit-based ranking model
│   ├── fine_tuning_notebook    # Notebook for fine-tuning last layers
├── ranking.py                  # Ranking algorithm
├── dataprocess.py               # Preprocessing text
├── main.py                      # Streamlit dashboard
├── requirements.txt             # Dependencies
├── .env                         # API keys & credentials (ignored in GitHub)
└── README.md                    # This file
```

## How It Works
1. **Fetch News**: The system retrieves the latest financial news articles.
2. **Preprocess Text**: Tokenization, stopword removal, and lemmatization.
3. **Predict Sentiment**: Classifies sentiment using a fine-tuned model.
4. **Calculate Ranking**:
   - Engagement from Reddit (posts & comments)
   - Source credibility based on a predefined list
5. **Display Results**: Interactive visualization on Streamlit.

## Fine-Tuning
- The model was fine-tuned on a subset of financial news data, updating less than 2% of the last layers for better domain adaptation.
- The fine-tuning notebook is available in the `notebook` folder under `mobile_notebook`.

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/financial-news-analysis.git
   cd financial-news-analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Create a `.env` file and add API keys for `NEWSAPI_KEY`, `CLIENT_ID`, `CLIENT_SECRET`, and `Redirect_uri`.
4. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```

## Deployment
- The system is deployed on AWS using EC2, S3, and Lambda.
- The model and fine-tuning components are stored in S3 and loaded dynamically.
- The Streamlit app is hosted on EC2 with automatic updates from the repository.

## Live Demo
Check out the live demo: [Financial News Dashboard](http://3.84.211.98:8501/).

## Future Improvements
- Extend the model to support multiple languages.
- Improve ranking by integrating real-time financial data.
- Optimize performance for large-scale deployment.
- Implement real-time updates for financial news analysis.

## Contributors
- **Your Name** - [GitHub](https://github.com/yourusername)

## License
This project is licensed under the MIT License.

