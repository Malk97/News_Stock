import praw
import os
from dotenv import load_dotenv
import os

load_dotenv() 

def get_reddit_posts(query):
    
    reddit = praw.Reddit(client_id=os.getenv("CLIENT_ID"),
                         client_secret=os.getenv('CLIENT_SECRET'),
                         user_agent=os.getenv('CLIENT_SECRET'), 
                         redirect_uri=os.getenv('Redirect_uri'))

    subreddit = reddit.subreddit('stocks')  
    posts = subreddit.search(query, limit=50, time_filter='week')
    
    post_data = []
    number_of_post = 0
    total = 0

    for submission in posts:
        number_of_post += 1
        score_plus_comments = submission.score + submission.num_comments
        post_data.append(score_plus_comments)
        total += score_plus_comments

    ratio = number_of_post / total if total != 0 else 0  
    
    normalized_values = []
    if post_data:  
        min_value = min(post_data)
        max_value = max(post_data)

        for value in post_data:
            if max_value != min_value:
                normalized_value = (value - min_value) / (max_value - min_value)
            else:
                normalized_value = 0  
            normalized_values.append(normalized_value)
        
    return sum(normalized_values) / number_of_post if number_of_post > 0 else 0

def ranking(text, sentiment_score, source_credibility):
    alpha, beta, gamma = 0.5, 0.3, 0.2
    rank_score = beta * get_reddit_posts(text) + gamma * source_credibility
    return rank_score
