
# TweetAnalyzer: Unveiling Sentiments in Tweets



## Contents

* ***3-FoldCrossVerificationResults:*** This folder contains 3 files that contain the result of the 3 cross validation splits that were used while fine-tuning the sentiment analysis model.
* ***ModelStatistics.txt:*** This text file contains statistical information about the 3 splits and the average values.
* ***PeasAI-TweetAnalyzer-ProjectReport.pdf:*** It is the project report file. It contains the link of the Kaggle Notebook that was used for fine-tuning along with some other useful information.
* ***app.py:*** The main code. The file serves as the main codebase for the sentiment analysis application. It functions as the entry point for the application, orchestrating the processing of input data (tweets) and providing the output of classified sentiments.
* ***finalTestSetResults.csv:*** This contains the results for the final test set used while fine-tuning.

## Track and Contributors ##

### Track
MACHINE LEARNING

### Contributors
Dhruv RJ (IMT2023032)

Pranav Sandeep (IMT2023058)

Venkat Ramireddy (IMT2023102)
## Problem Statement
In today's digital age, airline companies are inundated with an abundance of feedback and reviews disseminated through tweets, offering invaluable insights into customer experiences and sentiments. Recognizing the significance of this feedback for enhancing operational efficiency and customer satisfaction, our project endeavors to develop a robust solution. Our objective is to implement a sophisticated machine learning model capable of accurately classifying tweets pertaining to airline experiences. This model not only discerns sentiment but also categorizes tweets based on specific themes or topics, thereby providing comprehensive insights. By leveraging advanced natural language processing techniques, our solution aims to empower airline companies to proactively address issues, refine services, and cultivate positive customer relationships
## Goal
To develop a system to categorise tweets mentioning a specific business into three sentiment categories: negative (bad review), neutral (neutral review), and positive (positive review), while also identifying and labelling the specific aspects of the product or service being praised or criticised within each tweet.

We aim to develop a user-friendly graphical user interface (GUI) application using Python, facilitating efficient visualization and presentation of filtered tweets in a structured manner. This application will enhance accessibility and usability, providing a seamless experience for users to interact with and interpret sentiment analysis results derived from the specialized model tailored for analyzing tweets related to airline companies
## Features



***User-friendly Dashboard:***  We have designed a visually appealing and intuitive dashboard where users can view summarized sentiment analysis results at a glance.

***State-of-the-Art Sentiment Analysis Model Integration:***  Our application integrates the powerful Twitter-RoBERTa-Base-Sentiment-Latest model, which has been trained on an extensive dataset of approximately 124 million tweets spanning from January 2018 to December 2021.
This RoBERTa-based model is fine-tuned for sentiment analysis using the Tweet Eval benchmark, ensuring high accuracy. Leveraging this state-of-the-art model, we deliver precise sentiment analysis results to users.

***Tweet Details:*** Allow users to view detailed information about specific tweets, such as the full text of the tweet, associated hashtags, mentions, and retweet count.


## Tech Stack

- **Programming Language**: Python
- **Machine Learning Frameworks**: PyTorch, 
- **NLP Libraries**: Hugging Face Transformers
- **GUI Framework**: Custom Tkinter
- **Version Control**: Git (GitHub)
- **Development Environment**: Visual Studio Code, Kaggle Notebooks
- **Dependencies (Python Libraries)**:  
    *  pandas
    *  numpy
    *  torch 
    *  transformers
    * matplotlib
    * customtkinter
    * scipy
    * datasets
    * evaluate
## How to Run
* Install all the python libraries shown below-
    *  pandas
    *  numpy
    *  torch 
    *  transformers
    * matplotlib
    * customtkinter
    * scipy
* Now download PeasAI-TweetAnalyzer-main.zip from github and extract all the files.
* Run app.py using terminal.
## Applications of our Idea
- ***Brand Reputation:*** The model can help airline companies monitor their brand reputation in real-time by tracking sentiment trends across social media. 

- ***Customer Feedback Analysis:*** Airlines can use the sentiment analysis model to systematically analyze customer feedback from social media platforms like Twitter.

- ***Personalized Customer Engagement:*** By analyzing sentiment and categorizing tweets based on topics or themes, airlines can personalize their responses to customer inquiries or feedback on social media.

- ***Marketing Insights:*** Analyzing sentiment around marketing campaigns and promotions helps airlines gauge customer reactions and refine their marketing strategies for maximum impact.
## Further Improvements

* ***Integrating Real-Time Engagement and Predictive Analytics:*** We can leverage the capability of the Twitter API to not only classify tweets in real-time, but also predict potential virality or influence. This predictive ability empowers airlines to take immediate and targeted actions, such as responding to influential tweets promptly or amplifying content likely to go viral. 

* ***Streamlined Data Management:*** We can generate a structured data frame for each received tweet, to ensure clarity and ease of management. We can also export it as a CSV to facilitate seamless data storage and retrieval for future reference.

* ***Expanded Scope and Versatility:***  By extending training beyond airline-related tweets to encompass tweets from various sectors, our model becomes adaptable for use by any company seeking sentiment analysis insights. This expansion enables a wider range of businesses to leverage the benefits of our model.

* ***Enhanced Code Structure and GUI Refinement:*** The well-organised, object-oriented code facilitates future development of new features. The GUI can be refined for a more user-friendly experience.

## Demo video


Link- https://youtu.be/DCslvVYv0lc