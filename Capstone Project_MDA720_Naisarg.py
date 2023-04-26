#!/usr/bin/env python
# coding: utf-8

# ## YouTube Video Comment Sentiment Analysis

# #### Below is the API Authentication and information from the Google Cloud 'APIs and Services' for "YouTube Data API v3" required for this project.
# #### This is an public API valid for 30 days by using personal gmail account information

# ![image.png](attachment:image.png)

# #### Installing important packages as required

# In[2]:


get_ipython().system('pip install google-auth-oauthlib #library for Google Authentication with OAuth 2.0 credentials')


# In[4]:


get_ipython().system('pip install demoji #library for removing emojis from text')


# In[6]:


get_ipython().system('pip install langdetect #language detection library that detects the language of a text')


# In[8]:


get_ipython().system('pip install mlxtend #library for machine learning algorithms and tools, including ensemble methods, clustering')
#and feature selection


# In[12]:


get_ipython().system('pip install google-api-python-client #library for accessing Google APIs using Python, including YouTube API, Google Drive API, etc')


# In[109]:


#Important libraries for text analysis and machine learning

from googleapiclient.discovery import build #for Google API such as YouTube API
from googleapiclient.discovery import build 
from google_auth_oauthlib.flow import InstalledAppFlow #used for OAuth2 authentication flow for Google APIs

import pandas as pd #Pandas library used for data manipulation

import demoji #library used for handling emojis in text data

from langdetect import detect #used for language detection in text data

import re   #library for regular expression used for text cleaning and preprocessing

from textblob import TextBlob #library used for Sentiment Analysis

from sklearn import metrics #library for machine learning algorithms like metrics

from mlxtend.plotting import plot_confusion_matrix #library used for confusion metrics

from sklearn.feature_extraction.text import CountVectorizer #library used for text feature extraction

import numpy as np #library for NumPy used for numerical calculation

import matplotlib.pyplot as plt #library used to plot the data and results

import seaborn as sns #library used for visualization

import nltk
from nltk.corpus import stopwords #library used for text data preprocessing like stopword removal and tokenization
from nltk import word_tokenize

import string #library used for the string


# In[110]:


#libraries for NLTK (National Language Tool Kit) package which includes
#stopwords, TF-id, wordcloud and tokenization

import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import TfidfVectorizer
#from wordcloud import WordCloud
nltk.download('punkt')
nltk.download('stopwords')
from wordcloud import WordCloud


# ### Web Scrapping
# #### Extracting the YouTube Comments based on the video id

# In[58]:


#Retrieving comments from a YouTube video using the Google API client for YouTube.
#The function takes a YouTube video ID and returns a list of comments for that video.
#We are creating list1 to store the comments and append in the list with each comments.

from googleapiclient.discovery import build

api_key = 'AIzaSyA8i4TnEn7Pu4OIoRC4ZCwbaPHsdWq7UrE'
list1 = []
def video_comments(Uh9643c2P6k):
	# creating youtube resource object
	youtube = build('youtube', 'v3',
					developerKey=api_key)

	# retrieve youtube video results
	video_response=youtube.commentThreads().list(
	part='snippet',
	videoId= Uh9643c2P6k
	).execute()

	# iterate video response
	while video_response:
		# from each result object
		for item in video_response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                    list1.append(comment)
			# counting number of reply of comment
                    replycount = item['snippet']['totalReplyCount']
                    print(comment, end = '\n\n')
                    print(list1)
                    
		# Again repeat
		if 'nextPageToken' in video_response:
			video_response = youtube.commentThreads().list(
					part = 'snippet',
					videoId = video_id
				).execute()
		else:
			break
        
# Enter video id
video_id = "Uh9643c2P6k"

# Call function
video_comments(video_id)


# #### Creating the csv file by importing the comments data

# In[74]:


#importing the data in the list to a csv file for further analysis

import csv

header = ['Comment']  # Change this to the desired header text

with open('comments.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(header)  # Write the header row
    for comment in list1:
        writer.writerow([comment])


# In[75]:


#Reading the csv file to make sure with the comments imported
df = pd.read_csv('comments.csv')


# In[76]:


df.head()


# In[77]:


#Required nltk libraries for the sentiment analysis steps like tokenization, stopwords, etc.

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
#from wordcloud import WordCloud
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords


# In[78]:


print(df)


# In[80]:


df.info()


# ### Sentiment Analysis
# #### Identifying the polarity of the comments using TextBlob library for the Analysis

# In[82]:


#Importing tectblob library to check polarity of the comments for the sentiment analysis
from textblob import TextBlob

# Define a function to calculate sentiment polarity for each comment
def get_comment_sentiment(comment):
    # Use TextBlob to calculate polarity of the comment
    blob = TextBlob(comment)
    sentiment_polarity = blob.sentiment.polarity
    
#Define sentiment label based on polarity score
    if sentiment_polarity > 0:
        return 'positive'
    elif sentiment_polarity < 0:
        return 'negative'
    else:
        return 'neutral'

#Applying the sentiment function to each comment in the DataFrame
df['sentiment'] = df['Comment'].apply(get_comment_sentiment)

# Print the first and last 5 rows of the DataFrame with the sentiment column
print(df.head())
print(df.tail())


# #### Clearing and Preprocessing of the data

# In[83]:


#Clearing the data to make sure no null values being present in the dataframe
df.dropna()


# ### Generating the wordcloud for the visualization

# In[84]:


get_ipython().system('pip install wordcloud #libray used to generate a wordcloud')


# #### Preprocessing with the stop words and cleaning

# In[86]:


#Generating the stop words and cleaning the data for the further steps

stp_words=stopwords.words('english')
def clean_review(review):
  cleanreview=" ".join(word for word in review.
                       split() if word not in stp_words)
  return cleanreview
 
df['Comment']=df['Comment'].apply(clean_review)


# ## General wordcloud inclusive of all three polarities

# In[89]:


# Generating a wordcloud based on all comments in the DataFrame
consolidated = ' '.join(word for word in df['Comment'].astype(str))

# Generating a wordcloud based on comments with sentiment = positive, negative and neutral

wordCloud = WordCloud(width=1600, height=800, random_state=21, max_font_size=110)
plt.figure(figsize=(15,10))
plt.imshow(wordCloud.generate(consolidated), interpolation='bilinear')
plt.axis('off')
plt.show()


# In[100]:


print(consolidated) #Checking the consolidate


# #### Listing and checking the polarities with the consolidate

# In[94]:


print(df['sentiment'].unique()) #Making sure for the polarities of the sentiment to use in further steps


# ## Wordcloud for the positive sentiments in the comments

# In[97]:


#Wordcloud for the comments with the positive statements

consolidated=' '.join(word for word in df['Comment'][df['sentiment']=="positive"].astype(str))
wordCloud=WordCloud(width=1600,height=800,random_state=21,max_font_size=110)
plt.figure(figsize=(15,10))
plt.imshow(wordCloud.generate(consolidated),interpolation='bilinear')
plt.axis('off')
plt.show()


# ## Wordcloud for the negative polarity sentiments

# In[99]:


#Wordcloud for the comments with the negative statements

consolidated=' '.join(word for word in df['Comment'][df['sentiment']=="negative"].astype(str))
wordCloud=WordCloud(width=1600,height=800,random_state=21,max_font_size=110)
plt.figure(figsize=(15,10))
plt.imshow(wordCloud.generate(consolidated),interpolation='bilinear')
plt.axis('off')
plt.show()


# ## Wordcloud for the neautral polarity sentiments

# In[111]:


#Wordcloud for the comments with the neutral statements

consolidated=' '.join(word for word in df['Comment'][df['sentiment']=="neutral"].astype(str))
wordCloud=WordCloud(width=1600,height=800,random_state=21,max_font_size=110)
plt.figure(figsize=(15,10))
plt.imshow(wordCloud.generate(consolidated),interpolation='bilinear')
plt.axis('off')
plt.show()


# ### Generating the metrics using logistic regression and TF-ID

# In[101]:


#Transferring the text data to a numerical for the machine learning algorith 
#of generating metrics using TFIDVectorizer

cv = TfidfVectorizer(max_features=2500)
X = cv.fit_transform(df['Comment'] ).toarray()


# #### Allocating the data to train and test models for the confusion metrics

# In[103]:


#Allocating the 25% of the data to test and rest for the train

from sklearn.model_selection import train_test_split
x_train ,x_test,y_train,y_test=train_test_split(X,df['sentiment'],
                                                test_size=0.25 ,
                                                random_state=42)


# In[104]:


#Prediction on the sentiment based on the train and test data using logistic regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model=LogisticRegression()
 
#Model fitting
model.fit(x_train,y_train)
 
#testing the model
pred=model.predict(x_test)
 
#model accuracy
print(accuracy_score(y_test,pred))


# ## Confusion metrics for the visualization

# In[106]:


#Generating a confusion metrics for the visualization of the predicted results

unique_labels = np.unique(np.concatenate((y_test, pred)))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
cm_display.plot()
plt.show()

