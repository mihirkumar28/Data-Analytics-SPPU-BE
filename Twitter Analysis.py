#!/usr/bin/env python
# coding: utf-8

# # Import all libraries and packages

# In[30]:


import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
import seaborn as sns
import matplotlib.pyplot as plt


# # Load dataset

# In[17]:


data=pd.read_csv('twitter.csv',usecols=['label','tweet'])[['tweet','label']]


# In[18]:


data.head()


# In[19]:


data['label'].hist()


# In[20]:


from nltk import word_tokenize
#for i in range(0,):
tweet_tokens=word_tokenize(data['tweet'][0])
print(tweet_tokens)
    


# In[21]:


def preprocessing(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#','', tweet)
    tweet = re.sub('[^a-z]', ' ',tweet)
    tweet = tweet.split()
    ps = PorterStemmer()
    stop_words = stopwords.words('english')
    stop_words.remove('not')
    tweet = [ps.stem(word) for word in tweet if not word in set(stop_words)]
    tweet = ' '.join(tweet)
    return tweet


# In[22]:


print(preprocessing(data['tweet'][0]))


# In[23]:


length = len(data['tweet'])
vocabulary = ''
negative_vocabulary = ''
positive_vocabulary = ''
corpus = []
for i in range(0,5000):
    tweet = preprocessing(data['tweet'][i])
    corpus.append(tweet)
    vocabulary = vocabulary+ ' ' + tweet
    if data['label'][i] == 1:
        negative_vocabulary = negative_vocabulary+ ' ' + tweet
    else:
        positive_vocabulary = positive_vocabulary+ ' ' + tweet


# In[44]:


cloud=WordCloud(background_color="white")
cloud.generate(positive_vocabulary)
cloud.to_image()


# In[25]:


from sklearn.feature_extraction.text import CountVectorizer 
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
y = data.iloc[:5000, 1].values


# In[27]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state = 0)


# In[46]:


def classifier(clf,classifier_type):
    clf.fit(X_train,y_train)
    predictions=clf.predict(X_test)
    print("Classifier: ",classifier_type)
    print("\n",classification_report(predictions,y_test))
    
    cm=confusion_matrix(y_test,predictions)
    fig=sns.heatmap(cm,annot=True,cmap='Blues')
    plt.show()
    accuracy[classifier_type]=accuracy_score(y_test,predictions)
    
accuracy={}
classifier(LogisticRegression(),"Logistic Regression")
classifier(MultinomialNB(),"Multinomial Naive Bayes")
classifier(DecisionTreeClassifier(),"Decision Tree Classifier")
classifier(SVC(kernel="linear"),"Support Vector Classifier")


# In[42]:


print(accuracy.items())
plt.figure(figsize=(12,7))
plt.bar(range(len(accuracy)), list(accuracy.values()), align='center')
plt.xticks(range(len(accuracy)), list(accuracy.keys()))


# In[ ]:




