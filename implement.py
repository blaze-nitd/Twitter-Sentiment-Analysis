
# coding: utf-8

# In[1]:


import twitter
api=twitter.Api(consumer_key='qzp9N9pS4jvMWABSYm6noYT36',consumer_secret='2T1Qt4J8gAICp0lAwpHv9rvFwuvXbtuulcKjuSxhFW39Qi3WWg',access_token_key='4311969374-qhIRb4PlYeOZvuHCGfgVhiPRtp6pGZhn2eqRBS2',access_token_secret='OQCCxP0W3a9Zc169uCUQcLnuFHNtZ2o2WD6RHOjslXmq3')


def createTestData(search_string):
    try:
        tweets_fetched=api.GetSearch(search_string, count=100)
        return [{"text":status.text,"label":None} for status in tweets_fetched]
#createTestData returns a list of dictionaries where the key value of the dictionary is the text of the tweets and value id none
    except:
        print("Sorry, there was an error")
#print(api.VerifyCredentials())

search_string=input("Hi there! What are you searching for today?")
test_data=createTestData(search_string)


# In[2]:


def createTrainingDataCorpus(corpusFile):
    import csv
    corpus=[]
    i=1
    with open(corpusFile,'r') as csvFile:
        lineReader=csv.reader(csvFile,delimiter=',',quotechar="\"")
        for row in lineReader:
            if(i<=5000):
                corpus.append({"text":row[3],"label":'positive' if row[1] is '1' else 'negative'})
            i+=1
    return corpus


corpusFile="sental.csv"
trainingData=createTrainingDataCorpus(corpusFile)


# In[3]:


(trainingData[:10])


# In[4]:


import re
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords

class PreProcessTweets:
    def __init__(self):
        self._stopwords=set(stopwords.words('english')+list(punctuation)+['AT_USER','URL'])
        
    def processTweets(self,list_of_tweets):
        processedTweets=[]
        for tweet in list_of_tweets:
            processedTweets.append((self._processTweet(tweet["text"]),tweet["label"]))
        return processedTweets
    
    def _processTweet(self,tweet):
        tweet=tweet.lower()
        tweet=re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)     
        tweet=re.sub('@[^\s]+','AT_USER',tweet)
        tweet=re.sub(r'#([^\s]+)',r'\1',tweet)
        tweet=word_tokenize(tweet)
        return [word for word in tweet if word not in self._stopwords]
    
tweetProcessor=PreProcessTweets()
ppTrainingData=tweetProcessor.processTweets(trainingData)
ppTestData=tweetProcessor.processTweets(test_data)


# In[7]:


import nltk 
def buildVocabulary(ppTrainingData):
    all_words=[]
    for (words,sentiment) in ppTrainingData:
        all_words.extend(words)
    wordlist=nltk.FreqDist(all_words)
    word_features=wordlist.keys()
    return word_features

def extract_features(tweet):
    tweet_words=set(tweet)
    features={}
    for word in word_features:
        features['contains(%s)' % word]=(word in tweet_words)
    return features 

word_features = buildVocabulary(ppTrainingData)
trainingFeatures=nltk.classify.apply_features(extract_features,ppTrainingData)

NBayesClassifier=nltk.NaiveBayesClassifier.train(trainingFeatures)

# Support Vector Machines 
from nltk.corpus import sentiwordnet as swn
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer 

svmTrainingData=[' '.join(tweet[0]) for tweet in ppTrainingData]

vectorizer=CountVectorizer(min_df=1)
X=vectorizer.fit_transform(svmTrainingData).toarray()
vocabulary=vectorizer.get_feature_names()

swn_weights=[]

for word in vocabulary:
    try:
        synset=list(swn.senti_synsets(word))
        common_meaning =synset[0]
        if common_meaning.pos_score()>common_meaning.neg_score():
            weight=common_meaning.pos_score()
        elif common_meaning.pos_score()<common_meaning.neg_score():
            weight=-common_meaning.neg_score()
        else: 
            weight=0
    except: 
        weight=0
    swn_weights.append(weight)


swn_X=[]
for row in X: 
    swn_X.append(np.multiply(row,np.array(swn_weights)))
swn_X=np.vstack(swn_X)


# We have our documents ready. Let's get the labels ready too. 
# Lets map positive to 1 and negative to 2 so that everything is nicely represented as arrays 
labels_to_array={"positive":1,"negative":2}
labels=[labels_to_array[tweet[1]] for tweet in ppTrainingData]
y=np.array(labels)

# Let's now build our SVM classifier 
from sklearn.svm import SVC 
SVMClassifier=SVC()
SVMClassifier.fit(swn_X,y)




# First Naive Bayes 
NBResultLabels=[NBayesClassifier.classify(extract_features(tweet[0])) for tweet in ppTestData]

# Now SVM 
SVMResultLabels=[]
for tweet in ppTestData:
    tweet_sentence=' '.join(tweet[0])
    svmFeatures=np.multiply(vectorizer.transform([tweet_sentence]).toarray(),np.array(swn_weights))
    SVMResultLabels.append(SVMClassifier.predict(svmFeatures)[0])
    # predict() returns  a list of numpy arrays, get the first element of the first array 
    # there is only 1 element and array








# Step  : GEt the majority vote and print the sentiment 

if NBResultLabels.count('positive')>NBResultLabels.count('negative'):
    print(" Positive Sentiment. Positivity percentage: "+str(100*NBResultLabels.count('positive')/len(NBResultLabels))+"%")
else: 
    print("Negative Sentiment. Negativity percentage: "+str(100*NBResultLabels.count('negative')/len(NBResultLabels))+"%")
    
    
    
    
  

