# Twitter-Sentiment-Analysis

Want to know what is the prevailing sentiment of a related topic of your choice on twitter?  
This project has everything which will make your desire come true.  

The project is implemented in Python3.  
Uses twitter-python module to fetch data from twitter API.  
Uses Naive Bayes Classification and Support Vector Machines to train and classify the output.  
sentiwordnet is used to give magnitudes to each of the feature-vectors in svm training  

The file sental.csv has the training data within itself. Since it's size is toon large the code takes a lot of time to run. So, have some patience. I am working on it to improve its runtime.  

**Usage**  
1> Install python-twitter module. Run the command:  
`$ pip install python-twitter`  

2>Install nltk and numpy. Run the command:  
`$sudo pip install -U nltk`  
`$sudo pip install -U numpy` 

3>Install nltk data. Run the command in the python interpreter:  
`import nltk`  
`nltk.download()`  
A GUI will appear which will show the status of download. It will take some time. So have patience  

4>Go to the repository where all these files are saved with the help of : `cd <path>`.<path> is the required path according to your system which you have to see.  
  Then run the command: `python implement.py`  
You will get output after a certain span of time. Have patience!!
