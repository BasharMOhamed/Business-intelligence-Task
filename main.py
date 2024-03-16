import pandas as pd
import nltk
import string
# nltk.download("all")
data = pd.read_csv("Restaurant_Reviews.csv")

# Data Preproceessing
print(data.isna().sum())
print(data.duplicated().sum())
data = data.drop_duplicates()
print("After: ", data.duplicated().sum())

filtered_data = data[~data['Liked'].isin(['Yes', 'No'])]
print(filtered_data)

# TEXT PROCESSING

# 1- PUNCTUATION REMOVAL
data["Review"] = data["Review"].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
print(data["Review"])

# 2- CASE FOLDING
data["Review"] = data["Review"].apply(lambda x: x.lower())
print(data["Review"])

# 3- Tokenization
from nltk.tokenize import word_tokenize
data["Review"] = data["Review"].apply(lambda x: word_tokenize(x))
print(data["Review"])

# 4- STOP WORDS REMOVAL
from nltk.corpus import stopwords
stopwords = set(stopwords.words("english"))
print(stopwords)
stopwords.discard('not')
data["Review"] = data["Review"].apply(lambda x: [word for word in x if not word in stopwords])
print(data["Review"])



# 5- LEMMATIZATION WITH POS

from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
data["Review"] = data["Review"].apply(lambda x: nltk.pos_tag(x))
print(data["Review"])

tag_dict = {"J": wn.ADJ,
            "N": wn.NOUN,
            "V": wn.VERB,
            "R": wn.ADV}
def extract_wnpostag_from_postag(tag):
    return tag_dict.get(tag[0].upper(), None)

def lemmatize_tupla_word_postag(tupla):
    tag = extract_wnpostag_from_postag(tupla[1])
    return lemmatizer.lemmatize(tupla[0], tag) if tag is not None else tupla[0]



lemmatizer = WordNetLemmatizer()

def BagOfWords(arr):
    # ModifiedList = []
    # for tuple in arr:
    #    tuple = lemmatize_tupla_word_postag(tuple)
    #    ModifiedList.append(tuple)
    # return ModifiedList
    return ' '.join([lemmatize_tupla_word_postag(tuple) for tuple in arr])


data["Review"] = data["Review"].apply(lambda x: BagOfWords(x))
print(data["Review"])


# SENTIMENT ANALYSIS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):

    scores = analyzer.polarity_scores(text)

    sentiment = "Yes" if scores['pos'] > scores['neg'] or scores['pos'] > scores['neu'] else "No"

    return sentiment

# apply get_sentiment function
# data['scores']= data['Review'].apply(lambda x: [analyzer.polarity_scores(word) for word in x])

data['sentiment'] = data['Review'].apply(get_sentiment)

print(data)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(data['Liked'], data['sentiment']))

from sklearn.metrics import classification_report

print(classification_report(data['Liked'], data['sentiment']))