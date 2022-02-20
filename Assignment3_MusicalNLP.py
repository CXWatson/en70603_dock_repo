'''
EN705603 Creating AI Enabled Systems
Charles Xia
Python Script for printing out summary information
'''
import pandas as pd
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

musical_instruments_review_df = pd.read_csv("Musical_instruments_reviews.csv")
summary_reviews = musical_instruments_review_df["summary"]
work_tokenize_list = [nltk.word_tokenize(word) for word in summary_reviews]
porter_stem_list = [PorterStemmer().stem(word) for word in summary_reviews]
lemmatize_list = [WordNetLemmatizer().lemmatize(word,'v') for word in summary_reviews]
result = zip(work_tokenize_list,porter_stem_list,lemmatize_list)

for word in result:
    print(f"Tokenize: {word[0]}, Porter Stem: {word[1]}, Lemmatize: {word[2]}")
