import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


# to preprocess the data
def preProcessData(text):
    # to lower the text
    text = text.lower()
    # to remove mentions
    text = re.sub(r'@[a-z0-9]+', '', text)
    # to remove hyperlinks i.e, https://, http://...
    text = re.sub(r'https?://\S+', '', text)
    # to remove punctuations
    text = re.sub(r'[^a-z\s]', '', text)
    # to remove stopwords and stemming the words
    text = text.split()
    wNL = WordNetLemmatizer()
    ps = PorterStemmer()
    text = [wNL.lemmatize(word) for word in text if word not in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text
