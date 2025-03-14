import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stopWords = set(stopwords.words("english"))

def cleanText(d):
    try:
        tokens = d.lower().split()  # This will split the text by whitespace

        # Remove puncs and numbers
        punc = [w for w in tokens if w.isalpha()]

        # Remove bad characters (URLs, Tags, Mentions)
        d = re.sub('http\S*', '', d).strip()
        d = re.sub('www\S*', '', d).strip()
        d = re.sub('#\S*', '', d).strip()
        d = re.sub('@\S*', '', d).strip()

        # Remove upper brackets to keep negative auxiliary verbs in text
        d = d.replace("'", "")

        # Remove Stopwords
        sw = [t for t in punc if t not in stopWords]

        # Lemmatization
        cleaned = [WordNetLemmatizer().lemmatize(t) for t in sw]

        # Joining the text with each other
        return " ".join(cleaned)
    except Exception as e:
        print(f"Error in cleanText: {e}")
        return None
