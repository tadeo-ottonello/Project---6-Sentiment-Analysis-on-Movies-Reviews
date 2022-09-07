import re
import unidecode
import spacy
import nltk

from tqdm import tqdm
from contractions import CONTRACTION_MAP
from nltk.tokenize.toktok import ToktokTokenizer

tokenizer = ToktokTokenizer()
nlp = spacy.load('en_core_web_sm')
nltk.download('stopwords')
stopword_list = set(nltk.corpus.stopwords.words('english'))


def remove_html_tags(text):
    text = re.sub(re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});'), '', text)
    return text


def stem_text(text):
    porter = nltk.stem.PorterStemmer()
    token_words = tokenizer.tokenize(text)
    text = " ".join([porter.stem(word) for word in token_words])
    return text


def lemmatize_text(text):
    doc = nlp(text)
    text = " ".join([token.lemma_ for token in doc])
    return text


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    for contraction, word in contraction_mapping.items():
        text = text.replace(contraction, word)
    return text


def remove_accented_chars(text):
    text = unidecode.unidecode(text)
    return text


def remove_special_chars(text, remove_digits=False):
    text = re.sub("[^a-zA-Z0-9 ]", "", text)
    if remove_digits == True:
            text = re.sub("[0-9]", "", text)
    return text


def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
    if not is_lower_case:
        text = text.lower()
    words = tokenizer.tokenize(text)
    text = " ".join([word for word in words if not word in stopword_list])
    return text


def remove_extra_new_lines(text):
    text = text.replace("\n", " ")
    return text


def remove_extra_whitespace(text):
    text = text.split()
    text = " ".join(text)
    return text
    

def normalize_corpus(
    corpus,
    html_stripping=True,
    contraction_expansion=True,
    accented_char_removal=True,
    text_lower_case=True,
    text_stemming=False,
    text_lemmatization=False,
    special_char_removal=True,
    remove_digits=True,
    stopword_removal=True,
    stopwords=stopword_list
):
    
    normalized_corpus = []

    # Normalize each doc in the corpus
    for doc in tqdm(corpus):
        # Remove HTML
        if html_stripping:
            doc = remove_html_tags(doc)
            
        # Remove extra newlines
        doc = remove_extra_new_lines(doc)
        
        # Remove accented chars
        if accented_char_removal:
            doc = remove_accented_chars(doc)
            
        # Expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc)
            
        # Lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
            
        # Stemming text
        if text_stemming and not text_lemmatization:
            doc = stem_text(doc)
            
        # Remove special chars and\or digits    
        if special_char_removal:
            doc = remove_special_chars(
                doc,
                remove_digits=remove_digits
            )  

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)

         # Lowercase the text    
        if text_lower_case:
            doc = doc.lower()

        # Remove stopwords
        if stopword_removal:
            doc = remove_stopwords(
                doc,
                is_lower_case=text_lower_case,
                stopwords=stopwords
            )

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)
        doc = doc.strip()
            
        normalized_corpus.append(doc)
        
    return normalized_corpus
