# Завантажимо всі необхідні бібліотеки
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
import spacy
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import string
import itertools
from collections import Counter

import pandas as pd
import numpy as np
import spacy  # додано для роботи у пайтон локально

import nltk
# nltk.download('stopwords')  увімкнути якщо не встановлено
# перевіряє локальну папку, не завантажує нічого з інтернету
stopwords.words("english")

tqdm.pandas()


df = pd.read_csv('./Reviews.csv', index_col='Id')

df_head = df.head(5)
print(f"Thi is dataframe head with index_col='Id: \n ' {df_head}")

df_info = df.info()
print(f"Thi is dataframe info with index_col='Id: \n ' {df_info}")

df_loc = df.loc[df['Score'] != 3]
print(f"Thi is dataframe loc with index_col='Id: \n ' {df_loc}")

df_shape = df_loc.shape
print(f"Thi is dataframe_loc shape with index_col='Id: \n ' {df_shape}")


df['sentiment'] = [1 if score in [4, 5] else 0 for score in df['Score']]
print(f"Thi is dataframe sentiment ' {df['sentiment'].head(10)}")


print("\nВидалення дублікатів\n")

df_sum_duplicates = df.duplicated().sum()
print(f"Thi is dataframe duplicated sum 🤓 \n' {df_sum_duplicates}")

print("\nВидалимо записи-дублікати\n")

df_duplicates_remove = df.drop_duplicates().reset_index(drop=True)
print(f"Remove duplicates: \n {df_duplicates_remove}")


print("\nПошукаємо ідентичні відгуки про різні версії одного продукту.\n")

df_groupby_reviews = df.groupby(['UserId', 'Time', 'Text']).count(
).sort_values('ProductId', ascending=False).head(10)
print(f"Group by reviws: \n {df_groupby_reviews}")

print("\nПеревіримо на прикладі дублювання відгуків\n")

search_text = "I have two cats, one 6 and one 2 years old. Both are indoor cats in excellent health. I saw the negative review and talked to my vet about it. I've also asked a number of veterinary professionals what to feed my cats and they all answer the same thing: Science Diet. Sure, you'll see stories of how one person's cat had issues, but even if that's 100% true, it's 1 case out of millions. Science and fact aren't based on someone's experience.<br /><br />So my point is, I love my cats and I'm very concerned about their health. I trust people who actually have medical degrees and experience with a wide range of animals. My only caution is do not fall for some hype or scare tactic that recommends some unproven or untested food or some fad diet for your pet. Don't listen to me, don't listen to the negative review. ASK YOUR VET what they recommend, and follow their instructions. My guess is you'll end up buying the Science Diet anyhow."
duplicates_example = df.loc[
    (df['UserId'] == 'A36JDIN9RAAIEC') &
    (df['Time'] == 1292976000) &
    (df['Text'] == search_text)
]
print(f"Example of duplicates: \n {duplicates_example}")


print("\nВидалимо однакові відгуки\n")

df_remove_double_review = df.drop_duplicates(subset={"UserId", "Time", "Text"})

print(f"Print double reviews 🌑: \n{df_remove_double_review.shape}")

print(f"\nНормалізація тексту (text normalization)\n")

contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are"
}


print(f"\nЗавантажимо перелік стоп-слів із бібліотеки nltk.\n")

stop_words = set(stopwords.words('english')).union(
    {'also', 'would', 'much', 'many'})


print(f"\nВидалимо слова-заперечення зі списку стоп-слів.\n")

negations = {
    'aren',
    "aren't",
    'couldn',
    "couldn't",
    'didn',
    "didn't",
    'doesn',
    "doesn't",
    'don',
    "don't",
    'hadn',
    "hadn't",
    'hasn',
    "hasn't",
    'haven',
    "haven't",
    'isn',
    "isn't",
    'mightn',
    "mightn't",
    'mustn',
    "mustn't",
    'needn',
    "needn't",
    'no',
    'nor',
    'not',
    'shan',
    "shan't",
    'shouldn',
    "shouldn't",
    'wasn',
    "wasn't",
    'weren',
    "weren't",
    'won',
    "won't",
    'wouldn',
    "wouldn't"
}

stop_words = stop_words.difference(negations)

print(f"\nСтемінг - зазвичай є швидшим і менш складним за обчисленням порівняно з лематизацією, яка враховує морфологічні особливості мови.\n")

stemmer = PorterStemmer()


print("######################################")

print(f"\nРегулярні вирази (regular expressions, RegEx).\n")


print(f"\nЗавантажимо пайплайн бібліотеки spacy.\n")
# якщо не працює виконати: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])


print(f"\nВизначимо функцію нормалізації тексту.\n")


def normalize_text(raw_review):

    # Remove html tags
    # match <> and everything in between. [^>] - match everything except >
    text = re.sub("<[^>]*>", " ", raw_review)

    # Remove emails
    # match non-whitespace characters, @ and a whitespaces in the end
    text = re.sub("\\S*@\\S*[\\s]+", " ", text)

    # remove links
    # match http, s - zero or once, //,
    text = re.sub("https?:\\/\\/.*?[\\s]+", " ", text)
    # any char 0-unlimited, whitespaces in the end

    # Convert to lower case, split into individual words
    text = text.lower().split()

    # Replace contractions with their full versions
    text = [contractions.get(word, word) if word in contractions else word  # (word) --> (word,word)
            for word in text]

    # Re-splitting for the correct stop-words extraction
    text = " ".join(text).split()

    # Remove stop words
    text = [word for word in text if not word in stop_words]

    text = " ".join(text)

    # Remove non-letters
    # match everything except letters and '
    text = re.sub("[^a-zA-Z' ]", "", text)

    # Stem words. Need to define porter stemmer above
    # text = [stemmer.stem(word) for word in text.split()]

    # Lemmatize words. Need to define lemmatizer above
    doc = nlp(text)
    text = " ".join([token.lemma_ for token in doc if len(token.lemma_) > 1])

    # Remove excesive whitespaces
    text = re.sub("[\\s]+", " ", text)

    # Join the words back into one string separated by space, and return the result.
    return text


print(f"\nПеревіримо її на тестовому прикладі.\n")

text = """On a quest for the perfedc1112t,,, !!!! <br />%%2%% popcorn to \ncompliment the Whirley Pop.  Don\'t get older, I\'m beginning to \nappreciate the more "natural" popcorn varieties, and I suppose\n that\'s what attracted me to the Arrowhead Mills Organic Yellow\n Popcorn.<br /> <br />I\'m no "organic" food expert.  I just wanted\n some good tasting popcorn.  And, I feel like that\'s what I got.\n  Using the Whirley Pop, with a very small amount of oil, I\'ve had \ngreat results.\n"""  # виправлено текст, замінено лапки на тройні


print('Original text: ', text, '#'*30, sep='\n')

# -----------------виправлено, для роботи у пайтон----
normalized = normalize_text(text)

words = normalized.split()
chunks = [" ".join(words[i:i+9]) for i in range(0, len(words), 10)]

formatted = "\n\n".join(chunks)

print("\nNormalized text:\n", formatted, sep="")


print(f"Для демонстративних цілей зменшимо розмір набору даних до 5 тисяч прикладів")

df = df.groupby('sentiment').sample(2500, random_state=42)

print(f"\nДивимось на зменшений набір даних до 5000: \n {df.shape}")


print(f"\nЗастосуємо препроцесинг до тексту оглядів.\n")

df['text_normalized'] = df['Text'].progress_apply(normalize_text)

print("##############################")
print(f"\nВизначення метрики точності 🐝\n")

print(f"\nBag of Words\n")

print(f"\nРозділимо дані на навчальні й тестові\n")

train_idxs = df.sample(frac=0.8, random_state=42).index
test_idxs = [idx for idx in df.index if idx not in train_idxs]
X_train = df.loc[train_idxs, 'text_normalized']
X_test = df.loc[test_idxs, 'text_normalized']

y_train = df.loc[train_idxs, 'sentiment']
y_test = df.loc[test_idxs, 'sentiment']

print(f"\nСтворюємо й навчаємо об'єкт CountVectorizer\n")

vect = CountVectorizer().fit(X_train)

# len(vect.vocabulary_)
print(f"\nКількість унікальних слів:\n {len(vect.vocabulary_)}\n")

print(f"\nПодивимось на приклад ознак, які було виокремлено. \n")

print(vect.get_feature_names_out()[:5])

print(f"\nПеретворимо навчальні дані на матрицю документ-термін (document-term matrix).\n")

# перевірка
print(f"\nПеревірка\n")

print(type(vect))

print(type(X_test))


X_train_vectorized = vect.transform(X_train)
print(X_train_vectorized.shape)
print(f"\nThis is X_traine_vectorized: \n{X_train_vectorized}")

print(f"\nСтворимо клас моделі й навчимо її на отриманих даних.\n")

model = LogisticRegression(random_state=42)
model.fit(X_train_vectorized, y_train)


print(f"\nРозрахуємо точність моделі.\n")

predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))

print(f"\nДля подальшого швидкого тестування різних варіантів обробки даних напишемо допоміжну функцію.\n")


def get_preds(text_column, algorithm, ngrams=(1, 1)):

    X_train = df.loc[train_idxs, text_column]
    X_test = df.loc[test_idxs, text_column]

    y_train = df.loc[train_idxs, 'sentiment']
    y_test = df.loc[test_idxs, 'sentiment']

    if algorithm == 'cv':
        vect = CountVectorizer(ngram_range=ngrams).fit(X_train)
    elif algorithm == 'tfidf':
        vect = TfidfVectorizer(ngram_range=ngrams).fit(X_train)
    else:
        raise ValueError('Select correct algorithm: `cv` or `tfidf`')

    print('Vocabulary length: ', len(vect.vocabulary_))

    # transform the documents in the training data to a document-term matrix

    X_train_vectorized = vect.transform(X_train)
    print('Document-term matrix shape:', X_train_vectorized.shape)

    model = LogisticRegression(random_state=42)
    model.fit(X_train_vectorized, y_train)

    predictions = model.predict(vect.transform(X_test))

    print('AUC: ', roc_auc_score(y_test, predictions))


print(f"\nПеревіримо результати моделювання на нетокенізованих даних.\n")

print(f"\n Нетокенізовані моделі: \n {get_preds('Text', 'cv')}\n")


print("##########################")
print(f"\nTF-IDF: Term frequency-inverse document frequency (tf-idf)\n")

print(f"\nВикористаємо метод TF-IDF.\n")

print(f"\nThis is normal data: \n{get_preds('text_normalized', 'tfidf')}")

print("\nДля ненормалізованих даних:\n")

print(f"\nThis is not-normal data: \n{get_preds('Text', 'tfidf')}")

print("###########################")

print(f"\nN-Grams\n")

get_preds('text_normalized', 'cv', (1, 2))

get_preds('text_normalized', 'tfidf', (1, 2))

get_preds('text_normalized', 'cv', (2, 2))

get_preds('Text', 'cv', (2, 2))

get_preds('Text', 'tfidf', (2, 2))
