import nltk
from scipy import spatial
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import random
import csv
import re
import math

# Dictionary of Word with index
phrase_word2index = {}
name_input_word2index = {}
question_word2index = {}
talk_input_word2index = {}

# Threshold Value
intent_threshold = 0.74
question_threshold = 0.85

# Key Phrases Database
talk_phrase_corpus = {}
name_phrase_corpus = {}

# Intent Database
talk_input_corpus = {}
talk_answer_database = {}
name_input_corpus = {}
name_input_label_database = {}
question_corpus = {}
answer_database = {}

# Initialise system variables
num_word_in_phrase = 0
stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()
user = ""
name_intent_stopwords = ["hi", "please", "hello", "pls", "hey"]
posmap = {
    'ADJ': 'a',
    'ADV': 'r',
    'NOUN': 'n',
    'VERB': 'v'
}

# Download package
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('universal_tagset')

# Compute Term Log Frequency
def compute_TF(word_dict):
    tf_dict = {}
    for word, frequency in word_dict.items():
        tf_dict[word] = math.log10(1 + frequency)
    return tf_dict

# Compute Inverse Document Frequency
def compute_IDF(doc_list):
    num_doc = len(doc_list)
    idf_dict = dict.fromkeys(doc_list[0].keys(), 0)
    for doc in doc_list:
        for word, val in doc.items():
            if val > 0:
                idf_dict[word] += 1
    for word, val in idf_dict.items():
        idf_dict[word] = math.log10(num_doc / float(val))
    return idf_dict

# Compute Term Frequency-Inverse Document Frequency
def compute_TFIDF(tf, idfs):
    tfidf = {}
    for word, tf_value in tf.items():
        tfidf[word] = tf_value * idfs[word]
    return tfidf

# Remove punctuation in word
def remove_punc_in_sentence(list):
    new_list = []
    for word in list:
        new = re.sub(r'[^\w\s]', '', word)
        if new != "":
            new_list.append(new)
    return new_list

# Lemmatization and exclude the word with punctuation
def lemmatizing_sentence(sentence, remove_punc):
    tokens = word_tokenize(sentence)
    if remove_punc:
        tokens = remove_punc_in_sentence(tokens)
    post = nltk.pos_tag(tokens, tagset='universal')
    normalized_word_in_sentence = []
    for token in post:
        word = token[0].lower()
        tag = token[1]
        if tag in posmap.keys():
            normalized_word_in_sentence.append(lemmatizer.lemmatize(word, posmap[tag]))
        else:
            normalized_word_in_sentence.append(lemmatizer.lemmatize(word))
    return normalized_word_in_sentence

# Add word in phrase to dictionary
def add_phrase_word(word, word2index):
    global num_word_in_phrase
    if word not in word2index:
        word2index[word] = num_word_in_phrase
        num_word_in_phrase += 1

# Add word in question to dictionary
def add_question_word(word, word2index, num_word_question):
    if word not in question_word2index:
        word2index[word] = num_word_question

# Read name phrase and small talk phrase dataset using stemmer, create vocabulary
def read_phrase_file(file, corpus):
    with open(file, 'r', encoding='utf-8') as f:
        first_line = True
        readCSV = csv.reader(f, delimiter=',')
        num_phrases = 0
        for row in readCSV:
            if first_line:
                first_line = False
                continue
            key_phrase = row[0]
            tokens_without_punc = remove_punc_in_sentence(word_tokenize(key_phrase))
            normalized_word_phrase = [stemmer.stem(word.lower()) for word in tokens_without_punc]
            corpus[num_phrases] = normalized_word_phrase
            num_phrases += 1
            for word in normalized_word_phrase:
                add_phrase_word(word, phrase_word2index)

# Read name intent and small talk intent dataset using lemmatizing, create vocabulary
def read_intent_file(file, question_corpus, extra_database, word2index, question_index, extra_index):
    with open(file, 'r', encoding='utf-8') as f:
        first_line = True
        num_question = 0
        unique_word_question = 0
        readCSV = csv.reader(f, delimiter=',')
        for row in readCSV:
            if first_line:
                first_line = False
                continue
            question = row[question_index]
            normalized_word_in_question = lemmatizing_sentence(question, remove_punc=True)
            question_corpus[num_question] = normalized_word_in_question
            extra_database[num_question] = row[extra_index]
            num_question += 1
            for word in normalized_word_in_question:
                add_question_word(word, word2index, unique_word_question)
                unique_word_question += 1

# Use dictionary to create the term-document matrix of the corpus for easy implementation
def setup_term_document_matrix(corpus, word2index):
    term_document_dict = []
    for index, que in corpus.items():
        word_dict = dict.fromkeys(word2index.keys(), 0)
        for word in corpus[index]:
            word_dict[word] += 1
        term_document_dict.append(word_dict)
    return term_document_dict

# For each items in term document matrix, compute the cosine similarity with the query
# return items in highest similarity
def get_highest_similarity_item(query_doc, dict):
    similarity = {}
    for i in range(len(dict)):
        doc = [num for num in dict[i].values()]
        sim = 1 - spatial.distance.cosine(query_doc, doc)
        similarity[i] = sim
    similarity = sorted(similarity.items(), key=lambda x: x[1], reverse=True)
    max_val = similarity[0][1]
    selected_item = [item for item in similarity if item[1] == max_val]
    return max_val, selected_item

# Handle name intent
def handle_name(normalized_word_in_query, original_query):
    # Map search query on the name inputs collection-induced vector space
    query_dict = dict.fromkeys(name_input_word2index.keys(), 0)
    for word in normalized_word_in_query:
        if word in query_dict:
            query_dict[word] += 1
    # Apply tf-idf weighting to the tokens in query
    tf_dict = compute_TF(query_dict)
    query_dict = compute_TFIDF(tf_dict, name_input_idfs)
    query_doc = [num for num in query_dict.values()]
    value, selected_question = get_highest_similarity_item(query_doc, name_input_term_doc_dict)
    index = selected_question[0][0]
    global user
    if name_input_label_database[index] == '1':
        if user == "":
            return "You currently don't have a name. Can you tell me your name, please? "
        return "Your name is {name}. Anything you would like to know? ".format(name=user)
    else:
        # Name Extraction
        tokenize_query = remove_punc_in_sentence(word_tokenize(original_query))
        tokens_without_sw = [word for word in tokenize_query if not word.lower() in name_intent_stopwords]
        detect_pos_tag = nltk.pos_tag(tokens_without_sw)
        is_name = False
        name = ""
        for index, tag in enumerate(detect_pos_tag):
            if is_name:
                if tag[0] == 'am' and index != len(detect_pos_tag) - 1:
                    continue
                if tag[0] == 'm' and index != len(detect_pos_tag) - 1:
                    continue
                name = tag[0]
                break
            if tag[1] == 'TO' or tag[1] == 'PRP' or tag[1] == 'VBZ':
                is_name = True
        if name == "":
            return "Please input a valid name "
        else:
            user = name
            totalNameResponse = []
            totalNameResponse.append("Hello {name}, how can I help you? ".format(name=user))
            totalNameResponse.append("Hi {name}, nice to meet you. ".format(name=user))
            totalNameResponse.append("Hello {name}, how's everything going? ".format(name=user))
            totalNameResponse.append("Hi {name}, how can I help you? ".format(name=user))
            return random.choice(totalNameResponse)

# Handle small talk intent
def handle_talk(normalized_word_in_query):
    # Map search query on the small talk inputs collection-induced vector space
    query_dict = dict.fromkeys(talk_input_word2index.keys(), 0)
    for word in normalized_word_in_query:
        if word in query_dict:
            query_dict[word] += 1
    # Apply tf-idf weighting to the tokens in query
    tf_dict = compute_TF(query_dict)
    query_dict = compute_TFIDF(tf_dict, talk_input_idfs)
    query_doc = [num for num in query_dict.values()]
    value, selected_question = get_highest_similarity_item(query_doc, talk_input_term_doc_dict)
    if len(selected_question) > 1:
        random_answer = random.choice(selected_question)
        response = talk_answer_database[random_answer[0]] + " "
    else:
        answer = selected_question[0]
        response = talk_answer_database[answer[0]] + " "
    return response

# Handle question answering intent
def answer_question(normalized_word_in_query):
    # Map search query on the question collection-induced vector space
    query_dict = dict.fromkeys(question_word2index.keys(), 0)
    for word in normalized_word_in_query:
        if word in query_dict:
            query_dict[word] += 1
    # Apply tf-idf weighting to the tokens in query
    tf_dict = compute_TF(query_dict)
    query_dict = compute_TFIDF(tf_dict, question_idfs)
    query_doc = [num for num in query_dict.values()]
    if len(query_doc) != query_doc.count(0):
        value, selected_question = get_highest_similarity_item(query_doc, question_term_doc_dict)
        if value >= question_threshold:
            if len(selected_question) > 1:
                random_answer = random.choice(selected_question)
                response = answer_database[random_answer[0]] + " "
            else:
                answer = selected_question[0]
                response = answer_database[answer[0]] + " "
        else:
            response = "I am not able to answer this question at the moment. "
    else:
        response = "I am not able to answer this question at the moment. "
    return response


# Setup data corpus, add to word2index
read_phrase_file('talk_intent_key_phrases.csv', talk_phrase_corpus)
print("----Talk Phrases Dataset Setup----")
read_phrase_file('name_intent_key_phrases.csv', name_phrase_corpus)
print("----Name Phrases Dataset Setup----")
read_intent_file('name_dataset.csv', name_input_corpus, name_input_label_database, name_input_word2index, 0, 1)
print("----Name Intent Dataset Setup----")
read_intent_file('small_talk_dataset.csv', talk_input_corpus, talk_answer_database, talk_input_word2index, 0, 1)
print("----Small Talk Intent Dataset Setup----")
read_intent_file("COMP3074-CW1-Dataset.csv", question_corpus, answer_database, question_word2index, 1, 2)
print("----Provided Question Answering Intent Setup----")

# Use the vocabulary to compute the term-document matrix of the corpus
talk_phrases_term_doc_dict = setup_term_document_matrix(talk_phrase_corpus, phrase_word2index)
name_phrases_term_doc_dict = setup_term_document_matrix(name_phrase_corpus, phrase_word2index)
name_input_term_doc_dict = setup_term_document_matrix(name_input_corpus, name_input_word2index)
talk_input_term_doc_dict = setup_term_document_matrix(talk_input_corpus, talk_input_word2index)
question_term_doc_dict = setup_term_document_matrix(question_corpus, question_word2index)

# Apply TF-IDF weighting to the term-document matrix
phrase_idfs = compute_IDF(talk_phrases_term_doc_dict + name_phrases_term_doc_dict)
for i in range(len(talk_phrases_term_doc_dict)):
    tf_dict = compute_TF(talk_phrases_term_doc_dict[i])
    talk_phrases_term_doc_dict[i] = compute_TFIDF(tf_dict, phrase_idfs)
for i in range(len(name_phrases_term_doc_dict)):
    tf_dict = compute_TF(name_phrases_term_doc_dict[i])
    name_phrases_term_doc_dict[i] = compute_TFIDF(tf_dict, phrase_idfs)

question_idfs = compute_IDF(question_term_doc_dict)
for i in range(len(question_term_doc_dict)):
    tf_dict = compute_TF(question_term_doc_dict[i])
    question_term_doc_dict[i] = compute_TFIDF(tf_dict, question_idfs)

name_input_idfs = compute_IDF(name_input_term_doc_dict)
for i in range(len(name_input_term_doc_dict)):
    tf_dict = compute_TF(name_input_term_doc_dict[i])
    name_input_term_doc_dict[i] = compute_TFIDF(tf_dict, name_input_idfs)

talk_input_idfs = compute_IDF(talk_input_term_doc_dict)
for i in range(len(talk_input_term_doc_dict)):
    tf_dict = compute_TF(talk_input_term_doc_dict[i])
    talk_input_term_doc_dict[i] = compute_TFIDF(tf_dict, talk_input_idfs)

print("All preparation is done!")
print("----------------------------------Chatbot Start----------------------------------------------------------")

# Handling query
stop = False
conversation_string = "Hi! What is your name? "
while not stop:
    # Handle User Input
    query = input(conversation_string)
    if query == "":
        conversation_string = "Please input something. "
        continue
    if query == "bye":
        stop = True

    # Query Normalisation
    query_token_without_punctuation = remove_punc_in_sentence(word_tokenize(query))
    normalized_word_in_query = [stemmer.stem(word.lower()) for word in query_token_without_punctuation]

    # Map query on the phrase collection-induced vector space
    query_dict = dict.fromkeys(phrase_word2index.keys(), 0)
    for word in normalized_word_in_query:
        if word in query_dict:
            query_dict[word] += 1

    # Apply TF-IDF weighting to the tokens in query
    tf_dict = compute_TF(query_dict)
    query_dict = compute_TFIDF(tf_dict, phrase_idfs)
    query_doc = [num for num in query_dict.values()]

    # Compare similarity between query and phrases in name and small talk intent
    sim_with_name = 0
    sim_with_talk = 0

    # Check whether the input query contains certain dictionary words, if not set similarity to 0
    if len(query_doc) != query_doc.count(0):
        sim_with_name, _ = get_highest_similarity_item(query_doc, name_phrases_term_doc_dict)
        sim_with_talk, _ = get_highest_similarity_item(query_doc, talk_phrases_term_doc_dict)

    # Match the query to intents
    if sim_with_name >= intent_threshold and sim_with_name >= sim_with_talk:
        conversation_string = handle_name(lemmatizing_sentence(query, remove_punc=True), query)
    elif sim_with_talk >= intent_threshold:
        conversation_string = handle_talk(lemmatizing_sentence(query, remove_punc=True))
    else:
        conversation_string = answer_question(lemmatizing_sentence(query, remove_punc=True))
