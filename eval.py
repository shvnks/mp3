# From gensim.models import Word2Vec
import panda as pd
from gensim.models import KeyedVectors
import gensim.downloader as api

#################################### Task 1 ####################################

# Importing the file
def synonyms():
    csv = pd.read_csv('synonyms.csv', delimiter=',')
    return [row for row in csv]

# Function to compare 2 words
def wv_sim(w1, w2, wv):
    return wv.similarity(w1, w2)

# Function to select the model list
def select_in_model(option, model):
    return option in list(model.index_to_key)

# Function to validate if option is in the model
def check_in_model(options, model):
    for option in options:
        if select_in_model(option, model):
            return True
    return False

# Function to see if the question query and the options query are in the model
def query_verification(question, options, model):
    question_in_model = select_in_model(question, model)
    guessWord_in_model = check_in_model(options, model)
    return not question_in_model or not guessWord_in_model

# Function to pick the closest similar guess word to the question query
def best_sim(question, options, model):
    best_score = 0
    best_option = ""
    for option in options:
        option_score = model.similarity(question, option)
        if option_score > best_score:
            best_score = option_score
            best_option = option
    return best_option

# Function to retrieve the CSV data
def csv_row_data(test_details):
    csv_rows = []
    for test_detail in test_details:
        row, wv_suggestion = test_detail
        question, answer, option0, option1, option2, option3 = row
        label = query_label(answer, wv_suggestion)
        csv_row = (question, answer, wv_suggestion, label)
        csv_rows.append(csv_row)
    return csv_rows

# Function which returns the label
def query_label(answer, wv_suggestion):
    if answer == wv_suggestion:
        return "correct"
    if wv_suggestion is None:
        return "guess"
    else:
        return "wrong"

# Function which verifies the word vector
def suggestion_verification(word_vector):
    wv = word_vector.model
    test_details = []
    for row in synonyms():
        question, answer, option0, option1, option2, option3 = row
        options = [option0, option1, option2, option3]

        wv_suggestion = None
        if not query_verification(question, options, wv):
            wv_suggestion = best_sim(question, options, wv)

        test_details.append(row, wv_suggestion)
        word_vector.update_statistics(answer, wv_suggestion) #Must change the .update_statistics!!!!
    return test_details

#################################### End Task 1 ####################################

