# Authors: Anthony Wong, Derek Lam, Tyler Shanks
# Due: December 6, 2021
# COMP-472

# From gensim.models import Word2Vec
import pandas as pd
import os
import matplotlib.pyplot as plt
import gensim.downloader as api
from gensim.models import KeyedVectors
from tqdm import tqdm

#################################### WordVectorModel Class ####################################

class WordVectorModel:
    DEFAULT_MODEL = 'word2vec-google-news-300'

    def __init__(self, model_name=DEFAULT_MODEL):
        self.name = model_name
        self.model = self.load_model()
        self.vocabulary_size = len(list(self.model.index_to_key))
        self.answer_count = 0
        self.incorrect_count = 0

    def counters(self, answer, suggestion):
        if answer == suggestion:
            self.answer_count += 1
        if suggestion is not None:
            self.incorrect_count += 1

    def load_model(self):
        wv = None
        try:
            wv = KeyedVectors.load('saved_models/' + self.name)
        except:
            wv = api.load(self.name)
            # wv.save(self.name) remove????
        return wv

    def accuracy(self):
        return self.answer_count / self.incorrect_count

    def select_model_analysis_row(self):
        return self.name, self.vocabulary_size, self.answer_count, self.incorrect_count, self.accuracy()


########################################################################

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
    best_option = ''
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
        return 'correct'
    if wv_suggestion is None:
        return 'guess'
    else:
        return 'wrong'

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
        word_vector.counters(answer, wv_suggestion)
    return test_details


#################################### File management ####################################

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def rm_file(file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)


def del_output_files():
    file_path = 'outputs/model-details.csv'
    rm_file(file_path)
    file_path = 'outputs/analysis.csv'
    rm_file(file_path)


def output_analysis(word_vector):
    writer = csv.writer(open('outputs/analysis.csv', 'a+'), delimiter=',')
    writer.writerows([word_vector.select_model_analysis_row()])

def output_test_details(model, synonym_test_details):
    writer = csv.writer(open('outputs/' + model.name + '-details.csv', 'a+'), delimiter=',')
    csv_rows = csv_row_data(synonym_test_details)
    writer.writerows(csv_rows)


#################################### Graphing ####################################

def plot_accuracy(statistics):
    x_values = statistics[0]
    x_values.append('Random')
    y_values = statistics[1]
    y_values.append(0.25)
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.plot(x_values, y_values)
    plt.xlabel('Corpus name')
    plt.ylabel('Accuracy')
    plt.show()

def plot_guesses(guesses_statistics):
    x_values = guesses_statistics[0]
    y_values = guesses_statistics[1]
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.plot(x_values, y_values)
    plt.xlabel('Corpus name')
    plt.ylabel('Guess count')
    plt.show()

########################################################################

def main():
    print('####################################\n##### Welcome to COMP-472 MP3! #####\n####################################\n')
    # del_output_files()

    # Defining word vector models
    google_300 = WordVectorModel()
    wiki_200 = WordVectorModel('glove-wiki-gigaword-200')
    wiki_300 = WordVectorModel('glove-wiki-gigaword-300')
    twitter_25 = WordVectorModel('glove-twitter-25')
    twitter_200 = WordVectorModel('glove-twitter-200')

    models = [google_300, wiki_200, wiki_300, twitter_25, twitter_200]

    accuracy_statistics = [[], []]
    guesses_statistics = [[], []]
    for model in tqdm(models, desc = 'Iterating through models'):
        print('Generating statistics for model ' + model.name)
        test_details = suggestion_verification(model)

        output_test_details(model, test_details)
        output_analysis(model)

        accuracy_statistics[0].append(model.name)
        guesses_statistics[0].append(model.name)
        accuracy_statistics[1].append(model.select_model_accuracy())
        guesses_statistics[1].append(81 - model.non_guess_answer_count)

    plot_accuracy(accuracy_statistics)
    plot_guesses(guesses_statistics)

if __name__ == '__main__':
    main()
