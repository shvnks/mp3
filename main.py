# Authors: Anthony Wong, Derek Lam, Tyler Shanks
# Due: December 6, 2021
# COMP-472

# Importing necessary modules
import csv
import os
import matplotlib.pyplot as plt
import gensim.downloader as api
from gensim.models import KeyedVectors
from tqdm import tqdm


#################################### WordVectorModel Class ####################################

class WordVectorModel:
    # Initializing default method to "word2vec"
    DEFAULT_MODEL = 'word2vec-google-news-300'

    # Initializing default variables
    def __init__(self, model_name=DEFAULT_MODEL):
        self.name = model_name
        self.model = self.load_model()
        self.vocabulary_size = len(list(self.model.index_to_key))
        self.answer_count = 0
        self.incorrect_count = 0

    # Function to keep track of how many correct and incorrect suggestions
    def counters(self, answer, suggestion):
        if answer == suggestion:
            self.answer_count += 1
        if suggestion is not None:
            self.incorrect_count += 1

    # Function to load the model, returns a wordvector
    def load_model(self):
        wv = None
        try:
            wv = KeyedVectors.load('saved_models/' + self.name)
        except:
            wv = api.load(self.name)
        return wv

    # Function to return accuracy
    def accuracy(self):
        return self.answer_count / self.incorrect_count

    # Function to return the model analysis with its name, vocabulary_size, answer_count, incorrect_count and its accuracy
    def select_model_analysis_row(self):
        return self.name, self.vocabulary_size, self.answer_count, self.incorrect_count, self.accuracy()


########################################################################

# Function to read the csv and return each row within the file
def synonyms():
    csv_read = csv.reader(open("synonyms.csv"), delimiter=",")
    return [row for row in csv_read]


# Function to compare 2 words, returns their similarity as a float
def wv_sim(w1, w2, wv):
    return wv.similarity(w1, w2)


# Function to select the word (option) within the list, returns the word (option)
def select_in_model(option, model):
    return option in list(model.index_to_key)


# Function to validate if option is in the model, returns boolean
def check_in_model(options, model):
    for option in options:
        if select_in_model(option, model):
            return True
    return False


# Function to see if the question query and the options query are in the model
def query_verification(question, options, model):

    # Select the query in the model
    question_in_model = select_in_model(question, model)

    # Validate the query in the model
    guessWord_in_model = check_in_model(options, model)
    return not question_in_model or not guessWord_in_model


# Function to pick the best similar word from the model, returns the best_option
def best_sim(question, options, model):

    # Initialize variables
    best_score = 0
    best_option = ""

    # Iterate through the options
    for option in options:

        # If option is not within the model, go back to the beginning of the for loop for the next index
        if not select_in_model(option, model):
            continue

        # Calculate the options score
        option_score = model.similarity(question, option)

        # Compare the option_score and best_score, assign greatest value and word to best variables
        if option_score > best_score:
            best_score = option_score
            best_option = option
    return best_option


# Function to retrieve the CSV data, returns the csv row data
def csv_row_data(test_details):
    csv_rows = []

    # Iterating through the test_details to retrieve the question, answer, option0, option1, option2, option3
    for test_detail in test_details:
        row, wv_suggestion = test_detail
        question, answer, option0, option1, option2, option3 = row
        label = query_label(answer, wv_suggestion)
        csv_row = (question, answer, wv_suggestion, label)
        csv_rows.append(csv_row)
    return csv_rows


# Function to return the label (str)
def query_label(answer, wv_suggestion):

    # Comparing the answer to the wordvector suggestion
    if answer == wv_suggestion:
        return 'correct'
    if wv_suggestion is None:
        return 'guess'
    else:
        return 'wrong'


# Function which verifies the word vector suggestion, returns the test_details list
def suggestion_verification(word_vector):

    # Initialize variables
    wv = word_vector.model
    test_details = []

    # Iterate through the rows of synonyms from the csv
    for row in synonyms():
        question, answer, option0, option1, option2, option3 = row
        options = [option0, option1, option2, option3]
        wv_suggestion = None

        # Validating the query, applying the best similarity score to the wordvector suggestion
        if not query_verification(question, options, wv):
            wv_suggestion = best_sim(question, options, wv)

        # Storing details
        test_details.append((row, wv_suggestion))
        word_vector.counters(answer, wv_suggestion)
    return test_details


#################################### File management ####################################

# Initializing mkdir function (Making directory)
def mkdir(dir):

    # Validate path, if it doesn't exist, make the directory
    if not os.path.exists(dir):
        os.makedirs(dir)


# Initializing rm_file function (Removing file)
def rm_file(file_path):

    # Validate path, if it exists, remove the file
    if os.path.isfile(file_path):
        os.remove(file_path)


# Initializing del_output_files function
def del_output_files():
    file_path = 'outputs/model-details.csv'
    rm_file(file_path)
    file_path = 'outputs/analysis.csv'
    rm_file(file_path)


# Function to write the output_analysis to a csv file
def output_analysis(word_vector):
    writer = csv.writer(open('outputs/analysis.csv', 'a+'), delimiter=',')
    writer.writerows([word_vector.select_model_analysis_row()])


# Function to write the output_test_details to a csv file
def output_test_details(model, synonym_test_details):
    writer = csv.writer(open('outputs/' + model.name + '-details.csv', 'a+'), delimiter=',')
    csv_rows = csv_row_data(synonym_test_details)
    writer.writerows(csv_rows)


#################################### Graphing ####################################

# Creating accuracy graph
def plot_accuracy(statistics):

    # Generating x and y values
    x_values = statistics[0]
    x_values.append('Random')
    y_values = statistics[1]
    y_values.append(0.25)

    # Plot the graph and adjust its size
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.plot(x_values, y_values)
    plt.xlabel('Corpus name')
    plt.ylabel('Accuracy')
    plt.title('Accuracy plot')

    # Save output to a png
    plt.savefig('outputs/accuracy.png')


# Creating guess graph
def plot_guesses(guesses_statistics):

    # Generating x and y values
    x_values = guesses_statistics[0]
    y_values = guesses_statistics[1]

    # Plot the graph and adjust its size
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.plot(x_values, y_values)
    plt.xlabel('Corpus name')
    plt.ylabel('Guess count')
    plt.title('Guesses plot')

    # Save output to a png
    plt.savefig('outputs/guesses.png')


#################################### Main Method ####################################

def main():
    print(
        '####################################\n##### Welcome to COMP-472 MP3! #####\n####################################\n')
    del_output_files()

    # Defining word vector models
    print('Generating WordVectorModels...')
    print('Generating google_300 WordVectorModels...')
    google_300 = WordVectorModel()
    print('Generating wiki_200 WordVectorModels...')
    wiki_200 = WordVectorModel('glove-wiki-gigaword-200')
    print('Generating wiki_300 WordVectorModels...')
    wiki_300 = WordVectorModel('glove-wiki-gigaword-300')
    print('Generating twitter_25 WordVectorModels...')
    twitter_25 = WordVectorModel('glove-twitter-25')
    print('Generating twitter_200 WordVectorModels...')
    twitter_200 = WordVectorModel('glove-twitter-200')

    # Initializing an array with all the word vector models
    models = [google_300, wiki_200, wiki_300, twitter_25, twitter_200]

    # Initializing an array for accuracy and guess for graphing
    accuracy_statistics = [[], []]
    guesses_statistics = [[], []]

    # Generating all the statistics for each vector model
    for model in tqdm(models, desc='Iterating through models'):
        print('\nGenerating statistics for model ' + model.name + '\n')
        test_details = suggestion_verification(model)

        # Generating output
        print('Generating output...')
        output_test_details(model, test_details)
        output_analysis(model)

        # Generating statistics
        print('Generating statistics...')
        accuracy_statistics[0].append(model.name)
        accuracy_statistics[1].append(model.accuracy())
        guesses_statistics[0].append(model.name)
        guesses_statistics[1].append(81 - model.incorrect_count)

    # Generating accuracy and guess graph
    print('\nGenerating accuracy and guess graph...')
    plot_accuracy(accuracy_statistics)
    plot_guesses(guesses_statistics)

    print('MP3 Completed!')

if __name__ == '__main__':
    main()
