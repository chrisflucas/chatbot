#!/usr/bin/env python
# -*- coding: utf-8 -*-

# PA6, CS124, Stanford, Winter 2016
# v.1.0.2
# Original Python code by Ignacio Cases (@cases)
# Ported to Java by Raghav Gupta (@rgupta93) and Jennifer Lu (@jenylu)
######################################################################
import csv
import math
import re
import numpy as np

from movielens import ratings
from random import randint
from PorterStemmer import PorterStemmer
from copy import copy, deepcopy


class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    #############################################################################
    # `moviebot` is the default chatbot. Change it to your chatbot's name       #
    #############################################################################
    def __init__(self, is_turbo=False):
      def extract_movie_titles():
        tits = []
        for movie_array in self.titles:
          title = movie_array[0]
          m = re.search('\([1-3][0-9]{3}\)', title)
          if not m: m = re.search('\([1-3][0-9]{3} ?-\)', title) # Edge case regex for (2007-)
          if not m: m = m = re.search('\([1-3][0-9]{3} ?- ?[1-3][0-9]{3}\)', title) # Edge case regex for (2007-2013)
          if m: 
            year = m.group(0)
            tits.append(title[:title.index(year)].strip())
          else: tits.append(title)
        return tits
      self.name = 'l\'belle'
      self.is_turbo = is_turbo
      self.porter_stemmer = PorterStemmer()
      self.read_data()
      self.movie_titles = extract_movie_titles()
      self.user_vector = []
      self.NUM_MOVIES_THRESHOLD = 5

    #############################################################################
    # 1. WARM UP REPL
    #############################################################################

    def greeting(self):
      """chatbot greeting message"""
      #############################################################################
      # TODO: Write a short greeting message                                      #
      #############################################################################

      greeting_message = 'What do you want?'

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################

      return greeting_message

    def goodbye(self):
      """chatbot goodbye message"""
      #############################################################################
      # TODO: Write a short farewell message                                      #
      #############################################################################

      goodbye_message = 'Au revoir!'

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################

      return goodbye_message


    #############################################################################
    # 2. Modules 2 and 3: extraction and transformation                         #
    #############################################################################

    def process(self, input):
      """Takes the input string from the REPL and call delegated functions
      that
        1) extract the relevant information and
        2) transform the information into a response to the user
      
      Control flow following spec.  Pulled add_movie(self, user_input) functionality
      into main function because we need information about the validity of the user_input 
      in order to inform our response. 

      """
      #############################################################################
      # TODO: Implement the extraction and transformation in this method, possibly#
      # calling other functions. Although modular code is not graded, it is       #
      # highly recommended                                                        #
      #############################################################################
      if self.is_turbo == True:
        response = 'processed %s in creative mode!!' % input
      else:
        #response = 'processed %s in starter mode' % input
        movie = self.extract_movie(input)
        if len(movie) == 0: return 'Sorry, I don\'t understand. Tell me about a movie that you have seen.'
        if len(movie) > 1: return'Please tell me about one movie at a time. Go ahead.'
        movie = movie[0]

        sentiment = self.extract_sentiment(input)
        if sentiment == 3: return "I\'m sorry, I\'m not quite sure if you liked {}. Tell me more about \"{}\"".format(movie, movie)
        if sentiment > 3: response = "You liked \"{}\". Thank you!".format(movie)
        if sentiment < 3: response = "You did not like \"{}\". Thank you!".format(movie)
        
        self.add_to_vector(movie, sentiment)
        if len(self.user_vector) >= self.NUM_MOVIES_THRESHOLD: 
          response +=  " That\'s enough for me to make a recommendation."
          recommendation = self.recommend()
          response += " I suggest you watch \"{}\".".format(recommendation)
        else:  response += " Tell me about another movie you have seen."
      return response

    # def add_movie(self, user_input):
    #   movie = extract_movie(user_input)
    #   sentiment = extract_sentiment(user_input)
    #   add_to_vector(movie, sentiment)

    def extract_movie(self, user_input):
      return re.findall('"([^"]*)"', user_input)

    def extract_sentiment(self, user_input):
      num_pos = 0
      num_neg = 0

      for w in user_input.split(" "):
        word = self.porter_stemmer.stem(w)
        if word in self.sentiment.keys():
          sentiment = self.sentiment[word]
          if sentiment == "pos":
            num_pos += 1
          elif sentiment == "neg":
            num_neg += 1

      if num_pos > num_neg:
        return 5
      elif num_neg > num_pos:
        return 1
      else:
        return 3


    def add_to_vector(self, movie, sentiment):
      self.user_vector.append((movie, sentiment))


    #############################################################################
    # 3. Movie Recommendation helper functions                                  #
    #############################################################################

    def read_data(self):
      """Reads the ratings matrix from file"""
      # This matrix has the following shape: num_movies x num_users
      # The values stored in each row i and column j is the rating for
      # movie i by user j
      self.titles, self.ratings = ratings()
      reader = csv.reader(open('data/sentiment.txt', 'rb'))
      self.sentiment = dict(reader)

      #Go through all sentiment words and change to stemmed version
      for sentiment_word in self.sentiment.keys():
        stemmed_word = self.porter_stemmer.stem(sentiment_word)
        self.sentiment[stemmed_word] = self.sentiment.pop(sentiment_word)


    def binarize(self):
      """Modifies the ratings matrix to make all of the ratings binary"""

      pass


    def distance(self, u, v):
      """Calculates a given distance function between vectors u and v"""
      # TODO: Implement the distance function between vectors u and v]
      # Note: you can also think of this as computing a similarity measure

      pass


    def recommend(self):
      """Generates a list of movies based on the input vector u using
      collaborative filtering"""
      # TODO: Implement a recommendation function that takes a user vector u
      # and outputs a list of movies recommended by the chatbot
      formmated_vec = self.format_vec()
      ratings_matrix = self.generate_matrix(formmated_vec)
      ## Will need to write more functions for computing sims then returning best recommendation. ##
      pass

    def format_vec(self):
      '''
      Creates user vector based on their ratings for each movie in our movie matrix.
      This is normally going to be a VERY sparse vector. Used for computing cosine 
      similarity.
      '''
      num_movies = len(self.movie_titles)
      fv = [0] * num_movies
      for movie, rating in self.user_vector:
        curr_pos = self.movie_titles.index(movie)
        fv[curr_pos] = rating
      return fv

    def generate_matrix(self, user_vector):
      means_vector = [0]*len(self.titles)
      for index, movie in enumerate(self.ratings):
        total = np.sum(movie) + user_vector[index]
        length = np.count_nonzero(movie)
        if user_vector[index] != 0:
          length += 1
        if length > 0:
          means_vector[index] = total/length

      mean_centered_matrix = deepcopy(self.ratings)
      for movie_index, movie in enumerate(self.ratings):
        for rating_index, movie_rating in enumerate(movie):
          if self.ratings[movie_index][rating_index] != 0:
            centered_rate = self.ratings[movie_index][rating_index] - means_vector[movie_index]
            mean_centered_matrix[movie_index][rating_index] = centered_rate

      return mean_centered_matrix


    #############################################################################
    # 4. Debug info                                                             #
    #############################################################################

    def debug(self, input):
      """Returns debug information as a string for the input string from the REPL"""
      # Pass the debug information that you may think is important for your
      # evaluators
      debug_info = 'debug info'
      return debug_info


    #############################################################################
    # 5. Write a description for your chatbot here!                             #
    #############################################################################
    def intro(self):
      return """
      Your task is to implement the chatbot as detailed in the PA6 instructions.
      Remember: in the starter mode, movie names will come in quotation marks and
      expressions of sentiment will be simple!
      Write here the description for your own chatbot!
      """


    #############################################################################
    # Auxiliary methods for the chatbot.                                        #
    #                                                                           #
    # DO NOT CHANGE THE CODE BELOW!                                             #
    #                                                                           #
    #############################################################################

    def bot_name(self):
      return self.name


if __name__ == '__main__':
    Chatbot()
