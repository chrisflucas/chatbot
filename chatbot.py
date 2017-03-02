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
import string
import random
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
          title = self.format_movie(movie_array[0])
          m = re.search("([\w :;,.\'\"&\-!/?+*\[\]\(\)\{\}]+)(, The)", title)
          if m:
            title = "The " + m.group(1)
          tits.append(title.strip())
          # m = re.search('\([1-3][0-9]{3}\)', title)
          # if not m: m = re.search('\([1-3][0-9]{3} ?-\)', title) # Edge case regex for (2007-)
          # if not m: m = m = re.search('\([1-3][0-9]{3} ?- ?[1-3][0-9]{3}\)', title) # Edge case regex for (2007-2013)
          # if m: 
          #   year = m.group(0)
          #   tits.append(title[:title.index(year)].strip())
          # else: tits.append(title)
        return tits
      self.name = 'l\'belle'
      self.is_turbo = is_turbo
      self.porter_stemmer = PorterStemmer()
      self.read_data()
      self.movie_titles = extract_movie_titles()
      self.user_vector = []
      self.NUM_MOVIES_THRESHOLD = 5
      self.ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
      self.catch_all =["Let's get back to movies.", "Okay, got it.", "Hm that's not really what I want to talk about right now."]
      self.can_array=["I'm sorry I don't know how to{}.", "I can't{}.", "I won't{}", "How do you{}?"]
      self.where_array=["I'm sorry I don't know where{} is...", "I'll check the map for{}", "I'm not familiar with{}", "I could not tell you where{} is."]
      self.what_array=["I don't know what{} is...", "Who knows what{} is?", "I'll look up{} and see what I find.", "I'll check{} out and get back to you."]
      self.unknown_movie=['I\'m sorry, I don\'t think I have \"{}\"" in my database! Tell me about another movie that you have seen.', \
        'I haven\'t heard of \"{}\"... I wonder if it\'s good.', 
        'Is \"{}\"" the one where the girl meets the boy and...hm maybe not, I can\'t quite remember.'
      ]
      self.spelling_clairifcation = ''
      self.original_input = ''
      self.series_clarification = ''
      self.year_clarification = ''
      self.affirmations = ['yeah','yes','y','yea','ya','yes i did', 'i did','mhmm','yep', 'correct']
      self.negations = ['no','nope','no i didn\'t', 'no i didnt', 'nah', 'no i did not', 'wrong', 'n']

      #print self.movie_titles

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
    def deleteEdits(self, word):
      if len(word) <= 0: return []
      word = "<"+word
      ret=[]
      for i in xrange(1, len(word)):
        corruptLetters = word[i-1:i+1]
        correctLetters = corruptLetters[:-1]
        correction = "%s%s" % (word[1:i], word[i+1:])
        ret.append(correction)
      return ret

    def insertEdits(self, word):
      """Returns a list of edits of 1-insert distance words and rules used to generate them."""
      # TODO: write this
      # Tip: you might find EditModel.ALPHABET helpful
      # Tip: If inserting the letter 'a' as the second character in the word 'test', the corrupt
      #      signal is 't' and the correct signal is 'ta'. See slide 17 of the noisy channel model.
      word = "<" + word # append start token
      ret = []
      for letter in self.ALPHABET:
        for i in range(1,len(word)+1): # +1 for inserting at end.
          corruptLetters = word[i-1]
          correctLetters = word[i-1] + letter
          correction = word[1:i] + letter + word[i:]
          ret.append(correction)
      return ret

    def transposeEdits(self, word):
      """Returns a list of edits of 1-transpose distance words and rules used to generate them."""
      # TODO: write this
      # Tip: If tranposing letters 'te' in the word 'test', the corrupt signal is 'te'
      #      and the correct signal is 'et'. See slide 17 of the noisy channel model.
      ret = []
      for i in range(len(word) - 1):
        corruptLetters = word[i:i+2]
        correctLetters = word[i:i+2][::-1]
        correction = "%s%s%s" % (word[:i], correctLetters,word[i+2:])
        ret.append(correction)
      return ret

    def replaceEdits(self, word):
      """Returns a list of edits of 1-replace distance words and rules used to generate them."""
      # TODO: write this
      # Tip: you might find EditModel.ALPHABET helpful
      # Tip: If replacing the letter 'e' with 'q' in the word 'test', the corrupt signal is 'e'
      #      and the correct signal is 'q'. See slide 17 of the noisy channel model.
      ret = []
      for letter in self.ALPHABET:
        for i in range(len(word)):
          if letter == word[i]: continue
          corruptLetters = word[i]
          correctLetters = letter
          correction = correction = word[:i] + letter + word[i+1:]
          ret.append(correction)
      return ret

    def edits(self, word):
      """Returns a list of tuples of 1-edit distance words """
      return  self.deleteEdits(word) + \
        self.insertEdits(word) + \
        self.transposeEdits(word) + \
        self.replaceEdits(word)

    def find_closest_movie(self, movie):
      potentials = []
      misspelled = True

      #if it's in a series
      for title in self.movie_titles:
        if movie in title:
          potentials.append(title)
          misspelled = False
      if not misspelled: return potentials, False

      #if they added an article
      format_articles = [movie.split("An "), movie.split("A "), movie.split("The ")]
      articles = ["An", "A", "The"]
      for ind, bad_format in enumerate(format_articles):
        if len(bad_format)>1:
          movie_title = bad_format[1] + ", " + articles[ind]
          if movie_title in self.movie_titles:
            return [bad_format[1]], False

      all_corrections = self.edits(movie)
      for correction in all_corrections:
        if correction in self.movie_titles:
          return correction, True

      for correction in all_corrections:
        two_edits = self.edits(correction)
        for second_edit_correction in two_edits:
          if second_edit_correction in self.movie_titles:
            return second_edit_correction, True
      return None, misspelled

    def remove_accents(self, input_str):
      input_str = input_str.replace('é', 'e').replace('ô', 'o').replace('¡','!').replace('Á', 'A').replace('À','A').replace('Â','A').replace('Ä','A')
      input_str = input_str.replace('ü', 'u').replace('½', '1/2').replace('³', '3').replace('É', 'E').replace('ó', 'o')
      input_str = input_str.replace('è', 'e').replace('ö', 'o').replace('í','i').replace('ê', 'e').replace('î', 'i').replace('ì','i')
      input_str = input_str.replace('û', 'u').replace('ù', 'u').replace('à','a').replace('á','a').replace('ñ','n').replace('â','a')
      input_str = input_str.replace('ä', 'a').replace('ý','y').replace('å','a').replace('Ê', 'E').replace('æ','ae').replace('Ô','O')
      input_str=input_str.replace('ò','o').replace('Å','A').replace('ï','i').replace('ß', 'B').replace('ç','c').replace('ë','e')
      input_str=input_str.replace('ø','o').replace('ã','a').replace('İ','i').replace('ú','u').replace('ı','1')
      return input_str

    def format_movie(self, movie):
      movie = self.remove_accents(movie)
      match = re.match('([\w :;,.\'&/!\[\]\(\)\{\}?*’#·°\"$+@\-]+)(, The)?[\w\(\), :;·,\[\]\(\)\{\}.\'&/!?*’#°\"$+@\-]*(\([0-9]{4}-?\))', movie)
      if not match: 
        match = re.match('\([1-3][0-9]{3} ?- ?[1-3][0-9]{3}\)', movie)
        if not match: return movie
      if match.group(2): return "The " + match.group(1)
      return match.group(1)

    def unformat_movie(self, movie):
      split_movie = movie.split()
      if split_movie[0] == "The":
        res = ''
        for i in range(1, len(split_movie)):
          res += split_movie[i] + " "
        res = res.strip()
        return res + ", The"
      else:
        return movie

    def format_series_string(self, series_array):
      res=''
      length = len(series_array)-1
      for i in range(length):
        res += "\"" + series_array[i]+ "\", "
      return res + 'or \"'+series_array[length]+"\""

    def extract_series_from_list(self, input):
      for elem in self.series_clarification.split("\""):
        if input in elem: return elem

    def all_the_same_movie(self, arr):
      movie = arr[0]
      for i in arr:
        if i != movie: return False
      return True

    def extract_years_of_same_movie(self, title):
      years=[]
      for t, g in self.titles:
        if self.unformat_movie(title) in t:
          m = re.search('\(([1-3][0-9]{3})\)', t)
          if m: years.append(m.group(1))
      return years

    def format_year_string(self, years):
      ints = []
      for s in years:
        ints.append(int(s))
      ints = sorted(ints)
      res = ''
      length = len(ints)-1
      for i in range(length):
        res+= str(ints[i])+ ", "
      return res +'or '+str(ints[length])

    def contains_year(self,tit):
      m = re.search('\([1-3][0-9]{3}\)', tit)
      if not m: return False
      return True

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
      response = ""
      if self.spelling_clairifcation != '' and self.is_turbo:
        if input.lower() in self.affirmations:
          movie = [self.spelling_clairifcation]
          input = self.original_input
          self.spelling_clairifcation=''
        elif input.lower() in self.negations:
          self.spelling_clairifcation = ''
          self.original_input=''
          return "Shoot okay, let\'s try again. Tell me what you were saying. Maybe check your spelling too?"
        else:
          return "I did not get that. Were you talking about \"{}?".format(self.spelling_clairifcation)
      elif self.series_clarification != '' and self.is_turbo:
        my_input = input
        if "\"" in my_input: my_input = my_input.replace("\"", '') 
        if my_input in self.series_clarification:
          movie = [self.extract_series_from_list(my_input)]
          input = self.original_input
          self.series_clarification=''
        else:
          return "I did not get that. Which one were you talking about?  I know of {}".format(self.series_clarification)
      elif self.year_clarification != '' and self.is_turbo:
        if input in self.year_clarification:
          orig_movie = self.extract_movie(self.original_input)[0]
          movie = [orig_movie+" ("+input+")"]
          input = self.original_input
          self.year_clarification=''
        else:
          return "I only know of the movies made in the years {}. Which one of these did you mean?".format(self.year_clarification)
      else:
        movie = self.extract_movie(input)
        self.original_input=input

      if len(movie) == 0: # Arbitrary Input Cases
        if "\"" in input: return self.unknown_movie[random.randrange(0,len(self.unknown_movie))].format(self.extract_movie(input))
        if "What is" in input:
          pronoun = input.split("What is")[1]
          if "?" in pronoun: pronoun = pronoun[:len(pronoun)-1]
          return self.what_array[random.randrange(0, len(self.what_array))].format(pronoun)
        if "Can you" in input:
          ability = input.split("Can you")[1]
          if "?" in ability: ability = ability[:len(ability)-1]
          return self.can_array[random.randrange(0, len(self.can_array))].format(ability)
        if "Where is" in input:
          location = input.split("Where is")[1]
          if "?" in location: location = location[:len(location)-1]
          return self.where_array[random.randrange(0, len(self.where_array))].format(location)
        if self.detect_emotion(input) != "":
          return self.detect_emotion(input)
        return self.catch_all[random.randrange(0, len(self.catch_all))]
      if len(movie) > 1 and not self.is_turbo: return'Please tell me about one movie at a time. Go ahead.'
      
      for i, m in enumerate(movie):
        movie[i] = self.format_movie(m)
        if m not in self.movie_titles and not self.contains_year(m):
          if not self.is_turbo: return self.unknown_movie[random.randrange(0,len(self.unknown_movie))].format(m)
          closest_movie, misspelled = self.find_closest_movie(m)
          if misspelled:
            if closest_movie: 
              self.spelling_clairifcation = closest_movie
              return "Did you mean \"{}\"?".format(closest_movie)
            else: return "Sorry I am not sure what to make of the movie \"{}\". Maybe check your spelling?".format(m)
          else: # Case where it is in series.
            if closest_movie and len(closest_movie) > 1:
              if self.all_the_same_movie(closest_movie): # Case where same movie title, different years.
                years_array = self.extract_years_of_same_movie(closest_movie[0])
                self.year_clarification = self.format_year_string(years_array)
                return "Which \"{}\" movie did you mean? The one made in {}?".format(m, self.year_clarification)
              else:
                closest_movie_string = self.format_series_string(closest_movie)
                self.series_clarification = closest_movie_string
                return "Which \"{}\" movie did you mean? I can talk about {}.".format(m, closest_movie_string)
            elif closest_movie and len(closest_movie) == 1:
              movie[i] = closest_movie[0]
            else: return "Sorry I am not sure what to make of the movie \"{}\"".format(m)
        else:
          if self.movie_titles.count(m) > 1:# Case where same movie title, different years.
            years_array = self.extract_years_of_same_movie(m)
            self.year_clarification = self.format_year_string(years_array)
            return "Which \"{}\" movie did you mean? The one made in {}?".format(m, self.year_clarification)
          elif self.contains_year(m): 
            movie[i] = m

          # movie is in list of movies, but need to check if count > 1
          # give options for years.
      # for starter, only one movie.
      if not self.is_turbo or (self.is_turbo and len(movie)==1):
        movie = movie[0]
        rating = self.extract_sentiment(input)
        if rating == 3: return self.get_sentiment_response(movie, rating)
        response = self.get_sentiment_response(movie, rating) + " Thank you."
        self.add_to_vector(movie, rating)
      else:
        movie_sentiments = self.multiple_move_sentiment(movie, input)
        for m in movie_sentiments:
          response += self.get_sentiment_response(m[0], m[1])
          self.add_to_vector(m[0], m[1])
        response += " Thank you."


      # self.user_vector = [("Harry Potter and the Chamber of Secrets", 5),
      #               ("Harry Potter and the Prisoner of Azkaban", 5),
      #               ("Harry Potter and the Goblet of Fire", 5),
      #               ("Harry Potter and the Order of the Phoenix", 5),
      #               ("Harry Potter and the Deathly Hallows: Part 1", 5),
      #               ("Friends with Benefits", 1)]

      # self.user_vector = [("Bridesmaids", 5),
      #                     ("No Strings Attached", 5),
      #                     ("Friends with Benefits", 5),
      #                     ("Mean Girls", 5),
      #                     ("How to Lose a Guy in 10 Days", 5),
      #                     ("Born Yesterday", 5)
      #                     ]

      if len(self.user_vector) >= self.NUM_MOVIES_THRESHOLD: 
        response +=  " That\'s enough for me to make a recommendation."
        recommendation = self.recommend()
        response += " I suggest you watch \"{}\".".format(recommendation)
        # Do you want another recommendation? 
      else:  response += " Tell me about another movie you have seen."
      return response

    # def add_movie(self, user_input):
    #   movie = extract_movie(user_input)
    #   sentiment = extract_sentiment(user_input)
    #   add_to_vector(movie, sentiment)


    def detect_emotion(self, user_input):
      angry_words = ["angri", "mad", "piss", "hate", "livid", "frustrat", "unhappi"]
      happy_words = ["happi", "excit", "glad", "thrill"]
      user_input = user_input.replace(".", "").replace("!", "")
      word_array = user_input.split(" ")

      #detect anger
      for ind, w in enumerate(word_array):
        word = self.porter_stemmer.stem(w)
        prev_word = user_input.split(" ")[ind - 1]
        if word in angry_words:
          if prev_word == "not" or prev_word == "never" or prev_word.endswith("n't"):
            return "I'm glad you're not angry."
          else:
            return "I see you're angry...I hope not with me!"
        if word in happy_words:
          if prev_word == "not" or prev_word == "never" or prev_word.endswith("n't"):
            return "I'm sorry you're not happy. Tell me about some movies you watch when you're sad"
          else:
            return "You sound like you're in a good mood!"
      return ""



    #assumes movies are spelled correctly
    def get_sentiment_response(self, movie, rating):
      sentiment_responses = [0,
                            "Wow, you hated \"{}\". ".format(movie),
                            "You did not like \"{}\". ".format(movie),
                            "I\'m sorry, I\'m not quite sure if you liked {}. Tell me more about \"{}\"".format(movie, movie),
                            "You liked \"{}\". ".format(movie),
                            "Wow, you loved \"{}\". ".format(movie)
                            ]
      return sentiment_responses[rating]


    def multiple_move_sentiment(self, movies_vect, user_input):
      user_input = re.sub("\"[^\".]*\"", "<MOVIE>", user_input)
      user_input = user_input.replace(" and ",'*').replace(" but ", "*").replace(" or ", "*")
      user_input = user_input.split('*')
      
      scores = [0]*len(user_input)
      for ind, clause in enumerate(user_input):
        scores[ind] = self.calculate_sentiment(clause, 0)

      responses = [(0,0)]*len(movies_vect)
      for ind, score in enumerate(scores):
        if score == 3:
          if ind != 0:
            scores[ind] = scores[ind-1]
        responses[ind] = (movies_vect[ind], scores[ind])

      return responses

    def extract_movie(self, user_input):
      return re.findall('"([^"]*)"', user_input)

    def extract_sentiment(self, user_input):
      #take out movie name
      if self.spelling_clairifcation == '':
        movie_name = "\""+self.extract_movie(user_input)[0]+"\""
        user_input = user_input.replace(movie_name, "")
      else:
        movie_name = self.spelling_clairifcation
        user_input = self.original_input

      intensity = self.gauge_intensity(user_input)
      score = self.calculate_sentiment(user_input, intensity)
      return score


    def calculate_sentiment(self, user_input, intensity):
      score = 0
      user_input = user_input.replace(".", "").replace("!", "")
      word_array = user_input.split(" ")
      if "really" in word_array:
        word_array.remove("really")
      for ind, w in enumerate(word_array):
        word = self.porter_stemmer.stem(w)
        if word in self.sentiment.keys():
          prev_word = user_input.split(" ")[ind - 1]
          sentiment = self.sentiment[word]
          if sentiment == "pos":
            val = 1
          elif sentiment == "neg":
            val = -1
          if prev_word == "not" or prev_word == "never" or prev_word.endswith("n't"):
            val = val*-1
          score += val

      if score > 0:
        return (4 + intensity)
      elif score < 0:
        return (2 - intensity)
      else:
        return 3

    def remove_year(self, title):
      m = re.search('(\([1-3][0-9]{3}\))', title)
      yr = m.group(1)
      return title.replace(yr, '').strip(), yr.replace("(", "").replace(")","")

    def gauge_intensity(self, user_input):
      if "!" in user_input:
        return 1
      words = user_input.split(" ")
      intensifiers = ["love", "hate", "ador", "favorit", "worst", "realli", "veri"]
      for w in words:
        word = self.porter_stemmer.stem(w)
        if word in intensifiers:
          return 1

      return 0


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
      normalize_u = np.linalg.norm(u)
      normalize_v = np.linalg.norm(v)
      if normalize_u == 0 or normalize_v == 0: return 0
      num = np.dot(u,v)
      sim = num/(normalize_v*normalize_u)

      return sim


    def find_rating(self, movie_index, other_movies):
      similarity_vector = np.zeros(len(other_movies))
      movie = self.mean_centered_matrix[movie_index]
      

      #other movies is vect of indices
      for i, other_movie in enumerate(other_movies):
        other_movie_vect = self.mean_centered_matrix[other_movie]
        sim = self.distance(movie, other_movie_vect)
        if sim > 0:
          similarity_vector[i] = sim
      numerator = 0
      denomimnator = np.sum(similarity_vector) # -1 for its similarity with itself.
      if denomimnator == 0:
        return 0
      numerator = np.dot(similarity_vector, self.formatted_vec[list(other_movies.astype(np.int32))])
      
      predicted_rating = numerator/denomimnator
      return predicted_rating


    def recommend(self):
      """Generates a list of movies based on the input vector u using
      collaborative filtering"""
      # TODO: Implement a recommendation function that takes a user vector u
      # and outputs a list of movies recommended by the chatbot
      self.formatted_vec = self.format_vec()
      self.mean_centered_matrix = self.generate_matrix(self.formatted_vec)
      predictions = []
      other_movies = np.nonzero(self.formatted_vec)[0]
      for index, val in enumerate(self.formatted_vec):
        if val > 0:
          predictions.append((-1, index))
        else:
          predictions.append((self.find_rating(index, other_movies), index))
      predictions = sorted(predictions, reverse = True)
      print self.movie_titles[predictions[0][1]]
      print self.movie_titles[predictions[1][1]]
      print self.movie_titles[predictions[2][1]]
      print self.movie_titles[predictions[3][1]]
      print self.movie_titles[predictions[4][1]]
      print self.movie_titles[predictions[5][1]]
      print self.movie_titles[predictions[6][1]]
      print self.movie_titles[predictions[7][1]]
      return self.movie_titles[predictions[0][1]]
      ## Will need to write more functions for computing sims then returning best recommendation. ##

    def format_vec(self):
      '''
      Creates user vector based on their ratings for each movie in our movie matrix.
      This is normally going to be a VERY sparse vector. Used for computing cosine 
      similarity.
      '''
      num_movies = len(self.movie_titles)
      fv = [0] * num_movies
      for movie, rating in self.user_vector:
        if self.contains_year(movie):
          title_without_year, year = self.remove_year(movie)
          title_without_year = self.unformat_movie(title_without_year)
          movie = title_without_year + " ("+year+")"
          for i in range(len(self.titles)):
            t,g = self.titles[i]
            if t == movie:
              curr_pos = i
              break
        else:
          curr_pos = self.movie_titles.index(movie)
        fv[curr_pos] = rating
      return np.array(fv)

    def generate_matrix(self, user_vector):
      matrix = np.c_[self.ratings, user_vector]
      return matrix - np.mean(self.ratings, axis=1, keepdims=True)


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

      My group just spent a few hours formatting our code so that it responds to movies the way the rubric indicated
      (without the year i.e. "Titanic).
      Since this clarification is coming in late, is there any way we can be graded by the original rubric?
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
