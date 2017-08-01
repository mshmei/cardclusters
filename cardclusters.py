"""
The MIT License (MIT)

Copyright (c) 2017 Michael Songhao Mei

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import nltk, re, time, sys, pandas as pd, numpy as np
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from mtgsdk import Card
# TO-DO:: See if the below is necessary.
# from mtgsdk import Set
# from mtgsdk import Type
# from mtgsdk import Supertype
# from mtgsdk import Subtype
# from mtgsdk import Changelog

# These are used in the toknize and stem process where we only are interested in the words
# which add meaning to our body of rules text.
stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer('english')

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print ('%r (%r, %r) %2.2f sec' % \
              (method.__name__, args, kw, te-ts))
        return result
    return timed

def tokenize(text):
	# Create a set of tokens from the text body. Note that we eliminate all text within 
	# parentheses since this is often times redundant with a keyword mechanic
	tokens = [x.strip('') for x in re.split(r'\s+|[,:;.]\s*', re.sub(r'\([^)]*\)', ' ', text))]
	filtered_tokens = []
	# Filter out all tokens not containing letters
	for token in tokens:
		if re.search('[a-zA-Z]', token):
			filtered_tokens.append(token)
	return filtered_tokens

def tokenize_and_stem(text):
	# Take the root of each filtered token
	stems = [stemmer.stem(t) for t in tokenize(text)]
	return stems

def is_int(string):
	try:
		int(string)
		return 'i'
	except ValueError:
		return 's'
	except TypeError:
		return 'n'

class cardCluster:
	@timeit
	def __init__(self, set):
		if set == 'all':
			self.AllCards = Card.where(language='English').all()
		else:
			self.AllCards = Card.where(language='English', set=set).all()
		self.cards = {card.multiverse_id:{'name':card.name,
										  'multiverse_id':card.multiverse_id,
										  'text':card.text, 
										  'power':card.power, 
										  'toughness':card.toughness,
										  'color_identity':card.color_identity,
										  'type':card.type,
										  'cmc':card.cmc} for card in self.AllCards}

	@timeit
	# This compares the body of rules text if it exists and computes the pairwise cosine similarity
	# yielding a n x n matrix where the ijth component compares card i with card j.
	# A similar process is performed for all other relevant properties.
	def compare_text(self):
		tfidf_vectorizer = TfidfVectorizer(max_df=0.25, min_df=0.01, stop_words='english',
			use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
		tfidf = tfidf_vectorizer.fit_transform([props['text'] 
											    if props['text'] is not None else '' 
											    for k, props in self.cards.items()])
		text_sim = cosine_similarity(tfidf)
		return text_sim

	@timeit
	# We also compare the data type for the power and toughness properties since
	# cards like Tarmogoyf have '*' as their power rather than a numeric value
	# Note that we will have a separate cosine similarity check when the power and toughness
	# are numeric. 
	def compare_power_type(self):
		power_types = np.array([is_int(props['power'])
						   for k, props in self.cards.items()])
		count_vectorizer = CountVectorizer(vocabulary = ['i', 's', 'n'],
										   analyzer = "word",
										   preprocessor = None,
										   tokenizer = None,
										   stop_words = None,
										   token_pattern = "\\b\\w+\\b")
		power_types_vector = count_vectorizer.fit_transform(power_types)
		power_types_sim = cosine_similarity(power_types_vector)
		return power_types_sim

	def compare_toughness_type(self):
		toughness_types = np.array([is_int(props['toughness'])
						   for k, props in self.cards.items()])
		count_vectorizer = CountVectorizer(vocabulary = ['i', 's', 'n'],
										   analyzer = "word",
										   preprocessor = None,
										   tokenizer = None,
										   stop_words = None,
										   token_pattern = "\\b\\w+\\b")
		toughness_types_vector = count_vectorizer.fit_transform(toughness_types)
		toughness_types_sim = cosine_similarity(toughness_types_vector)
		return toughness_types_sim

	@timeit
	# Compares the power and toughness for all cards. We simply map non-numeric types to 0
	# so that the size of the matrix is identical to the one generated by our other properties.
	# We take d(x,y) / (1 + d(x,y)) s.t. our distance is in [0,1].
	def compare_power(self):
		powers = np.array([int(props['power']) 
						   if is_int(props['power']) == 'i' else 0 
						   for k, props in self.cards.items()])
		power_dist = pairwise_distances(powers.reshape(len(powers),1))
		power_sim = 1 - (power_dist / (1 + power_dist))
		return power_sim

	@timeit
	def compare_toughness(self):
		toughnesses = np.array([int(props['toughness']) 
						   if is_int(props['toughness']) == 'i' else 0 
						   for k, props in self.cards.items()])
		toughness_dist = pairwise_distances(toughnesses.reshape(len(toughnesses),1))
		toughness_sim = 1 - (toughness_dist / (1 + toughness_dist))
		return toughness_sim

	@timeit
	# Compares the color identity of the cards.
	def compare_color(self):
		colors = [' '.join(props['color_identity']).lower() 
				  if props['color_identity'] is not None 
				  else 'c' for k, props in self.cards.items()]
		count_vectorizer = CountVectorizer(vocabulary =['b', 'u', 'w', 'g', 'r', 'c'],
										   analyzer = "word",
										   preprocessor = None,
										   tokenizer = None,
										   stop_words = None,
										   token_pattern = "\\b\\w+\\b")

		colors_vector = count_vectorizer.fit_transform(colors)
		colors_sim = cosine_similarity(colors_vector)
		return colors_sim

	@timeit
	# Compares the different types of the cards.
	def compare_type(self):
		tfidf_vectorizer = TfidfVectorizer(max_df=0.80, min_df=0.01, stop_words='english',
			use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
		tfidf = tfidf_vectorizer.fit_transform([props['type'] 
											    if props['type'] is not None else '' 
											    for k, props in self.cards.items()])
		type_sim = cosine_similarity(tfidf)
		return type_sim

	@timeit
	# Compares the converted mana cost.
	def compare_cmc(self):
		cmcs = np.array([int(props['cmc']) 
						 if props['cmc'] is not None
						 else 0 for k, props in self.cards.items()])
		cmc_dist = pairwise_distances(cmcs.reshape(len(cmcs), 1))
		cmc_sim = 1 - (cmc_dist / (1 + cmc_dist))
		return cmc_sim

	@timeit
	# This takes all of the above similarity matrices and performs a weighted summation to
	# yield the final similarity matrix
	def compare_cards(self):
		texts = self.compare_text()
		power_types = self.compare_power_type()
		toughness_types = self.compare_toughness_type()
		powers = self.compare_power()
		toughnesses = self.compare_toughness()
		types = self.compare_type()
		colors = self.compare_color()
		cmcs = self.compare_cmc()
		# We need to map the power similarities for when the power type does not match to 0
		# A similar process must be performed for toughness
		overall_sim = (0.5 * texts
					  + 0.05 * np.multiply(power_types, powers)
					  + 0.05 * np.multiply(toughness_types, toughnesses)
					  + 0.2 * types
					  + 0.1 * cmcs
					  + 0.1 * colors)
		return overall_sim

	def generate_hashes(self):
		overall_sim = self.compare_cards()
		# Sorts the similarity matrix and extracts the top 50 results
		multiverse_ids = np.array([props['multiverse_id'] for k, props in self.cards.items()])
		sorted_sim_keys = multiverse_ids[overall_sim.argsort()[:,-51:-1]]
		sorted_sim_values = np.sort(overall_sim)[:,-51:-1]
		# Creates a nested dictionary where the key is the multiverse_id and the id is a
		# nested dictionary of key value pairs between associated cards and their distances
		sim_dict = {k:v for k, v in zip(multiverse_ids, zip(sorted_sim_keys, sorted_sim_values))}
		return sim_dict


if __name__ == '__main__':
	print ('Accessing API Endpoint now and extracting entire card collection')
	cc = cardCluster(set = sys.argv[1])
	print ('Comparing cards and forming n x n similarity matrix')
	print ('Yielding final dictionary of multiverse ids and their associated multiverse ids')
	gh = cc.generate_hashes()
	print (gh)
