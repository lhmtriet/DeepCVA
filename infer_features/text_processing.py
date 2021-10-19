import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
from pickle import PicklingError, UnpicklingError, PickleError


class TextProcessor:
	def __init__(self, oov_token=None):
		self.word_dict = {}
		self.char_dict = {}
		self.oov_token = oov_token
		self.min_ngram = 1
		self.max_ngram = 1
		self.char_min_ngram = 3
		self.char_max_ngram = 6
		self.maxlen = 100
		self.word_bound = False
		self.whole_word = False
		self.char_generator = False
		self.word_generator = False

	@staticmethod
	def load_model(file):
		try:
			with open(file, 'rb') as pickle_file:
				tp = pickle.load(pickle_file)
				print("Successfully loaded the file.")
			return tp
		except (UnpicklingError, PickleError, FileNotFoundError, EOFError):
			print("There are some errors in loading the file. Please try again")
			return None

	def save_model(self, file):
		try:
			with open(file, 'wb') as pickle_file:
				pickle.dump(self, pickle_file)
				print("Successfully saved the file.")
			return

		except (PicklingError, PickleError, FileNotFoundError, EOFError):
			print("There are some errors in saving the file. Please try again")
			return

	def transform_text(self, text_documents, type='word', tokenizer=None, maxlen=100, padding='pre'):

		self.maxlen = maxlen

		# 3 types: word, char and both
		if type == 'word':

			if not self.word_generator:
				print("The word generator has not been used.")
				return

			return self.transform_text_word(text_documents, tokenizer=tokenizer, maxlen=maxlen, padding=padding)

		elif type == 'char':

			if not self.char_generator:
				print("The word generator has not been used.")
				return

			return self.transform_text_char(text_documents, maxlen=maxlen, padding=padding)
		else:

			if not self.word_generator:
				print("The word generator has not been used.")
				return

			if not self.char_generator:
				print("The word generator has not been used.")
				return

			return np.c_[self.transform_text_word(text_documents, tokenizer=tokenizer, maxlen=maxlen, padding=padding),
						 self.transform_text_char(text_documents, maxlen=maxlen, padding=padding)]

	def transform_text_word(self, text_documents, tokenizer=None, maxlen=100, padding='pre'):

		min_ngram = self.min_ngram
		max_ngram = self.max_ngram

		all_ngrams = []
		all_ngrams_append = all_ngrams.append

		total_len = []

		for text_document in text_documents:

			if tokenizer is None:
				org_tokens = text_document.split()
			else:
				org_tokens = tokenizer(text_document)

			# print(org_tokens)

			token_len = len(org_tokens)

			cur_min_ngram = min_ngram

			if cur_min_ngram == 1:

				tokens = []

				for cur_word in org_tokens:

					# if len(tokens) == maxlen:
					# 	break

					if cur_word in self.word_dict:
						tokens.append(self.word_dict[cur_word])
					elif self.oov_token is not None:
						tokens.append(self.word_dict[self.oov_token])

				cur_min_ngram += 1
			else:
				tokens = []

			# bind method outside of loop to reduce overhead
			tokens_append = tokens.append
			space_join = " ".join

			for n in range(cur_min_ngram, min(max_ngram + 1, token_len + 1)):

				# if len(tokens) == maxlen:
				# 	break

				for i in range(token_len - n + 1):

					# if len(tokens) == maxlen:
					# 	break

					cur_word = space_join(org_tokens[i: i + n])

					if cur_word in self.word_dict:
						tokens_append(self.word_dict[cur_word])
					elif self.oov_token is not None:
						tokens_append(self.word_dict[self.oov_token])

			# print(len(tokens))

			total_len.append(len(tokens))

			if len(tokens) < maxlen:
				if padding == 'pre':
					tokens = [0] * (maxlen - len(tokens)) + tokens
				elif padding == 'post':
					tokens = tokens + [0] * (maxlen - len(tokens))
			else:
				tokens = tokens[:maxlen]

			# print(tokens)

			all_ngrams_append(tokens)

		return all_ngrams, total_len

	def transform_text_char(self, text_documents, maxlen=1024, padding='pre'):

		min_ngram = self.char_min_ngram
		max_ngram = self.char_max_ngram

		all_ngrams = []
		all_ngrams_append = all_ngrams.append

		total_len = []

		if self.word_bound:

			# Word-bound char-ngram generator
			for text_document in text_documents:

				ngrams = []

				# bind method outside of loop to reduce overhead
				ngrams_append = ngrams.append

				for w in text_document.split():

					# if len(ngrams) == maxlen:
					# 	break

					if self.whole_word:
						cur_word = '<' + w + '>'

						if cur_word in self.char_dict:
							ngrams_append(self.char_dict[cur_word])
						elif self.oov_token is not None:
							ngrams_append(self.char_dict[self.oov_token])

					# w = ' ' + w + ' '
					w_len = len(w)
					for n in range(min_ngram, max_ngram + 1):

						# if len(ngrams) == maxlen:
						# 	break

						offset = 0
						cur_word = w[offset:offset + n]

						if cur_word in self.char_dict:
							ngrams_append(self.char_dict[cur_word])
						elif self.oov_token is not None:
							ngrams_append(self.char_dict[self.oov_token])

						while offset + n < w_len:

							# if len(ngrams) == maxlen:
							# 	break

							offset += 1

							cur_word = w[offset:offset + n]

							if cur_word in self.char_dict:
								ngrams_append(self.char_dict[cur_word])
							elif self.oov_token is not None:
								ngrams_append(self.char_dict[self.oov_token])

						if offset == 0:  # count a short word (w_len < n) only once
							break

				total_len.append(len(ngrams))

				if len(ngrams) < maxlen:
					if padding == 'pre':
						ngrams = [0] * (maxlen - len(ngrams)) + ngrams
					elif padding == 'post':
						ngrams = ngrams + [0] * (maxlen - len(ngrams))
				else:
					ngrams = ngrams[:maxlen]

				all_ngrams_append(ngrams)
		else:

			# Non-word-bound char-ngram generator
			for text_document in text_documents:

				text_len = len(text_document)

				cur_min_ngram = min_ngram

				if min_ngram == 1:
					# no need to do any slicing for unigrams
					# iterate through the string

					for w in list(text_document):
						if w in self.char_dict:
							ngrams_append(self.char_dict[w])
						elif self.oov_token is not None:
							ngrams_append(self.char_dict[self.oov_token])

						# if len(ngrams) == maxlen:
						# 	break

					cur_min_ngram += 1
				else:
					ngrams = []

				# bind method outside of loop to reduce overhead
				ngrams_append = ngrams.append

				if self.whole_word:
					for w in text_document.split():

						# if len(ngrams) == maxlen:
						# 	break

						cur_word = '<' + w + '>'

						if cur_word in self.char_dict:
							ngrams_append(self.char_dict[cur_word])
						elif self.oov_token is not None:
							ngrams_append(self.char_dict[self.oov_token])

				for n in range(cur_min_ngram, min(max_ngram + 1, text_len + 1)):

					# if len(ngrams) == maxlen:
					# 	break

					for i in range(text_len - n + 1):

						# if len(ngrams) == maxlen:
						# 	break

						cur_word = text_document[i: i + n]

						if cur_word in self.char_dict:
							ngrams_append(self.char_dict[cur_word])
						elif self.oov_token is not None:
							ngrams_append(self.char_dict[self.oov_token])

				total_len.append(len(ngrams))

				if len(ngrams) < maxlen:
					if padding == 'pre':
						ngrams = [0] * (maxlen - len(ngrams)) + ngrams
					elif padding == 'post':
						ngrams = ngrams + [0] * (maxlen - len(ngrams))
				else:
					ngrams = ngrams[:maxlen]

				all_ngrams_append(ngrams)

		return np.asarray(all_ngrams), total_len

	def build_vocab_word(self, text_documents, min_ngram=1, max_ngram=1, min_df=0.01, max_df=1.0, tokenizer=None,
						 max_vocab=None, append_dict=False):

		self.min_ngram = min_ngram
		self.max_ngram = max_ngram

		self.word_generator = True

		if not append_dict:
			print("Start building the word vocab !!!")
			self.word_dict = {}
		else:
			print("Continue building the word vocab !!!")

		word_dfreq = defaultdict(int)

		# bind method outside of loop to reduce overhead
		space_join = " ".join

		for text_document in text_documents:

			if tokenizer is None:
				org_tokens = text_document.split()
			else:
				org_tokens = tokenizer(text_document)

			# print(org_tokens)

			cur_dfreq = {}

			token_len = len(org_tokens)

			cur_min_ngram = min_ngram

			if cur_min_ngram == 1:
				for token in org_tokens:
					if token not in cur_dfreq:
						cur_dfreq[token] = 1
						word_dfreq[token] += 1
				cur_min_ngram += 1

			for n in range(cur_min_ngram, min(max_ngram + 1, token_len + 1)):
				for i in range(token_len - n + 1):

					token = space_join(org_tokens[i: i + n])

					if token not in cur_dfreq:
						cur_dfreq[token] = 1
						word_dfreq[token] += 1

		if min_df is None and max_df is None and max_vocab is None:
			print("Please add more documents to continue building the vocab !!!")
			return

		freq_df = pd.DataFrame(word_dfreq.items(), columns=['Word', 'Count'])

		# print(freq_df.head())

		freq_df.sort_values(by=['Count'], ascending=False, inplace=True)

		if max_vocab is not None:
			freq_df = freq_df.iloc[:min(max_vocab, len(freq_df))]
		else:
			if isinstance(min_df, float):
				freq_df['Ratio'] = freq_df.Count / len(freq_df)
				freq_df = freq_df.loc[freq_df.Ratio >= min_df]
			else:
				freq_df = freq_df.loc[freq_df.Count >= min_df]

			if isinstance(max_df, float):
				if "Ratio" not in freq_df.columns:
					freq_df['Ratio'] = freq_df.Count / len(freq_df)
				freq_df = freq_df.loc[freq_df.Ratio <= max_df]
			else:
				freq_df = freq_df.loc[freq_df.Count <= max_df]

		# print(freq_df)

		top_words = freq_df.Word.values

		for index, word in enumerate(top_words):
			self.word_dict[word] = index + 1

		if self.oov_token is not None:
			self.word_dict[self.oov_token] = len(self.word_dict) + 1

		print("Done building the word vocab !!!")

	def build_vocab_char(self, text_documents, min_ngram=3, max_ngram=6, word_bound=False, whole_word=False,
						 min_df=0.01, max_df=1.0, max_vocab=None, append_dict=False):

		self.char_min_ngram = min_ngram
		self.char_max_ngram = max_ngram
		self.word_bound = word_bound
		self.whole_word = whole_word
		self.char_generator = True

		if not append_dict:
			print("Start building the character vocab !!!")
			self.char_dict = {}
		else:
			print("Continue building the character vocab !!!")

		word_dfreq = defaultdict(int)

		if self.word_bound:

			# Word-bound char-ngram generator
			for text_document in text_documents:

				cur_dfreq = {}

				for w in text_document.split():

					if self.whole_word:
						token = '<' + w + '>'

						if token not in cur_dfreq:
							cur_dfreq[token] = 1
							word_dfreq[token] += 1

					# w = ' ' + w + ' '
					w_len = len(w)
					for n in range(min_ngram, max_ngram + 1):
						offset = 0
						token = w[offset:offset + n]

						if token not in cur_dfreq:
							cur_dfreq[token] = 1
							word_dfreq[token] += 1

						while offset + n < w_len:
							offset += 1

							token = w[offset:offset + n]

							if token not in cur_dfreq:
								cur_dfreq[token] = 1
								word_dfreq[token] += 1

						if offset == 0:  # count a short word (w_len < n) only once
							break
		else:

			# Non-word-bound char-ngram generator
			for text_document in text_documents:

				cur_dfreq = {}

				text_len = len(text_document)

				cur_min_ngram = min_ngram

				if cur_min_ngram == 1:
					# no need to do any slicing for unigrams
					# iterate through the string

					for token in list(text_document):
						if token not in cur_dfreq:
							cur_dfreq[token] = 1
							word_dfreq[token] += 1

					cur_min_ngram += 1

				if self.whole_word:
					for w in text_document.split():

						token = '<' + w + '>'

						if token not in cur_dfreq:
							cur_dfreq[token] = 1
							word_dfreq[token] += 1

				for n in range(cur_min_ngram, min(max_ngram + 1, text_len + 1)):
					for i in range(text_len - n + 1):

						token = text_document[i: i + n]

						if token not in cur_dfreq:
							cur_dfreq[token] = 1
							word_dfreq[token] += 1

		if min_df is None and max_df is None and max_vocab is None:
			print("Please add more documents to continue building the vocab !!!")
			return

		freq_df = pd.DataFrame(word_dfreq.items(), columns=['Word', 'Count'])

		freq_df.sort_values(by=['Count'], ascending=False, inplace=True)

		if max_vocab is not None:
			freq_df = freq_df.iloc[:min(max_vocab, len(freq_df))]
		else:
			if isinstance(min_df, float):
				freq_df['Ratio'] = freq_df.Count / len(freq_df)
				freq_df = freq_df.loc[freq_df.Ratio >= min_df]
			else:
				freq_df = freq_df.loc[freq_df.Count >= min_df]

			if isinstance(max_df, float):
				if "Ratio" not in freq_df.columns:
					freq_df['Ratio'] = freq_df.Count / len(freq_df)
				freq_df = freq_df.loc[freq_df.Ratio <= max_df]
			else:
				freq_df = freq_df.loc[freq_df.Count <= max_df]

		# print(freq_df)

		top_words = freq_df.Word.values

		for index, word in enumerate(top_words):
			self.char_dict[word] = index + 1

		if self.oov_token is not None:
			self.char_dict[self.oov_token] = len(self.char_dict) + 1

		print("Done building the character vocab !!!")
