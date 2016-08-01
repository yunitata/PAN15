from __future__ import division
import nltk
import os
import io
import itertools
import re
import string
import enchant
from nltk.tag import map_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import OrderedDict


'''-----------------Clean up the text from punctuation------------------------------'''


def clean_up(s):
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in s if ch not in exclude)
    return s


def remove_punctuation(text):  # remove punctuation
    return re.sub(ur"\p{P}+", "", text)


'''-----------------Open file and make it as raw text or array of text---------------'''


def open_file(path):
    # open the file and convert the raw text into array/list
    text_ = []
    opened_file = io.open(path, 'r', encoding='utf-8')
    text = opened_file.read()
    lowers = text.lower()
    no_punctuation = lowers.rstrip()
    text_.append(no_punctuation.rstrip('\n'))
    return text_


def open_text(path):
    opened_file = io.open(path, 'r', encoding='utf-8')
    text = opened_file.read()
    lowers = text.lower()
    #no_punctuation = remove_punctuation(lowers)
    return lowers


'''-------------------------------text tokenization-------------------------'''


def tokenize(text):  # tokenize the text
    tokens = nltk.word_tokenize(text)
    return tokens
'''-----------------Stylometric features-------------------------------'''


def convert_nan(type):
    if type == 'NoneType':
        return int(0)


def syllables(word):
    count = 0
    vowels = 'aeiouy'
    word = word.lower().strip(".:;?!")
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index-1] not in vowels:
            count += 1
    if word.endswith('e'):
        count -= 1
    if word.endswith('le'):
        count += 1
    if count == 0:
        count += 1
    return count


def total_syllables(text):
    total = 0
    clean_text = clean_up(text)
    word_list = nltk.word_tokenize(clean_text)
    for i in range(len(word_list)):
        num_syl_per_word = syllables(word_list[i])
        total = total + num_syl_per_word
    return total


def count_three_more_syl(text):
    total = 0
    clean_text = clean_up(text)
    word_list = nltk.word_tokenize(clean_text)
    for i in range(len(word_list)):
        num_syl_per_word = syllables(word_list[i])
        if num_syl_per_word >= 3:
            total += 1
    return total


def percentage_word_with_three_syl(text):
    three_syl = count_three_more_syl(text)
    total_syl = total_syllables(text)
    return three_syl/total_syl


def word_six_letters(text):
    total = 0
    word_list = nltk.word_tokenize(text)
    for i in range(len(word_list)):
        total_letter = len([ltr for ltr in word_list[i] if ltr.isalpha()])
        if total_letter >= 6:
            total += 1
    return total


def average_number_of_words_per_sentence(text):
    word_count = 0
    sentence_list = nltk.sent_tokenize(text)
    for sentence in sentence_list:
        wordlist = nltk.word_tokenize(sentence)
        word_count += len(wordlist)
    return (float(word_count)/len(sentence_list))/10


def percentage_short_sent(text):
    short_sentence = 0
    sentence_all = len(nltk.sent_tokenize(text))
    sentence_list = nltk.sent_tokenize(text)
    for sentence in sentence_list:
        wordlist = nltk.word_tokenize(sentence)
        word_count = len(wordlist)
        if word_count < 8:
            short_sentence += 1
    return short_sentence/sentence_all


def percentage_long_sent(text):
    long_sentence = 0
    sentence_all = len(nltk.sent_tokenize(text))
    sentence_list = nltk.sent_tokenize(text)
    for sentence in sentence_list:
        wordlist = nltk.word_tokenize(sentence)
        word_count = len(wordlist)
        if word_count >15:
            long_sentence += 1
    return long_sentence/sentence_all


def lexical_diversity(text):
    tokenize_text = nltk.word_tokenize(text)
    result = len(tokenize_text)/len(set(tokenize_text))
    return result


def count_sentences(text):
    return len(nltk.sent_tokenize(text))/10


def count_punctuation(text):
    count = lambda l1, l2: sum([1 for x in l1 if x in l2])
    return count(text, set(string.punctuation)) / len(nltk.sent_tokenize(text))


def hapax_legomena_ratio(text):
    word_list = nltk.word_tokenize(text)
    fdist = nltk.FreqDist(word for word in word_list)
    fdist_hapax = nltk.FreqDist.hapaxes(fdist)
    return float(len(fdist_hapax)/len(word_list))


def average_word_length(text):
    word_list = nltk.word_tokenize(text)
    return float(sum(map(len, word_list))) / len(word_list)


def average_non_standard_word(lang, text):
    diction = enchant.Dict('EN-GB')
    if lang == 'EN':
        diction = enchant.Dict('EN-GB')
    elif lang == 'DU':
        diction = enchant.Dict('nl-NL')
    elif lang == 'GR':
        diction = enchant.Dict('el-GR')
    elif lang == 'SP':
        diction = enchant.Dict('es-ES')
    clean_text = clean_up(text)
    word_list = nltk.word_tokenize(clean_text)
    count = 0
    for i in range(len(word_list)):
        if diction.check(word_list[i]) is False:
            count += 1
    return count/len(word_list)


def fleschease(text):
    total_words = len(nltk.word_tokenize(text))
    total_sent = len(nltk.sent_tokenize(text))
    total_syl = total_syllables(text)
    reading_ease = 206.835-(1.015*(total_words/total_sent))-(84.6*(total_syl/total_words))
    return reading_ease


def kincaid(text):
    total_words = len(nltk.word_tokenize(text))
    total_sent = len(nltk.sent_tokenize(text))
    total_syl = total_syllables(text)
    grade = (0.39*(total_words/total_sent))+(11.8*(total_syl/total_words))-15.59
    return grade


def fogindex(text):
    total_words = len(nltk.word_tokenize(text))
    total_sent = len(nltk.sent_tokenize(text))
    three_more_syl = count_three_more_syl(text)
    index = ((total_words/total_sent)+(100*(three_more_syl/total_words))) * 0.4
    return index


def lix(text):
    total_words = len(nltk.word_tokenize(text))
    total_sent = len(nltk.sent_tokenize(text))
    total_word_six_letter = word_six_letters(text)
    lix_score = (total_words/total_sent)+(100*(total_word_six_letter/total_words))
    return lix_score


def count_ADV(text):
    word_list = nltk.word_tokenize(text)
    tag_word = nltk.pos_tag(word_list)
    tag_fd = nltk.FreqDist(map_tag('en-ptb', 'universal', tag) for (word,tag)in tag_word)
    adv = tag_fd.get('ADV')
    if adv is None:
        adv = 0
    return adv/len(word_list)


def count_ADJ(text):
    word_list = nltk.word_tokenize(text)
    tag_word = nltk.pos_tag(word_list)
    tag_fd = nltk.FreqDist(map_tag('en-ptb', 'universal', tag) for (word, tag)in tag_word)
    adj = tag_fd.get('ADJ')
    if adj is None:
        adj =0
    return adj/len(word_list)


def count_DET(text):
    word_list = nltk.word_tokenize(text)
    tag_word = nltk.pos_tag(word_list)
    tag_fd = nltk.FreqDist(map_tag('en-ptb', 'universal', tag) for (word, tag)in tag_word)
    det = tag_fd.get('DET')
    if det is None:
        det = 0
    return det/len(word_list)


def count_CONJ(text):
    word_list = nltk.word_tokenize(text)
    tag_word = nltk.pos_tag(word_list)
    tag_fd = nltk.FreqDist(map_tag('en-ptb', 'universal', tag) for (word, tag)in tag_word)
    conj = tag_fd.get('CONJ')
    if conj is None:
        conj = 0
    return conj/len(word_list)


def count_PRO(text):
    word_list = nltk.word_tokenize(text)
    tag_word = nltk.pos_tag(word_list)
    tag_fd = nltk.FreqDist(map_tag('en-ptb', 'universal', tag) for (word, tag)in tag_word)
    pro = tag_fd.get('PRON')
    if pro is None:
        pro = 0
    return pro/len(word_list)


def count_X(text):
    word_list = nltk.word_tokenize(text)
    tag_word = nltk.pos_tag(word_list)
    tag_fd = nltk.FreqDist(map_tag('en-ptb', 'universal', tag) for (word, tag)in tag_word)
    x = tag_fd.get('X')
    if x is None:
        x = 0
    return x/len(word_list)


def word_rareness(text):
    common_words = "google-10000-english.txt"
    with open(common_words, 'r') as f:
        com_words = {line.strip() for line in f}
    com_words_sorted = list(com_words)
    word_list = nltk.word_tokenize(string.lower(text))
    count = 0
    for word in word_list:
        if word not in com_words_sorted:
            count += 1
    return count/len(word_list)


def stylometric_features(file_path, lang):
    stylometry_vector = []
    raw_text = open_text(file_path)
    if lang == 'EN':
        stylometry_vector.append(average_non_standard_word(lang, raw_text))
    stylometry_vector.append(average_number_of_words_per_sentence(raw_text))
    stylometry_vector.append(lexical_diversity(raw_text))
    stylometry_vector.append(count_punctuation(raw_text))
    stylometry_vector.append(fleschease(raw_text))
    stylometry_vector.append(kincaid(raw_text))
    stylometry_vector.append(fogindex(raw_text))
    stylometry_vector.append(percentage_short_sent(raw_text))
    stylometry_vector.append(percentage_long_sent(raw_text))
    stylometry_vector.append(percentage_word_with_three_syl(raw_text))
    return stylometry_vector


'''------------------------------checking with truth file-------------------'''


def check_truth(name_dir, truth_dict):  # checking the truth between directory name and truth file
    for k, v in truth_dict.iteritems():
        if name_dir == truth_dict[k]:
            return truth_dict[v]


def truth(dir_path_train):  # function for generating truth dictionary (key and value)
    res = {}
    truth_file = ''
    for subdir, dirs, files in os.walk(dir_path_train):
        for file_ in files:
            if file_ == 'truth.txt':
                path_truth = subdir + os.path.sep + file_
                truth_file = io.open(path_truth, 'r', encoding='utf-8')
            for line in truth_file:
                key, value = line.split()
                key = key.replace(u'\ufeff', '')
                key = key.replace(u'\xef\xbb\xbf\xf3', '')
                res[key] = value
    sort = OrderedDict(sorted(res.items(), key=lambda s: s[0]))
    return sort

'''----------------------------generating lexicon for 8-gram char, 4-gram char, 2-gram words, unigram word, and
function word----------------------------------------------------------------------------------------------'''


def tfidf_lexicon_8char(path_train):
    token_dict = {}
    for subdir, dirs, files in os.walk(path_train):
        for file_ in files:
            file_path = subdir + os.path.sep + file_
            opened_file = io.open(file_path, 'r', encoding='utf-8', errors='ignore')
            text = opened_file.read()
            lowers = text.lower()
            no_punctuation = remove_punctuation(lowers)
            token_dict[file] = clean_up(no_punctuation)
    tf_idf_8char = TfidfVectorizer(input=token_dict.values(), tokenizer=tokenize, min_df=1, max_features=50, ngram_range=(8, 8), analyzer='char')
    tf_idf_fit_8char = tf_idf_8char.fit_transform(token_dict.values()).toarray()
    return tf_idf_8char.vocabulary_
#for english max_features = 100


def tfidf_lexicon_3char(path_train):
    token_dict = {}
    for subdir, dirs, files in os.walk(path_train):
        for file_ in files:
            file_path = subdir + os.path.sep + file_
            opened_file = io.open(file_path, 'r', encoding='utf-8', errors='ignore')
            text = opened_file.read()
            lowers = text.lower()
            no_punctuation = remove_punctuation(lowers)
            token_dict[file] = clean_up(no_punctuation)
    tf_idf_4char = TfidfVectorizer(input=token_dict.values(), tokenizer=tokenize, min_df=1, max_features=50, ngram_range=(3, 3), analyzer='char')
    tf_idf_fit_4char = tf_idf_4char.fit_transform(token_dict.values()).toarray()
    return tf_idf_4char.vocabulary_


def tfidf_lexicon_2word(path_train):
    token_dict = {}
    for subdir, dirs, files in os.walk(path_train):
        for file_ in files:
            file_path = subdir + os.path.sep + file_
            opened_file = io.open(file_path, 'r', encoding='utf-8')
            text = opened_file.read()
            lowers = text.lower()
            no_punctuation = remove_punctuation(lowers)
            token_dict[file] = clean_up(no_punctuation)
    tf_idf_2word = TfidfVectorizer(input=token_dict.values(), tokenizer=tokenize, min_df=1, max_features=50, ngram_range=(2, 2), analyzer='word')
    tf_idf_fit_2word = tf_idf_2word.fit_transform(token_dict.values()).toarray()
    return tf_idf_2word.vocabulary_


def tfidf_lexicon_unigram_word(path_train):
    token_dict = {}
    for subdir, dirs, files in os.walk(path_train):
        for file_ in files:
            file_path = subdir + os.path.sep + file_
            opened_file = io.open(file_path, 'r', encoding='utf-8')
            text = opened_file.read()
            lowers = text.lower()
            no_punctuation = remove_punctuation(lowers)
            token_dict[file] = clean_up(no_punctuation)
    tf_idf_unigram = TfidfVectorizer(input=token_dict.values(), tokenizer=tokenize, min_df=1, max_features=50, ngram_range=(1, 1), analyzer='word')
    tf_idf_fit_2word = tf_idf_unigram.fit_transform(token_dict.values()).toarray()
    return tf_idf_unigram.vocabulary_


def freq_function_word(file_path, fw_list):
    freq_norm = []
    text = open_file(file_path)
    text2 = open_text(file_path)
    total_words = len(nltk.word_tokenize(text2))
    fn = os.path.join(os.path.dirname(__file__), fw_list)
    with open(fn) as f:
        function_word = f.readlines()
    for i in range(len(function_word)):
        function_word[i] = function_word[i].strip('\n')
    freq_ = CountVectorizer(input=text, tokenizer=tokenize, min_df=1, ngram_range=(1, 1),
                            analyzer='word', vocabulary= function_word)
    freq_vec = freq_.fit_transform(text).toarray()
    vec = list(itertools.chain(*freq_vec))
    for i in xrange(len(vec)):
        freq_norm.append(vec[i]/total_words)
    return freq_norm

'''----------------------generating tfidf feature vector for each file----------------------------------'''


def tfidf(file_path, ngram_range, analyze, lexicon):
    text = open_file(file_path)
    tfidf_ = TfidfVectorizer(input=text, tokenizer=tokenize, min_df=1,
                             ngram_range=(ngram_range, ngram_range), analyzer=analyze, vocabulary=lexicon)
    tfidf_vec = tfidf_.fit_transform(text).toarray()
    vec = list(itertools.chain(*tfidf_vec))
    return vec


def write_to_file(path_train, model_output_path):
    path = os.path.join(model_output_path, 'train.txt')
    dict_file = open(path, 'w')
    dict_file.write(path_train)
    dict_file.close()
    output = os.path.abspath('train.txt')
    return output


