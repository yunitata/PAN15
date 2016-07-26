from __future__ import division
import feature_extractor
import os
import glob
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def minmax_sim(X, Y):
    min_vector = np.minimum(X, Y)
    max_vector = np.maximum(X, Y)
    sim = np.sum(min_vector)/np.sum(max_vector)
    return sim


def vec_diff(X, Y):
    dif = np.subtract(X, Y)
    return np.abs(np.mean(dif))


def vec_diff2(X, Y):
    dif = np.subtract(X, Y)
    return dif

def mean(a):
    return sum(a) / len(a)


def sim_score(path_problem, lexicon_8gram, lexicon_3gram, lexicon_bigram, lexicon_unigram):
    sim_score = {}
    for path, subdirs, files in os.walk(path_problem):
        for name_dir in subdirs:
            print(name_dir)
            vec_feature = []
            sim_score_fw = []
            sim_score_stylo = []
            sim_score_8gram = []
            sim_score_3gram = []
            sim_score_bigram = []
            sim_score_unigram = []
            lang = name_dir[:2]
            if lang == 'EN':
                fw_file = 'english.txt'
            elif lang == 'DU':
                fw_file = 'dutch.txt'
            elif lang == 'GR':
                fw_file = 'greek.txt'
            elif lang == 'SP':
                fw_file = 'spanish.txt'

            dir_path = os.path.join(path_problem, name_dir)
            for name1 in glob.glob(dir_path + "/unknown.txt"):
                file_path_unknown = os.path.join(dir_path, name1)
                fw_unknown = feature_extractor.freq_function_word(file_path_unknown, fw_file)
                stylo_unknown = feature_extractor.stylometric_features(file_path_unknown, lang)
                eight_gr_unknown = feature_extractor.tfidf(file_path_unknown, 8, 'char', lexicon_8gram)
                three_gr_unknown = feature_extractor.tfidf(file_path_unknown, 3, 'char', lexicon_3gram)
                bigram_unknown = feature_extractor.tfidf(file_path_unknown, 2, 'word', lexicon_bigram)
                unigram_unknown = feature_extractor.tfidf(file_path_unknown, 1, 'word', lexicon_unigram)

                for name2 in glob.glob(dir_path + "/known??.txt"):
                    file_path_known = os.path.join(dir_path, name2)

                    fw_known = feature_extractor.freq_function_word(file_path_known, fw_file)
                    stylo_known = feature_extractor.stylometric_features(file_path_known, lang)
                    eight_gr_known = feature_extractor.tfidf(file_path_known, 8, 'char', lexicon_8gram)
                    three_gr_known = feature_extractor.tfidf(file_path_known, 3, 'char', lexicon_3gram)
                    bigram_known = feature_extractor.tfidf(file_path_known, 2, 'word', lexicon_bigram)
                    unigram_known = feature_extractor.tfidf(file_path_known, 1, 'word', lexicon_unigram)

                    sim_score_fw.append(minmax_sim(fw_unknown, fw_known))
                    sim_score_stylo.append(vec_diff(stylo_unknown, stylo_known))
                    sim_score_8gram.append(cosine_similarity(eight_gr_unknown, eight_gr_known))
                    sim_score_3gram.append(cosine_similarity(three_gr_unknown, three_gr_known))
                    sim_score_bigram.append(cosine_similarity(bigram_unknown, bigram_known))
                    sim_score_unigram.append(cosine_similarity(unigram_unknown, unigram_known))

                vec_feature.append(np.mean(sim_score_stylo))
                vec_feature.append(np.mean(sim_score_fw))
                vec_feature.append(np.mean(sim_score_8gram))
                vec_feature.append(np.mean(sim_score_3gram))
                vec_feature.append(np.mean(sim_score_bigram))
                vec_feature.append(np.mean(sim_score_unigram))

            sim_score[name_dir] = vec_feature
    sort = OrderedDict(sorted(sim_score.items(), key=lambda s: s[0]))
    return sort

