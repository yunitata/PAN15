#!/usr/bin/python
import sys
import getopt
import feature_extractor
import classifier
import comparison
import os
import io


def main(argv):
    inputfile = ''
    outputfile = ''
    modelfile = ''

    if len(sys.argv) == 5:
        try:
            opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
        except getopt.GetoptError:
            print 'error'
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-h':
                print 'main.py -i /path/to/training/corpus -o path/to/output/directory'
                sys.exit()
            elif opt in ("-i", "--ifile"):
                inputfile = arg
            elif opt in ("-o", "--ofile"):
                outputfile = arg

        print 'Input file is', inputfile
        print 'Output file is', outputfile

        feature_extractor.write_to_file(inputfile, outputfile)

        print("creating lexicon for training")
        lexicon_8char = feature_extractor.tfidf_lexicon_8char(inputfile)
        lexicon_3char = feature_extractor.tfidf_lexicon_3char(inputfile)
        lexicon_bigram = feature_extractor.tfidf_lexicon_2word(inputfile)
        lexicon_unigram = feature_extractor.tfidf_lexicon_unigram_word(inputfile)
        print("creating feature vector for training data")
        sim_score = comparison.sim_score(inputfile, lexicon_8char, lexicon_3char, lexicon_bigram, lexicon_unigram)
        print(sim_score)
        truth = feature_extractor.truth(inputfile)
        #print(truth)
        print("training the classifier")
        model_path = classifier.classify(sim_score, truth, outputfile)
        print("finish training")
        print("classifier model is saved in", model_path)

    elif len(sys.argv) == 7:
        try:
            opts, args = getopt.getopt(argv, "hi:m:o:", ["ifile=", "mfile=", "ofile="])
        except getopt.GetoptError:
            print 'error'
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-h':
                print 'main.py -i /path/to/test/corpus -m path/to/classifier/model -o path/to/output/directory'
                sys.exit()
            elif opt in ("-i", "--ifile"):
                inputfile = arg
            elif opt in ("-m", "--mfile"):
                modelfile = arg
            elif opt in ("-o", "--ofile"):
                outputfile = arg

        print 'Input file is', inputfile
        print 'Model file is', modelfile
        print 'Output file is', outputfile
        file_path = ''
        for subdir, dirs, files in os.walk(modelfile):
            print("this is test")
            print(subdir)
            print (files)
            for file_ in files:
                print(file_)
                if file_ == 'train.txt':
                    file_path = subdir + os.path.sep + file_
        #print(file_path)
        opened_file = io.open(file_path, 'r')
        path_train = opened_file.read()
        print(path_train)
        lexicon_8char = feature_extractor.tfidf_lexicon_8char(path_train)
        lexicon_3char = feature_extractor.tfidf_lexicon_3char(path_train)
        lexicon_bigram = feature_extractor.tfidf_lexicon_2word(path_train)
        lexicon_unigram = feature_extractor.tfidf_lexicon_unigram_word(path_train)
        print("creating feature vector for test data")
        sim_score = comparison.sim_score(inputfile, lexicon_8char, lexicon_3char, lexicon_bigram, lexicon_unigram)
        print(sim_score)
        print("classify test data")
        output_path = classifier.classifier_predict(modelfile, sim_score, outputfile)
        print("finish")
        print "answer is saved in: ", output_path

if __name__ == "__main__":
   main(sys.argv[1:])


