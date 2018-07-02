import numpy as np
from sklearn.model_selection import KFold

import re, nltk, csv
import numpy as np
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import CountVectorizer

import pandas
from sklearn.model_selection import ShuffleSplit

l_tanggapan_positif = []
l_tanggapan_negatif = []
factory1 = StopWordRemoverFactory()
stopword = factory1.create_stop_word_remover()

factory2 = StemmerFactory()
stemmer = factory2.create_stemmer()

def preprocessing(dataset):
    teks = re.sub('(\d)+(\.)*(\d)*', '', dataset)  # hapus digit
    teks = re.sub('[/+@.,%-%^*"!#-$-\']', '', teks)  # hapus simbol

    #stopword removal
    prestopword = stopword.remove(teks)
    #teks = ' '.join([word for word in teks.split() if word not in prestopword])  # clearstopword

    #stemming
    hasilstem = stemmer.stem(prestopword)
    #teks = [stemmer.stem(word) for word in teks.split(" ")]

    tokenProcess = word_tokenize(hasilstem)
    teks = tokenProcess

    return teks

def preprocessing_uselist(listname):
    listteks = []
    for i in range(len(listname)):
        teks = re.sub('(\d)+(\.)*(\d)*', '', listname[i])  # hapus digit
        teks = re.sub('[/+@.,%-%^*"!#-$-\']', '', teks)  # hapus simbol

        #stopword removal
        prestopword = stopword.remove(teks)
        #teks = ' '.join([word for word in teks.split() if word not in prestopword])  # clearstopword

        #stemming
        hasilstem = stemmer.stem(prestopword)
        #teks = [stemmer.stem(word) for word in teks.split(" ")]

        tokenProcess = word_tokenize(hasilstem)
        listteks.extend(tokenProcess)

    return listteks

def bacafile(filename):
    semua = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if row:
                HasilPrepro = preprocessing(row[0])
                semua.append(HasilPrepro)
            else:
                continue
    return semua

def bacafile_uselist(listname):
    semua = []
    for i in range(len(listname)):
        if i:
            HasilPrepro = preprocessing(listname[i])
            semua.append(HasilPrepro)
        else:
            continue
    return semua

def pisah_list_kelas(filename):
    with open(filename) as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if row[1] == '0':
                l_tanggapan_negatif.append(row[0])
            elif row[1] == '1':
                l_tanggapan_positif.append(row[0])

def join_words(listname):
    sentence = ""
    for word in listname:
        sentence += word + " "
    return sentence 

def hitungkatadikelas(sentence):
    l_jumlah = []
    frequency = {}
    match_pattern = re.findall(r'\b[a-z]{3,15}\b', sentence)
 
    for word in match_pattern:
        count = frequency.get(word,0)
        frequency[word] = count + 1 # count(w,c)
        
    frequency_list = frequency.keys()
    
    for words in frequency_list:
        insert = words, frequency[words]+1
        l_jumlah.append(tuple(insert))

    return l_jumlah

def conditional_probabilistist(list_kata,jum_words,detailed_words,jum_vocab): #p(w|c) untuk semua kata pada kelas
    for i in range(len(list_kata)):
        p_w_c = list_kata[i][1] / (jum_words + jum_vocab)
        words = list_kata[i] + (round(p_w_c,5),)
        detailed_words.append(tuple(words))

def bow(listname):
    listmatrix = []
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(listname).todense() #untuk tanggapan ada atau tidaknnya pada bag
    # print(matrix)
    listmatrix.append(matrix)
    vocab = vectorizer.vocabulary_
    # print(vocab)
    return vocab

def training_sentiment(data_train):
    detailed_words_positif = []
    detailed_words_negatif = []
    result_prepro_positif = preprocessing_uselist(l_tanggapan_positif)
    result_prepro_negatif = preprocessing_uselist(l_tanggapan_negatif)
    #untuk bow, semua kalimat di satukan dan menjadi satu satu kata dalam list
    result_prepro_for_bow = preprocessing(data_train)
    dict_vocab = bow(result_prepro_for_bow)
    #Prior P(c)
    print("==============TRAINING (MULTINOMIAL NAIVE BAYES)=============")
    #komponen Prior
    n_kelas_negatif_in_all = len(l_tanggapan_negatif)
    n_kelas_positif_in_all = len(l_tanggapan_positif)
    n_tanggapan = n_kelas_negatif_in_all + n_kelas_positif_in_all
    print("SUM OF OPINION = ", n_tanggapan)
    print("SUM OF NEGATIVE OPINION = ", n_kelas_negatif_in_all)    
    print("SUM OF POSITIVE OPINION = ", n_kelas_positif_in_all)
    prior_positif = n_kelas_negatif_in_all/n_tanggapan #P(positif)
    prior_negatif = n_kelas_positif_in_all/n_tanggapan #P(negatif)
    print("===================[RESULT OF TRAINING MNB]===================")
    print("P(Positif) = ", prior_positif)
    print("P(Negative) = ", prior_negatif)
    #Komponen conditional Probabilistic P(w|c)
    sentence_positif = (join_words(result_prepro_positif))
    sentence_negatif = (join_words(result_prepro_negatif))
    list_jum_kata_positif = hitungkatadikelas(sentence_positif) 
    list_jum_kata_negatif = hitungkatadikelas(sentence_negatif)
    jum_words_positif = len(list_jum_kata_positif)
    jum_words_negatif = len(list_jum_kata_negatif)
    jum_vocab = len(dict_vocab)
    print("Count(Positif) = ", jum_words_positif)
    print("Count(Negatif) = ", jum_words_negatif)
    print("|V| = ", jum_vocab)
    #outputnya = (w, count(w,c)+1 , P(w|c))
    conditional_probabilistist(list_jum_kata_positif,jum_words_positif,detailed_words_positif,jum_vocab) #p(w|positif)
    conditional_probabilistist(list_jum_kata_negatif,jum_words_negatif,detailed_words_negatif,jum_vocab) #p(w|negatif)
    # print("Detailed Positif = ", detailed_words_positif) #list p(w|positif) semua w
    # print("Detailed Negatif = ", detailed_words_negatif) #list p(w|negatif) semua w
    #testing
    data_test = 'data/data_20_test.csv'
    print("==============TESTING (MULTINOMIAL NAIVE BAYES)==============")
    testing_sentiment(data_test,prior_positif,prior_negatif,detailed_words_positif,detailed_words_negatif)

def testing_sentiment(data_test,prior_pos,prior_neg,detailed_words_pos,detailed_words_neg):
    choosingclass = []
    allpwcpos = 1
    allpwcneg = 1
    chooseclasspos = 1
    chooseclassneg = 1
    result_prepro_test = bacafile(data_test)
    for j in range(len(result_prepro_test)):
        if(len(result_prepro_test[j])!=0):
            #look for all P(w|c)
            for k in range(len(result_prepro_test[j])):
                for l in range(len(detailed_words_pos)):
                    if(result_prepro_test[j][k] == detailed_words_pos[l][0]):
                        allpwcpos = allpwcpos * detailed_words_pos[l][2]
                    else:
                        continue
            for m in range(len(result_prepro_test[j])):
                for n in range(len(detailed_words_neg)):
                    if(result_prepro_test[j][m] == detailed_words_neg[n][0]):
                        allpwcneg = allpwcneg * detailed_words_neg[n][2]
                    else:
                        continue
            #0 condition
            if (allpwcpos == 1):
                chooseclasspos = 0
                chooseclassneg = prior_neg * allpwcneg
            elif (allpwcneg == 1):
                chooseclasspos = prior_pos * allpwcpos
                chooseclassneg = 0
            else:
                chooseclasspos = prior_pos * allpwcpos
                chooseclassneg = prior_neg * allpwcneg
            #choosing class
            if (chooseclasspos > chooseclassneg):
                detailed_test = result_prepro_test[j] , round(chooseclasspos,5), 'POSITIVE'
            elif (chooseclasspos < chooseclassneg):
                detailed_test = result_prepro_test[j] , round(chooseclassneg,5),'NEGATIVE'
            choosingclass.append(tuple(detailed_test))
            #reset the component
            allpwcpos = 1
            allpwcneg = 1
            chooseclasspos = 1
            chooseclassneg = 1
        else:
            continue
    print("===================[RESULT OF TESTING MNB]===================")
    ke = 1
    for i in range(len(choosingclass)):
        print("Data - ",ke, " = ",choosingclass[i])
        ke = ke + 1

def main():
    # data_train = 'data/data_100_w_sentiment.csv'
    # read_data_train = open(data_train).read()
    # pisah_list_kelas(data_train) #menjadi list positif dan list negatif yang berisi tanggapan masing masing kelas
    # training_sentiment(read_data_train)
    
    all_data = pandas.read_csv("data/data_5000.csv", )
    print(all_data)

    ss = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
    for train_index, test_index in ss.split(all_data):
        print("%s %s" % (train_index, test_index))

        # print("Isi data train")
        # for i in range(len(train_index)):
        #     print(X[train_index[i]])

        # print("Isi data Testing")
        # for j in range(len(test_index)):
        #     print(X[test_index[j]])
main()