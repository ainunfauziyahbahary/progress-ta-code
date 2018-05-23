import re, glob, nltk, string, csv
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
from collections import Counter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import CountVectorizer

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
        frequency[word] = count + 1
        
    frequency_list = frequency.keys()
    
    for words in frequency_list:
        insert = words, frequency[words]+1
        l_jumlah.append(tuple(insert))

    return l_jumlah

def conditional_probabilistist(list_kata,jum_words,detailed_words,jum_vocab):
    for i in range(len(list_kata)):
        p_w_c = list_kata[i][1] / (jum_words + jum_vocab)
        words = list_kata[i] + (p_w_c,)
        detailed_words.append(tuple(words))

def main():
    listmatrix = []
    detailed_words_positif = []
    detailed_words_negatif = []

    data_train = 'data/data_100_w_sentiment.csv'
    data_test = 'data/data_20_test.csv'
    read_data_train = open(data_train).read()
    pisah_list_kelas(data_train)
    #untuk keperluan selanjutnya
    result_prepro_positif = bacafile_uselist(l_tanggapan_positif)
    result_prepro_negatif = bacafile_uselist(l_tanggapan_negatif)
    result_prepro_positif2 = preprocessing_uselist(l_tanggapan_positif)
    result_prepro_negatif2 = preprocessing_uselist(l_tanggapan_negatif)
    # print("Positif 2 = ", result_prepro_positif2)
    # print("Negatif 2 = ", result_prepro_negatif2)
    result_prepro_for_bow = preprocessing(read_data_train) #untuk bow
    # print("2 = ",result_prepro_for_bow)
    # print("=== Bag Of Word (Vocabulary) ===")
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(result_prepro_for_bow).todense() #untuk tanggapan ada atau tidaknnya pada bag
    # print(matrix)
    listmatrix.append(matrix)
    vocab = vectorizer.vocabulary_
    # print(vocab)

    jum_tanggapan = len(result_prepro_positif) + len(result_prepro_negatif)
    jum_kelas_negatif_in_all = len(l_tanggapan_negatif)
    jum_kelas_positif_in_all = len(l_tanggapan_positif)

    print("TRAINING (MULTINOMIAL NAIVE BAYES)")
    #komponen Prior
    print("Jum tanggapan = ", jum_tanggapan)
    print("Jum negatif = ", jum_kelas_negatif_in_all)    
    print("Jum positif = ", jum_kelas_positif_in_all)    
    
    #Prior
    prior_positif = jum_kelas_positif_in_all/jum_tanggapan
    prior_negatif = jum_kelas_negatif_in_all/jum_tanggapan

    #search conditional Probabilistic komponen
    sentence_positif = (join_words(result_prepro_positif2))
    sentence_negatif = (join_words(result_prepro_negatif2))
    # print("Join Positif = ",sentence_positif)
    # print("Join Negatif = ",sentence_negatif)

    #Komponen Probabilistic -> P(kata , count(w,c)) + 1
    list_jum_kata_positif = hitungkatadikelas(sentence_positif)
    list_jum_kata_negatif = hitungkatadikelas(sentence_negatif)
    # print("Jumlah Kata Positif = ", list_jum_kata_positif)
    # print("Jumlah Kata Negatif = ", list_jum_kata_negatif)
    
    jum_words_positif = len(list_jum_kata_positif)
    jum_words_negatif = len(list_jum_kata_negatif)
    jum_vocab = len(vocab)
    print("count(positif) = ", jum_words_positif)
    print("count(negatif) = ", jum_words_negatif)
    print("|V| = ", jum_vocab)
    
    conditional_probabilistist(list_jum_kata_positif,jum_words_positif,detailed_words_positif,jum_vocab)
    conditional_probabilistist(list_jum_kata_negatif,jum_words_negatif,detailed_words_negatif,jum_vocab)

    #(kata,jumlah kemunculan kata, jumlah kemunculan kata pada kelas)
    print("Prior + = ", prior_positif)
    print("Prior - = ", prior_negatif)
    print("Detailed Positif = ", detailed_words_positif) 
    print("Detailed Negatif = ", detailed_words_negatif)

    print("TESTING (MULTINOMIAL NAIVE BAYES)")
    choosingclass = []
    allpwcpos = 1
    allpwcneg = 1
    chooseclasspos = 1
    chooseclassneg = 1
    result_prepro_test = bacafile(data_test)
    print(len(detailed_words_positif))
    for j in range(len(result_prepro_test)):
        if(len(result_prepro_test[j])!=0):
            for k in range(len(result_prepro_test[j])):
                for l in range(len(detailed_words_positif)):
                    if(result_prepro_test[j][k] == detailed_words_positif[l][0]):
                        allpwcpos = allpwcpos * detailed_words_positif[l][2]
                    else:
                        continue
            for m in range(len(result_prepro_test[j])):
                for n in range(len(detailed_words_negatif)):
                    if(result_prepro_test[j][m] == detailed_words_negatif[n][0]):
                        allpwcneg = allpwcneg * detailed_words_negatif[n][2]
                    else:
                        continue
            print("allpwcpos = ", allpwcpos)
            print("allpwcneg = ", allpwcneg)
            chooseclasspos = prior_positif * allpwcpos
            chooseclassneg = prior_negatif * allpwcneg

            detail_chooseclass_pos = prior_positif, allpwcpos, chooseclasspos
            detail_chooseclass_neg = prior_negatif, allpwcneg, chooseclassneg
            if (chooseclasspos > chooseclassneg):
                detailed_test = result_prepro_test[j] , detail_chooseclass_pos, detail_chooseclass_neg, 'positif'
            elif (chooseclasspos < chooseclassneg):
                detailed_test = result_prepro_test[j] , detail_chooseclass_pos, detail_chooseclass_neg,'negatif'
            choosingclass.append(tuple(detailed_test))
            allpwcpos = 1
            allpwcneg = 1
            chooseclasspos = 1
            chooseclassneg = 1
        else:
            continue
    print("Hasil = ", choosingclass)        
main()