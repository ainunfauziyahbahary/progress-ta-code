import re, glob, nltk, string, csv
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
from collections import Counter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import CountVectorizer

factory1 = StopWordRemoverFactory()
stopword = factory1.create_stop_word_remover()

factory2 = StemmerFactory()
stemmer = factory2.create_stemmer()

def readfile(namafile):
    dataset = open(namafile).read()
    #case folding
    dataset = dataset.lower()
    return dataset

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

def bacafile(filename):
    semua = []
    with open(filename, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            HasilPrepro = preprocessing(row[0])
            semua.append(HasilPrepro)
            #print(HasilPrepro)
        return semua

def main():
    handledata = open('data/data.csv').read()

    preproo= preprocessing(handledata)
    
    listmatrix = []
    print("=== Bag Of Word (Vocabulary) ===")
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(preproo).todense() #untuk tanggapan ada atau tidaknnya pada bag
    # print(matrix)
    listmatrix.append(matrix)
    vocab = vectorizer.vocabulary_
    # print(vocab)

    daftarDokumen = {}
    daftarTanggapan = {}
    # tampung data dari direktori ke variable daftarDokumen
    for i in handledata:  # range() untuk menghasilkan list
        daftarDokumen[i] = Counter(vocab)

    panjangTang = bacafile('data/class/positif.csv')
    print("panjang tanggapan + = " , len(panjangTang))
    print("String tanggapan = ", panjangTang) #ini kata kata positif
    # print("daftar dokumen ", daftarDokumen)

    # menampung daftar string
    daftarString = []
    for key, val in daftarDokumen.items():  # melompati(loop over) key di kamus
        for word, count_w in val.items():
            if word not in daftarString:
                daftarString.append(word)  # append untuk menambah objek word baru kedalam list
    print("daftar string Vocab : ", daftarString) #ini kata kata vocab

    # Membuat TF
    termfrekuensi = []
    with open('data/hasil/hasillikelihood.csv', 'w') as csvfile:
        reswriter = csv.writer(csvfile, delimiter=',', quotechar='|')
        tempWord = [' ']
        tempWord.extend(daftarString)  # menambah string2 dari variable daftarString ke tempWord
        reswriter.writerow(tempWord)  # write 1 row

        tampungSmua=[]
        for i in range(len(panjangTang)):
            # rowjudul = "Tanggapan "+ str(i)
            #hitung pertanggapan
            words = panjangTang[i]
            x = []
            tampungCurrent =[]
            count = 0
            totalsemua = 0
            for j in range(len(daftarString)):
                temp = daftarString[j]
                # print(temp)
                for k in range(len(words)):
                    if (temp == words[k]):
                        count += 1
                smooth = count + 1
                x.append(smooth)
                tampungCurrent.append(smooth)
                count = 0
            # print(totalsemua)
            tampungSmua.append(tampungCurrent)
            #termfrekuensi.append(list(counterlist))
            # x.insert(0, rowjudul)
            # reswriter.writerow(x)
        print(tampungSmua)

        hasilSmoothAll =[]
        totalsmooth = 0
        for i in range(len(tampungSmua)):
            first = []
            second = []
            dalem = tampungSmua[i]
            if i == 0:
                for j in range(len(dalem)):
                    hasilSmoothAll.append(dalem[j])
            else:
                for j in range(len(dalem)):
                    first.append(hasilSmoothAll[j])
                    second.append(dalem[j])
                hasilSmoothAll = []
            for k in range(len(first)):
                hasilSmoothAll.append(first[k]+second[k])
        totalsmooth = sum(hasilSmoothAll)         
        # print("ini baru hasilSmoothAll :",hasilSmoothAll)
        # print(totalsmooth)
        a=1
        tampunglikeli = []
        for a in range(len(hasilSmoothAll)):
            likeli = hasilSmoothAll[a] / totalsmooth 
            tampunglikeli.append(likeli)
        rowjudul = "Positif"
        tampunglikeli.insert(0, rowjudul)
        reswriter.writerow(tampunglikeli)

main()