import re
import math
from underthesea import word_tokenize
import numpy as np
from collections import defaultdict
from langdetect import detect

# đọc tệp
train = []
for line in open('E:/KPDL/train2.txt', 'r' , encoding="utf8"):
    train.append(line.strip())

# # lọc ngôn ngữ
# for line in open('F:/Desktop/sentiment_analysis_train.v2.0.txt', 'r', encoding="utf8"):
#     if detect(line) == "vi":
#         train.append(line.strip())
# print(len(train))
# for line in open('F:/Desktop/sentiment_analysis_train.v1.0.txt', 'r', encoding="utf8"):
#     if detect(line) == "vi":
#         train.append(line.strip())

# làm loại bỏ dấu
def removePunctuation(list):
    list = [re.sub(r'[^\w\s]',' ',line.lower()) for line in list]
    return list

# Loại bỏ khoảng trắng thừa
def removeSpace(list):
    list = [" ".join(line.split()) for line in list]
    return list

#tách từ
def splitWord(list):
    list = [word_tokenize(line) for line in list]
    return list
from string import digits
# loại bỏ số
def removeNumber(list):
    output = [re.sub(r'\d+', ' ', line.lower()) for line in list]
    return output
def removeLanguage(train):
    data = []
    dline = []
    for line in train:
        # print(line)
        for word in line:
            if (word == line[0]):
                dline.append(word)
                continue
            try:
                language = detect(word)
                if language != 'vi':
                    continue
            except:
                continue
            dline.append(word)
        # print(line)

            # import pdb; pdb.set_trace()

        data.append(dline)
        dline = []
    return data

def removeStopWord(list):
    stopWord =[]
    for line in open('F:/StopWords.txt', 'r' , encoding="utf8"):
        stopWord.append(line.strip())
    data = []
    dline = []
    for line in list:
        for word in line:
            if word in  stopWord:
                continue
            dline.append(word)
        data.append(dline)
        dline = []

    return data

# test = []
# for line in open('F:/Test.txt', 'r' , encoding="utf8"):
#     test.append(line.strip())
# test = removePunctuation(test)
# train = removeNumber(test)
# # train = removeLanguage(train)
# test = removeSpace(test)
# test = splitWord(test)
# removeStopWord(test)
# # print(test)
# train = removePunctuation(train)
# train = removeNumber(train)
# train = removeSpace(train)
train = splitWord(train)
# train = removeLanguage(train)
# train = removeStopWord(train)

def setLabel(train):
    for line in train:
        if (line[0] == '__label__xuat_sac'):
            line[0] ='__label__tot'
        if (line[0] == '__label__rat_kem'):
            line[0] = '__label__kem'
# setLabel(train)
# print(train)

# tách lấy nhãn
def splitLabel(train):
    aLabel = []
    count = 0
    for line in train:
        for label in aLabel:
            if label == line[0]:
                count = count +1;
        if count == 0: aLabel.append(line[0])
        count = 0
    return aLabel
# slabel = ['__label__tot', '__label__trung_binh', '__label__kem']
# tạo 1 từ điển nhãn
def dictLabel(train):
    label = dict()
    for i in splitLabel(train):
        label[i] = 0
    return label

# print(dictLabel())
# đếm số lượng từ trong các nhãn
def countWord(train):
    data = dict()
    for line in train:
        for word in line:
            if word == line[0]:
                continue
            key = word
            if word in data.keys():
                data[key][line[0]] +=1
            else:
                data.setdefault(word , dict())
                data[key] = dictLabel(train)
                data[key][line[0]] += 1
    return data
# print((C))
# countWord(train)
# số lượng mỗi từ trong một label
def countAllWordInLabel(train):
    label = dictLabel(train)
    for line in train:
        for i in line:
            if i == line[0]:
                continue
            label[line[0]] += 1
    return label

def writeToFile(dict, file):
    file = open(file, 'w', encoding="utf8")
    label = dict
    for key, vla in label.items():
        file.write('%s:%s \n' % (key, vla))
    file.close()

# writeToFile(countAllWordInLabel(train), 'E:/Label.txt')

# đưa dữ liệu đã thống kê ra file
def writePriorToFile(dict):
    file = open("E:/KPDL/prior.txt",'w',encoding="utf8")
    label = dict
    for key, vla in label.items():
        file.write('%s:%s \n' % (key, vla))
    file.close()

#load label from file
def openPriorFromFile():
    d = {}
    with open("E:/KPDL/prior.txt") as f:
        for line in f:
            (key, val) = line.split(':')
            d[key] = val.strip()
    return d
#
# def writeLikeihoodToFileRead(dict):
#     file = open("E:/likeihoodREAD.txt", 'w', encoding="utf8")
#     label = dict
#     for key, vla in label.items():
#         for vkey in vla.keys():
#             file.write('%s:%s:%s \n' % (key, vkey, label[key][vkey]))
#     file.close()

import pickle
def writeLikeihoodToFile(dict):
    f = open("E:/KPDL/file.pkl", "wb")
    pickle.dump(dict, f)
    f.close()

# đố lượng từ trong file
# def openLikeihoodFromFile():
#     data = dict()
#     label = dict()
# #     word = []
#     with open("E:/likeihood.txt", encoding="utf8") as f:
#         for line in f:
#             (key,vkey,vvla) = line.split(':')
#             label[vkey] = vvla.strip()
#             data[key] = label
#     return data

def openLikeihoodFromFile():
    data  = pickle.load(open( "E:/KPDL/file.pkl", "rb" ))
    return data
def compute_IDF(list):
    N_doc = len(list)
    count = 0
    icount = countWord(list)
    idf = {}
    for word in icount.keys():
        for line in list:
            if word in line: count += 1

        idf[word] = math.log10(N_doc/count)
        count = 0
    # print(N_doc)
    f = open("E:/KPDL/IDF.pkl", "wb")
    pickle.dump(idf, f)
    f.close()
    return idf
# test = []
# for line in open('F:/Test.txt', 'r' , encoding="utf8"):
#     test.append(line.strip())
# test = removePunctuation(test)
# train = removeNumber(test)
# # train = removeLanguage(train)
# test = removeSpace(test)
# test = splitWord(test)
# compute_IDF(test)

def openIDF():
    data = pickle.load(open("E:/KPDL/IDF.pkl", "rb"))
    return data

def trainNaiveBayes(train):
    documents = countAllWordInLabel(train)
    # Tính P(ci)
    N_doc = 0;
    for i in documents.keys():
        N_doc += documents[i]
    prior = {}
    for i in documents.keys():
        prior[i] = (documents[i] / N_doc)
    writePriorToFile(prior)

    # Tính P(fj|ci)
    c = likeihood= countWord(train)

    for key in c.keys():
        for vkey in c[key].keys():
            a = (c[key][vkey] +1) / (documents[vkey] + N_doc)
            likeihood[key][vkey] = a
    # import pdb; pdb.set_trace()
    # writeLikeihoodToFileRead(likeihood)
    writeLikeihoodToFile(likeihood)
    # tính IDF
    compute_IDF(train)
    # return likeihood
    # return logprior





def compute_TF(line):
    tf = {}
    N_line = len(line)
    wordDict = dict.fromkeys(line, 0)
    for word in line:
        wordDict[word] +=1
        # print(wordDict)
    # print(N_line)
    for word in wordDict.keys():
        tf[word] = wordDict[word] / N_line
    return tf
# test =[]
# for line in open('F:/Test.txt', 'r' , encoding="utf8"):
#     test.append(line.strip())
# test = removePunctuation(test)
# test = removeSpace(test)
# test = splitWord(test)
# compute_TF(test[0])
# print(test)

def predictNaiveBayes(file):
    test = []
    for line in open(file, 'r', encoding="utf8"):
        test.append(line.strip())

    test = removePunctuation(test)
    # test = removeNumber(test)
    test = removeSpace(test)
    test = splitWord(test)

    idf = openIDF()
    # setLabel(test)

    prior = openPriorFromFile()
    likeihood = openLikeihoodFromFile()
    vmax = {}
    count = all = 0
    max_keys_list = []

    for line in test:
        max_keys = line
        tf = compute_TF(line)
        for key in prior.keys():
            clabel = math.log10(float(prior[key]))
            # clabel = 0
            for i in line:
                if (i not in likeihood.keys()):
                    continue
                tfidf = tf[i]*idf[i]
                clabel = clabel + tfidf*math.log10(float(likeihood[i][key]))
                # import pdb; pdb.set_trace()
            vmax[key] =clabel
        max_value = max(vmax.values())

        for i in vmax.keys():
            if (max_value == vmax[i]):
                max_keys = i;
                break
        max_keys_list.append(max_keys)

        if max_keys == line[0]:
            count += 1
        all += 1
    # print(max_keys_list)
    docx = count/all
    print(docx)

# trainNaiveBayes(train)
predictNaiveBayes('E:/sentiment_analysis_test.v1.0.txt')
# print(predictNaiveBayes('F:/Desktop/sentiment_analysis_test.v1.0.txt'))