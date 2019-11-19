from underthesea import word_tokenize
import re
from langdetect import detect
# def removeLanguage(train):
#     data = []
#     dline = []
#     for line in train:
#         # print(line)
#         for word in line:
#             if (word == line[0]):
#                 dline.append(word)
#                 continue
#             try:
#                 language = detect(word)
#                 if language != 'vi':
#                     continue
#             except:
#                 continue
#             dline.append(word)
#         # print(line)
#
#             # import pdb; pdb.set_trace()
#
#         data.append(dline)
#         dline = []
#     return data
def splitWord(list):
    list = [word_tokenize(line) for line in list]
    return list

def removeStopWord(list):
    stopWord =[]
    for line in open('StopWords.txt', 'r' , encoding="utf8"):
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

def removeNumber(list):
    output = [re.sub(r'\d+', ' ', line.lower()) for line in list]
    return output

def removePunctuation(list):
    list = [re.sub(r'[^\w\s]',' ',line.lower()) for line in list]
    return list


def removeSpace(list):
    list = [" ".join(line.split()) for line in list]
    return list

def checkVietnamese(list, content):
    syllables = content.split()
    number_vietnamese_syllables = 0
    for syllable in syllables:
        if syllable not in list:
            continue
        number_vietnamese_syllables += 1
    return (number_vietnamese_syllables / len(syllables)) > 0.5

def removeNotVietnamese(list):
    vietnamese_dictionary = []
    for line in open('syllables_dictionary_1.txt', 'r', encoding="utf8"):
        vietnamese_dictionary.append(line.strip())
    data_list = []
    dline = []
    for line in list:
        content = ' '.join(line[1:])
        if not checkVietnamese(vietnamese_dictionary, content):
            continue
        for word in line:
            dline.append(word)
        data_list.append(dline)
        dline = []
    return data_list

if __name__ == '__main__':
    train = []
    for line in open('sentiment_analysis_train.v1.0.txt', 'r' , encoding="utf8"):
        train.append(line.strip())

    # train = removePunctuation(train)
    # train = removeNumber(train)
    # train = removeSpace(train)
    train = splitWord(train)
    # train = removeStopWord(train)
    # train = removeLanguage(train)
    # train = removeNotVietnamese(train)
    count = 0
    f = open("trainSV.txt", "w", encoding="utf8")

    for line in train:
        for word in line:
            count +=1
        if (count > 1):
            for word in line:
                f.write("%s " % word)
            f.write("\n")
        count = 0
    f.close()