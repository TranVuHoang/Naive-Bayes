from underthesea import word_tokenize
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfTransformer
)


def calculate_tf_idf(
        data, count_vectorizer=None, tf_idf_transformer=None
):
    if not (count_vectorizer and tf_idf_transformer):
        count_vectorizer = CountVectorizer()
        tf_idf_transformer = TfidfTransformer()
        x_counts = count_vectorizer.fit_transform(data)
        x_tf_idf = tf_idf_transformer.fit_transform(x_counts)
    else:
        x_counts = count_vectorizer.transform(data)
        x_tf_idf = tf_idf_transformer.transform(x_counts)
    return [count_vectorizer, tf_idf_transformer, x_counts, x_tf_idf]


if __name__ == '__main__':
    label_map = {
        '__label__trung_binh': 0,
        '__label__kem': 1,
        '__label__rat_kem': 2,
        '__label__tot': 3,
        '__label__xuat_sac': 4
    }
    data_with_label = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: []
    }
    with open('trainSV.txt') as file:
        for line in file:
            if line[-1] == '\n':
                data = line[:-1]
            else:
                data = line
            split_data = data.split(' ')
            label = split_data[0]
            content = ' '.join(split_data[1:])
            if label not in label_map:
                continue
            data_with_label[label_map[label]].append(content)

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for label, contents in data_with_label.items():
        contents_length = len(contents)
        separate_index = (contents_length * 2) // 3
        for index, content in enumerate(contents):
        # for content in contents:
            content = content.lower()
            words = word_tokenize(content)
            new_words = list(map(
                lambda word: '_'.join(word.split(' ')), words)
            )
            content_after_handling = ' '.join(new_words)
            if index <= separate_index:
                x_train.append(content_after_handling)
                y_train.append(label)
            else:
                x_test.append(content_after_handling)
                y_test.append(label)
    # x_test = []
    print('Start training')
    (count_vectorizer, tf_idf_transformer, x_train_counts,
        x_train_tf_idf) = calculate_tf_idf(x_train)
    # with open('sentiment_analysis_test.v1.0.txt') as file:
    #     for line in file:
    #         x_test.append(line)

    (count_vectorizer, tf_idf_transformer, x_test_counts,
        x_test_tf_idf) = calculate_tf_idf(
            x_test,
            count_vectorizer,
            tf_idf_transformer
        )

    clf = LinearSVC(C=0.1)
    clf.fit(x_train_tf_idf, y_train)
    max_label_map = []
    number_true = 0
    for index, data_test in enumerate(x_test_tf_idf):
        y_predict = clf.predict(data_test.toarray())
        if y_predict == y_test[index]:
            number_true += 1
    #     max_label_map.append(y_predict[0])
    # max_keys_list =[]

    # for index in max_label_map:
    #     for key in label_map.keys():
    #         if label_map[key] == index:
    #             max_keys_list.append(key)
    precision = number_true * 100 / len(y_test)
    print(precision)
    #
    # f = open("kqSVM.txt", "w", encoding="utf8")
    #
    # for key in max_keys_list:
    #     f.write("%s " % key)
    #     f.write("\n")
    #
    # f.close()


