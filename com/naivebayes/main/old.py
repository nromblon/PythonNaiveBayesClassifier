import os
import numpy as np
import csv
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from collections import Counter


# ---NaiveBayes Util Functions
def make_dictionary(train_dir):
    emails = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    all_words = []
    for mail in emails:
        with open(mail) as m:
            for i, line in enumerate(m):
                if i == 2:  # Body of email is only 3rd line of text file
                    words = line.split()
                    all_words += words

    dictionary = Counter(all_words)
    list_to_remove = list(dictionary)
    for item in list_to_remove:
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]

    dictionary = dictionary.most_common(3000)

    return dictionary


def extract_features(mail_dir, dictionary):
    files = [os.path.join(mail_dir, fi) for fi in os.listdir(mail_dir)]
    print("extracting")
    features_matrix = np.zeros((len(files), 3000))
    docID = 0;
    fileCount = 0
    for fil in files:
        with open(fil) as fi:
            fileCount += 1
            print("extracting features of file# ", fileCount)
            for i, line in enumerate(fi):
                if i == 2:
                    words = line.split()
                    for word in words:
                        wordID = 0
                        for i, d in enumerate(dictionary):
                            if d[0] == word:
                                wordID = i
                                features_matrix[docID, wordID] = words.count(word)
            docID = docID + 1

    return features_matrix

def generate_result_dict(actual, predicted):
    data = [["", "Actual", "Predicted"]];
    for i in range(0, len(actual)):
        data.append(["Email " + str(i), str(actual[i]), str(predicted[i])])

    return data

def export_to_csv(filename, data):
    myFile = open(filename, 'w', newline="")
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(data)


# main
# train_dir = "C:\\Users\\ASUS i7\\Documents\\ADVSTAT\\lingspam_less_salt\\train-mails"
train_dir = "C:\\Users\\Luis Madrigal\\Documents\\ADVSTAT\\lingspam_less_salt\\train-mails"
dictionary = make_dictionary(train_dir)

# Prepare feature vectors per training mail and its labels

train_labels = np.zeros(702)
train_labels[351:701] = 1
train_matrix = extract_features(train_dir, dictionary)

# Training SVM and Naive bayes classifier

model1 = MultinomialNB()
model2 = LinearSVC()
model1.fit(train_matrix,train_labels)
model2.fit(train_matrix,train_labels)

# Test the unseen mails for Spam
# test_dir = 'C:\\Users\\ASUS i7\\Documents\\ADVSTAT\\lingspam_less_salt\\test-mails'
test_dir = 'C:\\Users\\Luis Madrigal\\Documents\\ADVSTAT\\lingspam_less_salt\\test-mails'
test_matrix = extract_features(test_dir, dictionary)
test_labels = np.zeros(260)
test_labels[130:260] = 1

result1 = model1.predict(test_matrix)
result2 = model2.predict(test_matrix)
result3 = model1.predict_proba(test_matrix)
print(result1)
print (len(result1))
print(confusion_matrix(test_labels,result1))
print(result2)
print (len(result2))
print(confusion_matrix(test_labels,result2))
print(result3)

export_to_csv("MultinomialNB.csv", generate_result_dict(test_labels, result1))

export_to_csv("ConfusionMatrix.csv", confusion_matrix(test_labels, result1))