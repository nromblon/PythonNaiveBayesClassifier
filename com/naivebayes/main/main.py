import os
import numpy as np
import tkinter as tk
from tkinter import filedialog as fd
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from collections import Counter


class Application(tk.Frame):

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.grid()
        # init widget variables
        self.quitButton = None
        self.train_browseButton = None
        self.files_label = None
        self.initWidgets()
        # init naive bayes variables
        self.train_dirs = None
        self.train_dict = {}
        self.train_labels = None
        self.train_matrix = None


    def initWidgets(self):
        self.quitButton = tk.Button(self, text='Quit', command=self.quit)
        self.quitButton.grid(row=1, column=0)

        self.train_browseButton = tk.Button(self,text="Browse", command=self.browse_train)
        self.train_browseButton.grid(row=0, column=0)

        self.files_label = tk.Label(self,text="files")
        self.files_label.grid(row=0, column=1)

    def browse_train(self):
        self.train_dirs = fd.askopenfilenames()
        self.files_label['text'] = self.train_dirs

    # ---NaiveBayesStandardFunctions
    def train(self, train_dir):
        self.train_dict = self.make_dictionary(train_dir)


    # ---NaiveBayes Util Functions
    @staticmethod
    def make_dictionary(self, train_dir):
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

    @staticmethod
    def extract_features(self, mail_dir, dictionary):
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


# MAIN
app = Application()
app.master.title('NaiveBayes Spam Classifier')
app.mainloop()
