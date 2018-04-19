import os
import numpy as np
import tkinter as tk
from tkinter import filedialog as fd
from tkinter.ttk import Progressbar
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from collections import Counter


# naive bayes python implementation obtained from Abhijeet Kumar
# at "Email Spam Filtering : A python implementation with scikit-learn" -- January 23, 2017
# URL: https://appliedmachinelearning.wordpress.com/2017/01/23/email-spam-filter-python-scikit-learn/

class Application(tk.Frame):

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)

        self.pack(expand=True, fill='both')
        # init widget variables
        self.quitButton = None
        self.train_browseButton = None
        self.train_listBox = None
        self.train_Progressbar = None
        self.trainState_Label = None
        self.train_model = MultinomialNB()
        self.test_listBox = None
        # init naive bayes variables
        self.train_dirs = None
        self.train_dict = {}
        self.train_labels = None
        self.train_matrix = None
        self.training_list = []
        self.test_list = []

        self.init_widgets()

    def init_widgets(self):
        # right frame
        rightFrame = tk.Frame(self)
        rightFrame.pack(fill='y', side='right')
        # right frame widgets
        # training widgets
        self.train_browseButton = tk.Button(rightFrame, text="Select Training Data", command=self.browse_train)
        self.train_browseButton.pack(side='top')
        # listbox frame
        listFrame = tk.Frame(rightFrame)
        listFrame.pack(fill='x', side='top')
        scrollbar = tk.Scrollbar(listFrame)
        scrollbar.pack(side='right', fill='y')
        self.train_listBox = tk.Listbox(listFrame, bg='white', selectmode="extended")
        self.train_listBox.pack(side='right', expand=True, fill='x')
        self.train_listBox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.train_listBox.yview)
        # end of list box frame
        listbox_Label = tk.Label(rightFrame, text="Select the files that are labelled as SPAM")
        listbox_Label.pack(side='top')
        trainBtn = tk.Button(rightFrame, text="Train", command=self.train)
        trainBtn.pack()
        self.trainState_Label = tk.Label(rightFrame)
        self.trainState_Label.pack()

        # test widgets
        test_browseButton = tk.Button(rightFrame, text="Select Testing Data", command=self.browse_test)
        test_browseButton.pack()
        # listbox frame
        test_listFrame = tk.Frame(rightFrame)
        test_listFrame.pack(fill='x', side='top')
        test_scrollbar = tk.Scrollbar(test_listFrame)
        test_scrollbar.pack(side='right', fill='y')
        self.test_listBox = tk.Listbox(test_listFrame, bg='white', selectmode="extended")
        self.test_listBox.pack(side='right', expand=True, fill='x')
        self.test_listBox.config(yscrollcommand=test_scrollbar.set)
        test_scrollbar.config(command=self.train_listBox.yview)
        # end of list box frame
        listbox_Label = tk.Label(rightFrame, text="Select the files that are labelled as SPAM")
        listbox_Label.pack(side='top')
        testButton = tk.Button(rightFrame, text="Test", command=self.train)

        self.quitButton = tk.Button(rightFrame, text='Quit', command=self.quit)
        self.quitButton.pack(side='bottom')

        # left frame
        leftFrame = tk.Frame(self, bg="red")
        leftFrame.pack(expand=True, fill='both', side='right')
        files_tkLabel2 = tk.Label(leftFrame, text="files")
        files_tkLabel2.pack()
        train_browseButton2 = tk.Button(leftFrame, text="Browse", command=self.browse_train)
        train_browseButton2.pack()

    def browse_train(self):
        self.training_list = list(fd.askopenfilenames())
        self.train_listBox.delete(0, 'end')
        for file in self.training_list:
            self.train_listBox.insert('end', file)

    def browse_test(self):
        self.test_list = list(fd.askopenfilenames())
        self.test_listBox.delete(0, 'end')
        for file in self.test_list:
            self.test_listBox.insert('end', file)

    # ---NaiveBayes Standard Functions
    def train(self):
        self.trainState_Label.config(text="Training . . .")
        # prepare overall frequency count of words in all training mails
        self.train_dict = self.make_dictionary(self.training_list)
        print(self.train_dict)
        self.train_labels = np.zeros(len(self.training_list))
        # prepare labels and feature vectors for each training mail
        for i in range(0, len(self.training_list)):
            if i in self.train_listBox.curselection():
                self.train_labels[i] = 1

        self.train_matrix = self.extract_features(self.training_list, self.train_dict)
        # fit the naive bayes model with the feature vectors
        self.train_model.fit(self.train_matrix, self.train_labels)
        self.trainState_Label.config(text="Training Complete!")

    # ---NaiveBayes Util Functions
    @staticmethod
    def make_dictionary(train_list):
        emails = train_list
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
    def extract_features(mails, dictionary):
        files = mails
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
