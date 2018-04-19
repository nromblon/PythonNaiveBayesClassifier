import numpy as np
import tkinter as tk
import csv
from tkinter import filedialog as fd
from sklearn.naive_bayes import MultinomialNB
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
        self.train_browseButton = None
        self.train_listBox = None
        self.train_Progressbar = None
        self.trainState_Label = None
        self.train_model = MultinomialNB()
        self.test_listBox = None
        self.view_listBox = None
        self.predicted_label = None
        self.actual_label = None
        self.email_text = None
        self.truepositive_lbl = None
        self.falsenegative_lbl = None
        self.truenegative_lbl = None
        self.falsepositive_lbl = None
        # init naive bayes variables
        self.train_dirs = None
        self.train_dict = {}
        self.train_labels = None
        self.train_matrix = None
        self.training_list = []
        self.test_list = []
        self.test_matrix = None
        self.test_labels = None
        self.result = None

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
        self.train_listBox.pack(side='right', expand=True, fill='both')
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
        self.test_listBox.pack(side='right', expand=True, fill='both')
        self.test_listBox.config(yscrollcommand=test_scrollbar.set)
        test_scrollbar.config(command=self.test_listBox.yview)
        # end of list box frame
        listbox_Label = tk.Label(rightFrame, text="Select the files that are labelled as SPAM")
        listbox_Label.pack(side='top')
        testButton = tk.Button(rightFrame, text="Test", command=self.test)
        testButton.pack()

        # left frame
        leftFrame = tk.Frame(self)
        leftFrame.pack(expand=True, fill='both', side='right')

        # email traversal
        view_listFrame = tk.Frame(leftFrame)
        view_listFrame.pack(fill='both', expand=True, side='left')
        view_scrollbar = tk.Scrollbar(view_listFrame)
        view_scrollbar.pack(side='right', fill='y')
        self.view_listBox = tk.Listbox(view_listFrame, bg='white', selectmode="browse")
        self.view_listBox.pack(side='right', expand=True, fill='both')
        self.view_listBox.bind(sequence='<<ListboxSelect>>', func=self.email_onselect)
        self.view_listBox.config(yscrollcommand=view_scrollbar.set)
        view_scrollbar.config(command=self.view_listBox.yview)

        # email viewer
        viewer_frame = tk.Frame(leftFrame)
        viewer_frame.pack(fill='both', expand=True, side='top')
        self.predicted_label = tk.Label(viewer_frame, text="Predicted: ")
        self.predicted_label.pack()
        self.actual_label = tk.Label(viewer_frame, text="Actual: ")
        self.actual_label.pack()
        # email body frame
        text_frame = tk.Frame(viewer_frame)
        text_frame.pack(fill='both', expand=True)
        email_scrollbar = tk.Scrollbar(text_frame)
        email_scrollbar.pack(side='right', fill='y')
        self.email_text = tk.Text(text_frame, wrap='word')
        self.email_text.pack(side='right', fill='both', expand=True)
        self.email_text.config(yscrollcommand=email_scrollbar.set)
        self.email_text.insert('end'," ")
        email_scrollbar.config(command=self.email_text.yview)
        # end of email body frame

        # summary
        summary_frame = tk.Frame(leftFrame)
        summary_frame.pack(fill='x', expand=True, side='top')
        saveresults_button = tk.Button(summary_frame, text="Export results to CSV", command=self.export_results)
        saveresults_button.pack(side='bottom')
        left_summary_frame = tk.Frame(summary_frame)
        left_summary_frame.pack(fill='both', expand=True, side='left')
        self.truepositive_lbl = tk.Label(left_summary_frame, text="True Positives: ")
        self.truepositive_lbl.pack()
        self.falsenegative_lbl = tk.Label(left_summary_frame, text="False Negatives: ")
        self.falsenegative_lbl.pack()
        right_summary_frame = tk.Frame(summary_frame)
        right_summary_frame.pack(fill='both', expand=True, side='left')
        self.falsepositive_lbl = tk.Label(right_summary_frame, text="False Positives: ")
        self.falsepositive_lbl.pack()
        self.truenegative_lbl = tk.Label(right_summary_frame, text="True Negatives: ")
        self.truenegative_lbl.pack()

    def email_onselect(self, event):
        index = self.view_listBox.curselection()
        file = self.view_listBox.get(index)
        self.email_text.delete(1.0, 'end')
        with open(file, 'r') as m:
            self.email_text.insert('end', m.read())

        res_text = "Spam" if self.result[index] == 1 else "Not Spam"
        self.predicted_label.config(text="Predicted: " + res_text)
        res_text = "Spam" if self.test_labels[index] == 1 else "Not Spam"
        self.actual_label.config(text="Actual: " + res_text)

    def export_results(self):
        dlg = fd.asksaveasfile(defaultextension='.csv', title="Save File", filetypes=[("CSV file", "*.csv")])
        filename = dlg.name
        print(filename)
        if not filename:
            return

        data = self.generate_confusion_dict(self.result_matrix) + ["",""] \
               + self.generate_result_dict(self.test_labels, self.result)
        self.export_to_csv(filename, data)

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

    def test(self):
        # prepare labels and feature vectors for each test mail
        self.test_matrix = self.extract_features(self.test_list, self.train_dict)
        self.test_labels = np.zeros((len(self.test_list)))
        # prepare labels and feature vectors for each testing mail
        for i in range(0, len(self.training_list)):
            if i in self.test_listBox.curselection():
                self.test_labels[i] = 1

        self.result = self.train_model.predict(self.test_matrix)
        self.result_matrix = confusion_matrix(self.test_labels, self.result)
        print(self.result)
        print(self.result_matrix)

        self.view_listBox.delete(0, 'end')
        for file in self.test_list:
            self.view_listBox.insert('end', file)

        self.truepositive_lbl.config(text="True Positive: "+str(self.result_matrix[1][1]))
        self.falsepositive_lbl.config(text="False Positive: "+str(self.result_matrix[0][1]))
        self.truenegative_lbl.config(text="True Negative: "+str(self.result_matrix[0][0]))
        self.falsenegative_lbl.config(text="False Negative: "+str(self.result_matrix[1][0]))

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

    @staticmethod
    def generate_result_dict(actual, predicted):
        data = [["", "Actual", "Predicted"]]
        for i in range(0, len(actual)):
            data.append(["Email " + str(i), str(actual[i]), str(predicted[i])])

        return data

    @staticmethod
    def generate_confusion_dict(confusion):
        data = [["", "", "Predicted", ""], ["", "", "Not Spam", "Spam"],
                ["Actual", "Not Spam", confusion[0][0], confusion[0][1]],
                ["", "Spam", confusion[1][0], confusion[1][1]]]

        return data

    @staticmethod
    def export_to_csv(filename, data):
        myFile = open(filename, 'w', newline="")
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(data)


# MAIN
app = Application()
app.master.title('NaiveBayes Spam Classifier')
app.mainloop()
