"""
Gather sentiment from a database.
The database contains a list of sentences. Rank them from 1 to 5 indicating from most negative to most positive.
Then rank them with respect to a certain subjective/entity.
Basically, whether they would affect entity in positive/negative way
Used for study relationship between news titles and stock market.

There is difference between news and twitter moods. Maybe we want to use twitter moods data because human helps
use interpret the news (facts) data into moods and how they affect the stock directly.
"""

import os
import sys

import pandas as pd
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QTextEdit, QGridLayout, QApplication, QLineEdit, QPushButton, QFileDialog, \
    QGroupBox, QRadioButton, QHBoxLayout


class SentimentLibrary(object):
    def __init__(self):
        self.input_csv = None
        self.output_csv = None

    def get_outputfile(self, inputFile):
        dir_path = os.path.dirname(os.path.realpath(inputFile))
        return os.path.join(dir_path, inputFile.split('.')[0] + '_output.csv')

    def setInputFile(self, inputFile):
        """ The output file has the same content but with one more column of label

        Args:
            inputFile: a csv file containing the sentences and other information

        Returns:

        """
        self.input_csv = pd.read_csv(inputFile)
        # output file is in the same directory, with name append with out.
        self.output_csv_path = self.get_outputfile(inputFile)
        if not os.path.isfile(self.output_csv_path):
            self.output_csv = pd.DataFrame(columns=list(self.input_csv.columns) + ['Rating'])
            self.current_index = 0

        else:
            self.output_csv = pd.read_csv(self.output_csv_path)
            self.current_index = self.output_csv.shape[0]

        # read the current output file and go to the line that without a sentiment.

    def getCurrentSentence(self):
        return self.input_csv.iloc[self.current_index]['News']

    def submitResult(self, rating):
        # copy current index from source, add rating and append to output
        row = self.input_csv.iloc[self.current_index].copy()
        row['Rating'] = rating
        self.output_csv = self.output_csv.append(row)
        self.current_index += 1

    def save(self):
        if self.output_csv is not None:
            self.output_csv.to_csv(self.output_csv_path, index=False)
            print('Save output to {}'.format(self.output_csv_path))

    def close(self):
        if self.input_csv is not None:
            self.input_csv.close()
            self.output_csv.to_csv(self.output_csv_path)
            print('Dump output to {}'.format(self.output_csv_path))


class SentimentRater(QMainWindow):
    def __init__(self):
        super(SentimentRater, self).__init__()
        self.window_loc = (250, 250, 800, 400)  # left, top, width, height

        self.sentimentLib = SentimentLibrary()

        self.initUI()

    def close(self) -> bool:
        print('Close')
        self.sentimentLib.close()
        return super(SentimentRater, self).close()

    def initUI(self):
        self.setWindowTitle("Sentiment Rater")

        # input file is used to display the select file to rate sentiment
        self.inputFileSectionTextBox = QLineEdit(parent=self)
        self.inputFileSectionTextBox.resize(600, 20)
        self.inputFileSectionTextBox.move(20, 20)
        self.inputFileSectionTextBox.setReadOnly(True)

        self.selectFileButton = QPushButton('Select File', parent=self)
        self.selectFileButton.move(630, 15)
        self.selectFileButton.clicked.connect(self.selectFile)

        # a text box to display sentiment sentences
        self.sentenceTextBox = QTextEdit(parent=self)
        self.sentenceTextBox.resize(600, 100)
        self.sentenceTextBox.move(20, 60)
        # self.sentenceTextBox.setReadOnly(True)

        # 5 mutual exclusive checkbox
        self.groupBox = QGroupBox("Rating", parent=self)
        self.groupBox.move(20, 200)
        self.groupBox.resize(200, 150)

        self.radioGroup = []

        radio1 = QRadioButton("Very Negative (1)")
        radio2 = QRadioButton("Negative (2)")
        radio3 = QRadioButton("Neutral (3)")
        radio4 = QRadioButton("Positive (4)")
        radio5 = QRadioButton("Very Positive (5)")

        self.radioGroup.append(radio1)
        self.radioGroup.append(radio2)
        self.radioGroup.append(radio3)
        self.radioGroup.append(radio4)
        self.radioGroup.append(radio5)

        radio1.setChecked(True)

        vbox = QHBoxLayout()
        grid = QGridLayout()

        vbox.addLayout(grid)

        grid.addWidget(radio1)
        grid.addWidget(radio2)
        grid.addWidget(radio3)
        grid.addWidget(radio4)
        grid.addWidget(radio5)

        self.groupBox.setLayout(vbox)
        self.setGeometry(*self.window_loc)

        self.submitButton = QPushButton('Submit', parent=self)
        self.submitButton.move(300, 200)
        self.submitButton.clicked.connect(self.onSubmit)

        self.saveButton = QPushButton('Save', parent=self)
        self.saveButton.move(300, 250)
        self.saveButton.clicked.connect(self.onSave)

        self.show()

    @pyqtSlot()
    def onSave(self):
        self.sentimentLib.save()

    @pyqtSlot()
    def onSubmit(self):
        # get the result from checkbox
        for i in range(len(self.radioGroup)):
            if self.radioGroup[i].isChecked():
                self.sentimentLib.submitResult(i)
                print("The rating is {}. i is {}".format(self.radioGroup[i].text(), i))
                break

        sentence = self.sentimentLib.getCurrentSentence()
        # self.sentenceTextBox.setReadOnly(False)
        self.sentenceTextBox.clear()
        self.sentenceTextBox.setPlainText(sentence)
        self.sentenceTextBox.show()
        # self.sentenceTextBox.setReadOnly(True)
        QApplication.processEvents()

    @pyqtSlot()
    def selectFile(self):
        options = QFileDialog.Options()

        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "CSV File (*.csv)", options=options)

        self.sentimentLib.setInputFile(fileName)
        self.inputFileSectionTextBox.setText(fileName)

        sentence = self.sentimentLib.getCurrentSentence()
        self.sentenceTextBox.setPlainText(sentence)
        QApplication.processEvents()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SentimentRater()
    sys.exit(app.exec_())
