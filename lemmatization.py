import nltk
import csv
import jamspell
import re
import time

nltk.download("stopwords")

from nltk.corpus import stopwords
from pymystem3 import Mystem
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from collections import defaultdict
from collections import Counter
from operator import itemgetter

mystem = Mystem()
russian_stopwords = stopwords.words("russian")


corrector = jamspell.TSpellCorrector()
corrector.LoadLangModel('ru_small.bin')


lemmatizedWords = defaultdict(Counter)
numberOfWords = 0


def lemmatizeWord(word):
    global numberOfWords
    numberOfWords += 1
    return mystem.lemmatize(word)[0]


def preprocessLine(line):
    return re.sub(r'(.)(\1){2,}', r'\1', re.sub(r'[ ]{2,}', ' ', re.sub(r'[^а-яё\s]', ' ', line).replace('ё', 'e')))

def correctSpelling(line):
    return corrector.FixFragment(line)


def wordIsCorrect(word):
    return word not in russian_stopwords and word is not ' '


def getWords(line):
    return line.split()


def processLine(line):
    preprocessed = preprocessLine(line.lower())
    correctedWords = getWords(correctSpelling(preprocessed))
    words = getWords(preprocessed)
    for wordIndex in range(len(words)):
        word = correctedWords[wordIndex]
        if wordIsCorrect(word):
            lemma = lemmatizeWord(word)
            lemmatizedWords[lemma][words[wordIndex]] += 1


def lineIsCorrect(line):
    try:
        return detect(line) == 'ru'
    except LangDetectException as e:
        return False


def getDataFromLine(line):
    return line[3].strip()

def processText(text):
    for line in text:
        try:
            data = getDataFromLine(line)
        except IndexError:
            continue
        if lineIsCorrect(data):
            processLine(data)


def getDataFromFile(fileName):
    with open(fileName, newline='') as csv_file:
        return list(csv.reader(csv_file, delimiter='\t'))


def output():
    with open('results.csv', 'w') as resultFile:
        writer = csv.writer(resultFile)
        for lemma, counter in lemmatizedWords.items():
            words = []
            for word, occurrence in sorted(counter.items(), key=itemgetter(1), reverse=True):
                words.append(word + "[" + str(occurrence) + "]")
            writer.writerow([lemma, words])

if __name__ == '__main__':
    startTime = time.time()
    processText(getDataFromFile('test.csv'))
    endTime = time.time()
    print("Performance: %d words per second" % (numberOfWords / (endTime - startTime)))
    output()