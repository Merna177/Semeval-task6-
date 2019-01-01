import csv
import nltk.stem


class Preprocessing:

    def __init__(self, filename):
        self.filename = filename
        self.lemmatizer_object = nltk.stem.WordNetLemmatizer()
        self.labels = []
        self.tweets = []
        Preprocessing.readFile(self,filename)
        Preprocessing.divide_data_set(self)

    def remove_noise(self,input_text):
        noise_list = ["a", "about", "after", "all", "also", "an", "another", "any", "and", "are", "as", "and", "at",
                      "be",
                      "because", "been", "before", "being", "between", "but", "both", "by", "came", "can", "come",
                      "could ",
                      "did", "do", "each", "even", "for", "from", "further", "furthermore", "get", "got", "has", "had",
                      "he", "have", "her", "here", "him", "himself", "his", "how", "hi", "however", "i", "if", "in",
                      "into",
                      "is", "it", "its", "indeed", "just", "like", "made", "many", "me", "might", "more", "moreover",
                      "most", "much", "must", "my never", "not", "now of", "on", "only", "other", "our", "out", "or",
                      "over", "said", "same", "see", "should", "since", "she", "some", "still", "such", "take", "than",
                      "that", "the", "their", "them", "then", "there", "these", "therefore", "they", "this", "those",
                      "through", "to", "too", "thus", "under", "up", "was", "way", "we", "well", "were", "what", "when",
                      "where", "which", "while", "who", "will", "with", "would", "your", "null"]
        words = input_text.split()  # Split words by space
        noise_free_words = [word for word in words if word.lower() not in noise_list]  # Get a list of non-noise words
        noise_free_text = " ".join(noise_free_words)  # Get a string of non-noise words
        return noise_free_text

    def remove_regex(self,input_text):
        # split tweet by space
        words = input_text.split()
        regex_free_text = ""
        # check if word is alpha(contain letters only) , then add it to regex_free_text
        for word in words:
            if word.isalpha():
                # Lemmatization, on the other hand, is an organized & step by step procedure of obtaining
                # the root form of the word, it makes use of vocabulary (dictionary importance of words)
                # and morphological analysis (word structure and grammar relations).
                # reduces the inflected words properly ensuring that the root word belongs to the language
                # pos="V"-->to give a root for each word !
                regex_free_text += self.lemmatizer_object.lemmatize(word, pos="v")  # V Msdr
                regex_free_text += " "
        return regex_free_text

    def readFile(self,filename):
        my_file = open(filename, encoding="utf-8")
        # return value of csv file is an iterator
        read = csv.reader(my_file, delimiter='\t')
        # splitting = read.split('\t')
        flag = 0
        # flag ---> used to skip the header of the file
        # column one for tweets , column 2 for our ouput(NOT or OFF)
        for row in read:
            if flag == 0:
                flag = 1
                continue;
            self.tweets.append(row[1])
            self.labels.append(row[2])

    def divide_data_set(self):
        # loop for each tweet remove regex & noise
        for tweet in range(0, len(self.tweets)):
            self.tweets[tweet] = Preprocessing.remove_regex(self,self.tweets[tweet])
            self.tweets[tweet] = Preprocessing.remove_noise(self,self.tweets[tweet])
        # we have to divide our dataset into 2 parts (training data and test data)
        lenofLabel = (int)(len(self.labels) / 2)
        self.lenofTweets = (int)(len(self.tweets) / 2)
        self.train_labels = self.labels[:lenofLabel]
        self.test_labels = self.labels[lenofLabel:]
        self.train_tweets = self.tweets[:lenofLabel]
        self.test_tweets = self.tweets[lenofLabel:]