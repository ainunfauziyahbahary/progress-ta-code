word1 = "A"
word2 = "few"
word3 = "good"
word4 = "words"
wordList = ["A", "few", "more", "good", "words"]


#Joining a list of words
sentence = "Second:"
for word in wordList:
    sentence += " " + word
sentence += "."
print (sentence)