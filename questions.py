import nltk
import sys
import os
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }

    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    # Dictionary of filenames and its contents
    contents = {}

    # For each file in the directory
    for filename in os.listdir(directory):

        # Open the files to read
        with open(os.path.join(directory, filename) , 'r', encoding="utf8") as f:
            contents[filename] = f.read()
            f.close()

    return contents


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    stopwords = nltk.corpus.stopwords.words('english')
    return [word for word in nltk.tokenize.word_tokenize(document.lower()) if word.isalpha() and word not in stopwords]


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idf_dict = {}

    for text in documents.values():
        for word in text:
            if word not in idf_dict:
                idf_dict[word] = compute_word_idf(word, documents)

    return idf_dict
    

def compute_word_idf(word, documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words and a specific word, return the IDF value of the word.
    """
    total_documents = len(documents)
    num_documents_with_word = sum([1 if word in document else 0 for document in documents.values()])
    return math.log(total_documents / num_documents_with_word)


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    query_tf = {}

    for filename, document in files.items():
        query_tf[filename] = dict()
        for word in query:
            tf = document.count(word)
            query_tf[filename][word] = tf

    file_tfidf = {}

    for filename, word_tf in query_tf.items():
        file_tfidf[filename] = sum([tf * idfs[word] for word, tf in word_tf.items()])

    sorted_file_tfidf = [filename for filename, tfidf in sorted(file_tfidf.items(), key=lambda x: x[1], reverse=True)]

    return sorted_file_tfidf[:n]

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentences_mwm = {}
    for sentence, sentence_words in sentences.items():
            for word in query:
                if word in sentence_words:
                    if sentence in sentences_mwm:
                        sentences_mwm[sentence][0] = sentences_mwm[sentence][0] + idfs[word]
                    else:
                        sentences_mwm[sentence] = [idfs[word], query_term_density(query, sentences, sentence)]

    sorted_sentences = [sentence for sentence, idf in sorted(sentences_mwm.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)]

    return sorted_sentences[:n]


def query_term_density(query, sentences, sentence):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), calculate the query term density.
    Query term density is defined as the proportion of words in the 
    sentence that are also words in the query. For example, if a sentence
    has 10 words, 3 of which are in the query, then the sentence’s 
    query term density is 0.3
    """
    words_in_sentence = len(sentences[sentence])
    words_in_query = sum([1 for word in sentences[sentence] if word in query])

    return words_in_query / words_in_sentence


if __name__ == "__main__":
    main()
