# import the spacy library
import spacy

# import displacy
from spacy import displacy

# import word_tokenize
from nltk.tokenize import word_tokenize

# load the english model and initialize an object called 'nlp'
nlp = spacy.load("en_core_web_sm")

# remove stopwords
def remove_stopwords(text):
    """
    Removes stopwords passed from the text passed as an arguments
    
    Arguments:
    text: raw text from where stopwords need to removed
    
    Returns:
    tokens_without_sw: list of tokens of raw text without stopwords
    """
    # getting list of default stop words in spaCy english model
    stopwords =nlp.Defaults.stop_words
    
    # tokenize text
    text_tokens = word_tokenize(text)
    
    # remove stop words:
    tokens_without_sw = [word for word in text_tokens if word not in stopwords]
    
    # return list of tokens with no stop words
    return tokens_without_sw

# tokenize words
def tokenize_word(text):
    """
    Tokenize the text passed as an arguments into a list of words(tokens)
    
    Arguments:
    text: raw text
    
    Returns:
    words: list containing tokens in text
    """
    # passing the text to nlp and initialize an object called 'doc'
    doc = nlp(text)
    
    # Tokenize the doc using token.text attribute
    words = [token.text for token in doc]
        
    # return list of tokens
    return words

# tokenize sentence
def tokenize_sentence(text):
    """
    Tokenize the text passed as an arguments into a list of sentence
    
    Arguments:
    text: raw text
    
    Returns:
    sentences: list of sentences
    """
    # passing the text to nlp and initialize an object called 'doc'
    doc = nlp(text)
    
    # tokenize the sentence using sents attributes
    sentences = list(doc.sents)
    
    # return tokenize sentence
    return sentences

# remove punctuations
def remove_punctuation(text):
    """
    removes punctuation symbols present in the raw text passed as an arguments
    
    Arguments:
    text: raw text
    
    Returns: 
    not_punctuation: list of tokens without punctuation
    """
    # passing the text to nlp and initialize an object called 'doc'
    doc = nlp(text)
    
    not_punctuation = []
    # remove the puctuation
    for token in doc:
        if token.is_punct == False:
            not_punctuation.append(token)
    
    return not_punctuation

# lower casing
def lower_casing(text):
    """
    Accepts text as arguments and return text in lowercase
    
    Arguments:
    text: raw text
    
    Returns:
    text_to_lower: text converted to lower case
    """
    text_to_lower = text.lower()
    
    return text_to_lower

# Lemmatization
def lemmatization(text):
    """
    obtain the lemma of the each token in the text, append to the list, and returns the list
    
    Arguments:
    text: raw text
    
    Returns:
    token_lemma_list: list containing token with its lemma
    """
    
    # passing the text to nlp and initialize an object called 'doc'
    doc = nlp(text)
    
    token_lemma_list = []
    # Lemmatization
    for token in doc:
        token_lemma_list.append((token.text, token.lemma_))
    
    return token_lemma_list

# Pos-tagging
def pos_tagging(text):
    # passing the text to nlp and initialize an object called 'doc'
    doc = nlp(text)
    
    pos_list = []
    for token in doc:
        pos_list.append((token.text, token.pos_, token.tag_))
    return pos_list

# Named entity recognition
def named_entity_recognition(text):
    """
    returns entity_text and entity labels as a tuple
    
    Arguments:
    text: raw text
    
    Returns:
    entity_text_label: entity text and labels as a tuple
    """
    # passing the text to nlp and initialize an object called 'doc'
    doc = nlp(text)
    
    #named entity recogniton using doc.ents
    entity_text_label = []
    
    for entity in doc.ents:
        entity_text_label.append((entity.text, entity.label_))
        
    return entity_text_label

# define sample text, stopwords removal
sample_text = "Oh man, this is pretty cool. We will do more such things."
# remove stopwords calling defined functions
filtered_sentence = remove_stopwords(sample_text)
print("**filtered sentence without stop words:**\n", filtered_sentence)

# tokenize  words
words = tokenize_word(sample_text)
# print tokens
print("\n**Word tokens**\n", words)

# tokenize sentence
sentences = tokenize_sentence(sample_text)
# print sentences
print("\n**Sentence tokens**\n", sentences)

# remove punctuation
not_punctuation = remove_punctuation(sample_text)
print("\n**list of tokens without punctutaions**\n", not_punctuation)

# lower casing
print("\n**Lower Casing**\n", lower_casing(sample_text))

# define sample text, Lemmatization
sample_text = "The Republican president is being challenged by Democratic Party nominee Joe Biden"
# Lemmatization
token_lemma_list = lemmatization(sample_text)
#printing
print("\n**Lemmatization**")
for token_lemma in token_lemma_list:
    print(token_lemma[0], '-->', token_lemma[1])


 # define sample text, POS Tagging
sample_text = 'Antibiotics do not help, as they do not work against viruses.'
# pos_tagging
pos_list = pos_tagging(sample_text)
# display
print("**POS Tagging**")
for pos in pos_list:
    print(pos[0], pos[1], pos[2])  


# define sample text, NER
sample_text = "The Republican president is being challenged by Democratic Party nominee Joe Biden, who \
                is best known as Barack Obama’s vice-president but has been in US politics since the 1970s"
# Named Entity Recognition
entity_text_label = named_entity_recognition(sample_text)
# display entity text and label
print("\n**Named Entity Recognition**")
for text_label in entity_text_label:
    print(text_label[0], '->', text_label[1])
# Visualizing the named entity description
print("\n***VISUALIZING NAMED ENTITY RECOGNIZER***")
displacy.render(nlp(sample_text), style = "ent",jupyter = True)