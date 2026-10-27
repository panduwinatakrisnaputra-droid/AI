# Example 4.22 The NLTK.py Program
# https://www.nltk.org/
# pip install nltk


# ============================================================
# Import library
# ============================================================

import nltk


# ============================================================
# Download NLTK resources
# ============================================================

# ============================================================
# Download NLTK resources
# ============================================================

# ============================================================
# Download NLTK resources
# ============================================================

nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')


# ============================================================
# Input sentence
# ============================================================

sentence = """
Artificial intelligence (AI), is intelligence demonstrated by
machines,
unlike the natural intelligence displayed by humans and animals.

Leading AI textbooks define the field as the study of
"intelligent agents":

any device that perceives its environment and takes actions that
maximize its chance of successfully achieving its goals.

Colloquially, the term "artificial intelligence" is often used
to describe machines (or computers) that mimic "cognitive"
functions that humans associate with the human mind, such as
"learning" and "problem solving".
"""


# ============================================================
# Tokenization
# ============================================================

tokens = nltk.word_tokenize(sentence)

print("Tokens:")
print(tokens)


# ============================================================
# Part-of-Speech Tagging
# ============================================================

tagged = nltk.pos_tag(tokens)

print("\nPart-of-Speech Tags:")
print(tagged)


# ============================================================
# Named Entity Recognition
# ============================================================

entities = nltk.chunk.ne_chunk(tagged)

print("\nNamed Entities:")
print(entities)