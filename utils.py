import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def extract_named_entities(text):
    from nltk import word_tokenize, pos_tag, ne_chunk
    return [chunk for chunk in ne_chunk(pos_tag(word_tokenize(text))) if hasattr(chunk, 'label')]
