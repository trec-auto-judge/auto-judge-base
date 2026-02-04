
import re
from typing import List


# def get_limit_length_chunks(sentences:List[str], limit:int) -> List[str]:
#     """Breaks texts into chunks of limited size, obeying sentence boundaries."""
#     # sentences = get_sentence_chunks_on_newline(text=text)
#     chunks:List[str] = list()
    
#     curr_chunk:List[str] = list()
#     curr_chunk_length:int = 0
    
#     for s in sentences:
#         if curr_chunk_length + len(s) <= limit:
#             curr_chunk.append(s)
#             curr_chunk_length += len(s)
#         else:
#             chunks.append( "\n".join(curr_chunk))
#             curr_chunk = list()
#             curr_chunk.append(s)
#             curr_chunk_length = len(s)
    
#     if curr_chunk_length >= 0:
#         chunks.append( "\n".join(curr_chunk))
#     return chunks
    

def get_limit_length_chunks(sentences: List[str], limit: int) -> List[str]:
    """
    Break a list of sentences into newline-separated text chunks,
    each no longer than *limit* characters.
    """
    if not (isinstance(sentences, list) and all(isinstance(s,str) for s in sentences)):
        raise RuntimeError(f"expecting sentences list of string, but got {type(sentences)}: {sentences}")
    
    chunks: List[str] = []

    curr_chunk: List[str] = []
    curr_chunk_len: int = 0

    for s in sentences:
        if curr_chunk_len + len(s) <= limit:
            # fits: append sentence
            curr_chunk.append(s)
            curr_chunk_len += len(s)
        else:
            # flush current chunk
            if curr_chunk:                        # guard against empties
                chunks.append("\n".join(curr_chunk))
            # start new chunk with this sentence
            curr_chunk = [s]
            curr_chunk_len = len(s)

    # flush last chunk (if non-empty)
    if curr_chunk:
        chunks.append("\n".join(curr_chunk))

    return chunks



def get_paragraph_chunks(text:str) -> List[str]:
    """Breaks paragraphs on double-newlines"""
    
    # Normalize multiple blank lines to just two newlines
    normalized_text = re.sub(r'\n\s*\n+', '\n\n', text)

    # Then split
    paragraphs = [p.strip() for p in normalized_text.split('\n\n') if p.strip()]
    return paragraphs

def get_sentence_chunks_on_newline(text: str) -> List[str]:
    """Breaks text on single newlines; each line is assumed to be a sentence."""

    # Normalize multiple newlines to single newlines
    normalized_text = re.sub(r'\n+', '\n', text)

    # Then split on single newlines
    sentences = [s.strip() for s in normalized_text.split('\n') if s.strip()]
    return sentences

def get_sentence_chunks_blingfire(text) -> List[str]:
    """Breaks sentences with blingfire's sentence splitter"""
    
    import blingfire
    
    # Sentence split
    sentences = blingfire.text_to_sentences(text).split('\n')
    return sentences





import re
class NltkInitializer():
    # import nltk
    # import nltk.stem #  import PorterStemmer
    # import fuzzywuzzy.fuzz #import fuzz
    # import nltk.corpus # import stopwords
    # import nltk.corpus # import stopwords
    # import nltk.tokenize # word_tokenize

    # x = nltk.download('stopwords')
    # y = nltk.download('punkt')  
    # stemmer = nltk.stem.PorterStemmer()

    # word_tokenize = nltk.tokenize.word_tokenize
    # stopwords = nltk.corpus.stopwords
    # fuzzratio = fuzzywuzzy.fuzz.ratio
    # # def fuzz(*kwargs):
    # #     return fuzzywuzzy.fuzz(kwargs)
    
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

    # Download necessary resources (only if not already present)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

    STOP_WORDS = set(stopwords.words('english'))
    STEMMER = PorterStemmer()





def normalize_text_sequence(text: str) -> list[str]:
    """
    Normalize text for near-duplicate detection while preserving sequence.
    Returns a list of stemmed content words in order.
    """
    # 1. Lowercase
    text = text.lower()

    # 2. Strip punctuation and symbols (preserve alphanumeric and whitespace)
    text = re.sub(r'[^\w\s]', '', text)

    # 3. Tokenize
    tokens = NltkInitializer.word_tokenize(text)

    # 4. Filter out stopwords
    content_tokens = [token for token in tokens if token not in NltkInitializer.STOP_WORDS]

    # 5. Stem each token
    stemmed_tokens = [NltkInitializer.STEMMER.stem(token) for token in content_tokens]

    return stemmed_tokens
     