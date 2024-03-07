import ir_datasets
import nltk
import spacy
import gensim
from sympy import sympify, to_dnf, Not, And, Or

# Cargar el corpus y mostrar un documento
dataset = ir_datasets.load("cranfield")
documents = [doc.text for doc in dataset.docs_iter()]

tokenized_docs = []
vector_represent = []
dictionary = {}
vocabulary = []
corpus = []

nlp = spacy.load("en_core_web_sm")


def tokenization_spacy(texts):
    '''nlp(doc) tokeniza el document y luego guarda token per token en la lista'''
    return [[token for token in nlp(doc)] for doc in texts]


def remove_noise_spacy(tokenized_docs):
    '''Elimina los tokens que no estan estan compuestos solamente por caracteres alfabeticos '''
    return [[token for token in doc if token.is_alpha] for doc in tokenized_docs]


def remove_stopwords_spacy(tokenized_docs):
    '''Busca si el token pertenece a la lista de stop_words'''
    stopwords = spacy.lang.en.stop_words.STOP_WORDS
    return [[token for token in doc if token.text not in stopwords] for doc in tokenized_docs]


def morphological_reduction_spacy(tokenized_docs, use_lemmatization=True):
    stemmer = nltk.stem.PorterStemmer()
    return [[token.lemma_ if use_lemmatization else stemmer.stem(token.text) for token in doc]
            for doc in tokenized_docs]


def filter_tokens_by_occurrence(tokenized_docs, no_below=5, no_above=0.5):
    global dictionary
    dictionary = gensim.corpora.Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)

    filtered_words = [word for _, word in dictionary.iteritems()]
    filtered_tokens = [
        [word for word in doc if word in filtered_words]
        for doc in tokenized_docs
    ]

    return filtered_tokens


def build_vocabulary(dictionary):
    vocabulary = list(dictionary.token2id.keys())
    return vocabulary


def vector_representation(tokenized_docs, dictionary, vector_repr, use_bow=True):
    global corpus
    # devuelve un dicc  word:frecuency
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
    if use_bow:
        vector_repr = corpus
    else:
        tfidf = gensim.models.TfidfModel(corpus)

        vector_repr = [tfidf[doc] for doc in corpus]

    return vector_repr


tokenized_docs = tokenization_spacy(documents)
tokenized_docs = remove_noise_spacy(tokenized_docs)
tokenized_docs = remove_stopwords_spacy(tokenized_docs)
tokenized_docs = morphological_reduction_spacy(tokenized_docs)
tokenized_docs = filter_tokens_by_occurrence(
    tokenized_docs)  # dictionary se construye aqui
vocabulary = build_vocabulary(dictionary)
vector_represent = vector_representation(
    tokenized_docs, dictionary, vector_represent)  # aqui se construye el


def query_to_dnf(query):
    query = query.lower()  # Convertir la consulta a minúsculas
    # Definir los patrones de búsqueda y reemplazo
    patterns = {"and": "&", "or": "|", "not": "~"}
    words = query.split()
    # Realizar el reemplazo iterativamente
    for i, word in enumerate(words):
        if (word in patterns):
            words[i] = patterns[word]

    processed_query = " ".join(words)

    # Convertir a expresión sympy y aplicar to_dnf
    query_expr = sympify(processed_query, evaluate=False)
    query_dnf = to_dnf(query_expr, simplify=True)

    return query_dnf


def get_document_with_term(term):
    documents = set()  # Inicializar un conjunto para almacenar los documentos

    # Iterar sobre los documentos y agregar el índice del documento al conjunto si el término está presente
    for i, doc in enumerate(tokenized_docs):
        if term in doc:
            documents.add(i)
    return documents


def get_matching_docs(query_dnf):
    global tokenized_docs,dictionary,corpus
    matching_documents = []
    conjunctive_docs = set([i for i, doc in enumerate(tokenized_docs)])
    conjunctive_sets = [set()] 

    str_query = str(query_dnf)
    conjunctive_terms = str_query.split('|')

    for conjunctive_term in conjunctive_terms:
        str_terms = str(conjunctive_term)
        terms = str_terms.split('&')
        terms = [term.strip() for term in terms]
        for term in terms:
            if term.startswith("~"):
                term_str = term_str[1:]
                term_docs = get_document_with_term(term)
                conjunctive_docs = conjunctive_docs.difference(term_docs)
            else:
                term_docs = get_document_with_term(term)  # Documentos que contienen el término
                conjunctive_docs = conjunctive_docs.intersection(term_docs)

        conjunctive_sets.append(conjunctive_docs)
        conjunctive_docs = set()
        
    return set().union(*conjunctive_sets)
    

query_dnf = query_to_dnf('experimental and house')
print(get_matching_docs(query_dnf))

vector_tfidf = []
vector_tfidf = vector_representation(tokenized_docs, dictionary, vector_tfidf, use_bow=False)
inverted_dict = {v: k for k, v in dictionary.items()}

def get_tfidf_term_doc(t_id,doc,isNot = False):
    for (term_id,tfidf_term) in doc:
        if(term_id == t_id and not isNot):
            return tfidf_term
        elif(term_id == t_id and isNot):
            return -tfidf_term
        
    return 0
    
def get_conjuntive_vector(conjuntive_term,doc):
    term_id = -1
    vetor_weight = []
    for term in conjuntive_term:

        if term in inverted_dict.keys():

            if term.startswith("~"):
                term_str = term_str[1:]
                term_id = inverted_dict[term_str]
                vetor_weight.append(get_tfidf_term_doc(term_id,doc,True))
            else:
                term_id = inverted_dict[term]
                vetor_weight.append(get_tfidf_term_doc(term_id,doc))
        else:
            vetor_weight.append(0) 
    return  vetor_weight
        
def calculate_conjunctive_similarity(w_vector, p):
    t = len(w_vector)
    sum_of_weights = sum([(1 - w) ** p for w in w_vector])
    similarity_score = 1 - ((1 / t) * sum_of_weights) ** (1 / p)
    return similarity_score

def calculate_disjunctive_similarity(w_vector, p = 2):
    t = len(w_vector)
    sum_of_weights = sum([w ** p for w in w_vector])
    similarity_score = ((1 / t) * sum_of_weights) ** (1 / p)
    return similarity_score

def get_weight_conjunctive_term(conjuntive_term,doc,p = 2):
    w_vector = get_conjuntive_vector(conjuntive_term,doc)
    return calculate_conjunctive_similarity(w_vector,p)

def get_weight_docs(query_dnf):
    global tokenized_docs,dictionary,corpus,vector_tfidf
    weight_doc = {}
    weight_conjunctive_terms = []

    str_query = str(query_dnf)
    conjunctive_terms = str_query.split('|')

    for i,weight_doc_term in enumerate(vector_tfidf):

        for conjunctive_term in conjunctive_terms:
            str_terms = str(conjunctive_term)
            terms = str_terms.split('&',)
            for termino in terms:
                termino.replace("(","")
                termino.replace(")","")
            terms = [term.strip() for term in terms]
            weight_conjunctive_terms.append(get_weight_conjunctive_term(terms,weight_doc_term))
            if(len(conjunctive_terms) > 1 ):
                weight_doc[i] = calculate_disjunctive_similarity(weight_conjunctive_terms)
            else:  
                weight_doc[i] = weight_conjunctive_terms[0]

        weight_conjunctive_terms = []

    weight_doc = dict(sorted(weight_doc.items(), key=lambda item: item[1], reverse=True))

    return weight_doc


query_dnf = query_to_dnf("( experimental and investigation ) or (vector and not water)")

titles = [doc.title for doc in dataset.docs_iter()]

def search_extended_boolean_model(query_dnf,n = 10):
    search_list = []
    relevace = get_weight_docs(query_dnf)
    for i in relevace.keys():
        search_list.append(titles[i])
    return search_list[:n]
    
print(search_extended_boolean_model(query_dnf,n = 10))
