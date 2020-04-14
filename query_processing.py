NUMBER = 0
TITLE = 1
CONCEPTS = 2


def read_queries(filename):
        import xml.etree.ElementTree as ET
        tree = ET.parse(filename)
        root = tree.getroot()
        queries = []
        for topic in root:
            # Every topic
            number, title, question, narrative, concepts = topic
            query_id = number.text[-3:]
            title = title.text
            concepts = concepts.text.split("\n")[1].split("。")[0].split("、")
            queries.append((query_id, title, concepts))
        return queries


def sentences_to_terms(sentences, unigram=False):
    terms_text = []
    for sentence in sentences:
        # Rule: Bigrams and unigram
        if len(sentence) <= 2:
            terms_text.append(sentence)
        else:
            for i in range(len(sentence)-1):
                terms_text.append(sentence[i:i+2])
                if unigram:
                    terms_text.append(sentence[i:i+1])
            if unigram:
                terms_text.append(sentence[-1:])
    # TODO: modify this to add weight with occurrence?
    return list(set(terms_text))


def terms_text_to_terms_id(vocab, terms):
    terms_id = []
    for term in terms:
        # Translate text to id in vocab.all
        id_1 = vocab.index(term[0])
        if len(term) == 1:
            id_2 = -1
        else:
            id_2 = vocab.index(term[1])
        terms_id.append((id_1,id_2))
    return terms_id


def terms_id_to_inverted_file_index(inverted_file_terms, terms_id):
    inverted_indexes = []
    for term in terms_id:
        # Find inverted-file index of each term and remove the ones not in vocab.all
        try:
            index = inverted_file_terms.index(term)
            inverted_indexes.append(index)
        except ValueError:
            print(f'no term {term}')
            terms_id.remove(term)
    return inverted_indexes
