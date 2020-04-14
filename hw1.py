import math
import numpy
import read_model

'''
Every thing we have to do
Rules are marked with 'Rule:' in the comments
'''

mode = 'train'

if __name__ == "__main__":
    '''
    import time
    
    start_time = time.time()

    # Vocabulary
    #   - Read vocab.all
    vocab = read_model.read_vocab("./model/vocab.all")
    print('Vocab:\n', vocab[:10])

    # Documents
    #   - Read file-list
    docs = read_model.read_docs("./model/file-list")
    print('Docs:\n', docs[:10])
     
    # Indexing (Done for you)
    #   - Read inverted-file (postings list)
    inverted_file_terms, inverted_file_postings = read_model.read_inverted_file("./model/inverted-file")
    print(inverted_file_terms[:10])
    print(inverted_file_postings[:10])

    print("seconds elaspsed: ", time.time() - start_time)

    # read_model.pickle_models(vocab, docs, (inverted_file_terms, inverted_file_postings), "./model/models.pickle")
    '''

    '''
    start_time = time.time()
    vocab, docs, inverted_file = read_model.load_models_from_pickle("./model/models.pickle")
    print(vocab[:10])
    print(docs[:10])
    print(inverted_file[:10])
    print("seconds elaspsed(Read Models): ", time.time()-start_time)
    '''

    # Calculate the Inverse Document Frequency
    # Inverse Document Frequency (IDF):
    #   log(N/df), df is the number of documents which the term occurs
    # TF-IDF: w = tf * idf

    
    def idf(df, N, base=10):
        return math.log(N/df, base)
    

    def tf_idf(tf, idf):
        return tf * idf
    

    doc_cnt = len(docs)

    '''
    The User Flow
    '''

    # Enter query
    import query_processing as qp
    queries = qp.read_queries(f"./queries/query-{mode}.xml")

    # Query Optimization

    # Searching
    #   - Fetch all postings list of the query indexes
    #   - Merge Postings lists
    #   - Create a "Postings ID"-"Index Term" Matrix
    #   - Use VSM to determine ranking
    #       - Rocchio Feedback?
    #       - Latent Sematic Indexing?

    # 1st Version: Index Purely From Concepts
    # 
    # Step 1: Fetch all relevant postings list
    # For all concepts
    #   Turn concepts into unigrams/bigrams
    #   Turn unigrams/bigrams from text into vocab_id
    #   Search the inverted-file for these vocab_id
    #       Record the doc_ids and their term-frequency
    # Step 2: Merge Postings lists and create a postings-term matrix
    #    
    #


    def concept_to_terms(concepts):
        terms_text = []
        for concept in concepts:
            # Rule: Bigrams only, unless concept is unigram
            if len(concept) <= 2:
                terms_text.append(concept)
            else:
                for i in range(len(concept)-1):
                    terms_text.append(concept[i:i+2])
        # Rule: Remove duplicate bigrams. TODO: modify this to add weight with occurrence?
        return set(terms_text)


    def terms_text_to_id(terms):
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
    
    
    def terms_id_to_inverted_index(terms):
        inverted_indexes = []
        for term in terms:
            # Find inverted-file index of each term and remove the ones not in vocab.all
            try:
                index = inverted_file_terms.index(term)
                inverted_indexes.append(index)
            except ValueError:
                print(f'no term {term}')
                terms.remove(term)
        return inverted_indexes


    def get_merged_postings_list(inverted_file_term_indexes):
        merged_postings = set()
        for i in inverted_file_term_indexes:
            for postings in inverted_file_postings[i]:
                merged_postings.add(postings[0])
        merged_postings = list(merged_postings)
        merged_postings.sort()
        return merged_postings
    

    def zeroed_matrix(i, j):
        import copy
        A = []
        row = [0] * j
        for _ in range(i):
            A.append(copy.deepcopy(row))
        return A


    def fill_matrix(mat, inverted_file_term_indexes, merged_postings, inverted_file_postings):
        # Rule: weight = tf * idf
        for i in range(len(terms_id)):
            # print(len(terms_id), len(inverted_file_term_indexes), len(inverted_file_postings))
            postings = inverted_file_postings[inverted_file_term_indexes[i]]
            for posting in postings:
                mat[i][merged_postings.index(posting[0])] = tf_idf(posting[1], idf(len(postings), doc_cnt, base=10))


    def write_rank(f, query_number, docs):
        f.write(f'{query_number[-3:]},')
        for doc in docs[:-1]:
            f.write(f'{doc.lower()} ')
        f.write(f'{docs[len(docs)-1].lower()}\n')


    def simple_query_vector(length):
        # Very Simple Search: Find the posting with the most aggregate term frequency
        # This is as if the query vector is unit in every dimension
        qv = []
        for _ in range(length):
            qv.append(1)
        return qv
    

    '''
    From Query to VSM
    - Read Queries from File
    - For Every Query Do...
        - Query Processing (Query -> Term)
        - Term to VSM
    '''
    # ------------------------------------------------------------
    # Read Queries
    f = open(f"./queries/my_ans_{mode}.csv", "w")
    f.write("query_id,retrieved_docs\n")
    # ------------------------------------------------------------
    # For every query do...
    for query in queries:
        # ------------------------------------------------------------
        # Query Processing
        
        terms_text = concept_to_terms(query[qp.CONCEPTS])
        # print(f'Terms (text)({len(terms_text)}): ', terms_text)
        
        # ------------------------------------------------------------
        # Term to VSM

        terms_id = terms_text_to_id(terms_text)
        terms_id.sort()
        print(f'Terms ID({len(terms_id)}): ', terms_id)
        for term in terms_id:
            id_1, id_2 = term
            if id_2 == -1:
                vocab_2 = ""
            else:
                vocab_2 = vocab[id_2]
            print(f"{vocab[id_1]}{vocab_2}, ", end='')
        print(" ")

        inverted_indexes = terms_id_to_inverted_index(terms_id)
        #print('Inverted File ID: ', inverted_indexes)
        
        merged_postings = get_merged_postings_list(inverted_indexes)
        #print("Length of Merged Postings: ", merged_postings)

        VS = zeroed_matrix(len(terms_id), len(merged_postings))
        print(f"VS Dimensions: {len(VS)}, {len(VS[0])}")

        fill_matrix(VS, inverted_indexes, merged_postings, inverted_file_postings)
        
        '''
        Searching

        - Define Query Vector qv
        - Rocchio Feedback to Improve qv
        - Calculate Similarity for each Docs
        - Rank
        '''
        # ------------------------------------------------------------
        # Define Query Vector

        # Uniform Unit Vector
        query_vec = simple_query_vector(len(terms_id))
        
        # Higher Term Weight on Title
        terms_title = concept_to_terms([query[qp.TITLE]])
        terms_title_id = terms_text_to_id(terms_title)
        extra_weight = 10
        for i in range(len(terms_id)):
            if terms_id[i] in terms_title_id:
                query_vec[i] += extra_weight
        print(query_vec)
        
        # Rocchio (only positive feedback) q' = alpha*q + beta/|D|*Sum(d) - gamma/|D|*Sum(d)
        iterations = 1
        alpha = 1
        beta = 10
        for _ in range(iterations):
            # Sum(d), Go through every column
            doc_vec = [0] * len(terms_id)
            for j in range(len(merged_postings)):
                for i in range(len(terms_id)):
                    # Rule: Add 1 if term exists
                    if VS[i][j] > 0:
                        doc_vec[i] += 1
            # q' = alpha*q + beta/|D|*Sum(d)
            reweight = beta/len(merged_postings)
            for i in range(len(terms_id)):
                query_vec[i] = alpha* query_vec[i] + doc_vec[i] * reweight
        print(query_vec)

        # ------------------------------------------------------------
        # Calculate Similarity

        # Dot Product
        postings_score = []
        for j in range(len(merged_postings)):
            score = 0
            for i in range(len(terms_id)):
                score += VS[i][j] * query_vec[i]
            postings_score.append(score)
        # print(postings_score)
        
        # ------------------------------------------------------------
        # Ranking

        # Get Rank
        ranking = []
        for i in range(len(merged_postings)):
            ranking.append((postings_score[i],docs[merged_postings[i]].split("/")[-1:][0]))
        
        
        # Sort ranking
        ranking.sort(reverse=True)
        print(ranking[:10])
        for i in range(len(ranking)):
            ranking[i] = ranking[i][1]
        write_rank(f, query[qp.NUMBER], ranking[:10])

        # break # Just for faster testing
    
    f.close()


    # Returns ranking