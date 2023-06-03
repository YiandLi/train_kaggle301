import hnswlib


def build_index(embeddings, ids):
    index = hnswlib.Index(space="cosine", dim=embeddings.shape[-1])
    
    # Initializing index
    # max_elements - the maximum number of elements (capacity). Will throw an exception if exceeded
    # during insertion of an element.
    # The capacity can be increased by saving/loading the index, see below.
    #
    # ef_construction - controls index search speed/build speed tradeoff
    #
    # M - is tightly connected with internal dimensionality of the data. Strongly affects memory consumption (~M)
    # Higher M leads to higher accuracy/run_time at fixed ef/efConstruction
    index.init_index(max_elements=embeddings.shape[0], ef_construction=200, M=1000)
    
    # Controlling the recall by setting ef:
    # higher ef leads to better accuracy, but slower search
    index.set_ef(1000)
    
    # Set number of threads used during batch search/construction
    # By default using all available cores
    index.set_num_threads(16)
    
    index.add_items(embeddings, ids)
    
    return index