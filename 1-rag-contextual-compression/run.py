from rag_cc import RAGContextualCompression

user_query = " Where does the water present in the egg go after boiling the egg?"

# step 1: initialize agent
data_path = "data/rag-con-comp-data"
ragcc = RAGContextualCompression(data_path=data_path)

# step 2: load & preprocess documents
docs = ragcc.load_documents()
doc_chunks = ragcc.preprocess_documents(docs)

# step 3: initialize vector store
db = ragcc.setup_vector_store(doc_chunks)

# step 4: retrieve documents
retrieved_docs = ragcc.retrieve_documents(db, user_query)

# step 5: setup compression and redundancy filters to optimize document retrieval
contextual_comp_retriever = ragcc.setup_compression_pipeline_retriever(db)

# step 6: generate the final answer to the user query
answer = ragcc.generate_answer(retriever=contextual_comp_retriever, user_query=user_query)
print(answer)
