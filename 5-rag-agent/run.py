from rag_agent import RAGAgent


user_query = "Explain the changes of Global mean surface temperature based on AR6 report"

# step 1: initialize agent
data_path = "data/rag-agent-data"
rag_agent = RAGAgent()

# step 2: load & preprocess documents
docs = rag_agent.load_documents(data_path=data_path)
doc_chunks = rag_agent.preprocess_documents(docs)

# step 3: initialize vector store (Chroma DB)
db = rag_agent.setup_vector_store(chunks=doc_chunks)

# step 4: document retrieval & grading
retrieved_docs = rag_agent.retrieve_documents(db, user_query)
relevant_docs = rag_agent.grade_document_relevance(user_query, retrieved_docs)

# step 5: generate similar search queries & simulate search results
similar_queries = rag_agent.generate_similar_search_queries(user_query, relevant_docs)
simulated_results = rag_agent.simulate_search_results(db, similar_queries)

# step 6: results fusion
fused_results = rag_agent.reciprocal_rank_fusion(simulated_results)

# step 7: generate answer to user query
answer = rag_agent.generate_answer(fused_results, user_query)
print(answer)