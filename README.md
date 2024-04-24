# Retrieval Augmented Generation (RAG)

This repository contains detailed documentation and example notebooks for implementing Retrieval Augmented Generation systems for both different types of data data using Python.

## Notebooks Overview

### 1. [RAG for Semi-Structured Data](rag-semi-structured-data.ipynb)

This notebook demonstrates the implementation of a RAG system for semi-structured data.

📑 **Dataset**: [Medical Equipment Spare Parts Inventories](https://www.kaggle.com/datasets/mohdkhidir/medical-equipment-spare-parts-inventories-datasets)

- **Data Retrieval and Processing**: Fetching inventory data from MongoDB and processing it into a format suitable for embedding and retrieval.
- **Embedding and Vector Store**: Creating and using embeddings for the inventory data using SentenceTransformer and managing data with Chroma vector store.
- **Querying**: Implementing classical search and vector-based retrieval to respond to user queries about inventory items.
- **Chain Integration**: Building an end-to-end query-response chain that combines retriever, prompt template, and language model to generate responses.
- **Self-Querying Retriever**: Enhancing retrieval capabilities by enabling the system to parse and respond based on structured queries derived from user input.


### 2. [RAG for Structured Data](rag-structured-data.ipynb)

This notebook explores the use of a RAG system for structured data with SQL databases.

📑 **Dataset**: [Medical Equipment Spare Parts Inventories](https://www.kaggle.com/datasets/mohdkhidir/medical-equipment-spare-parts-inventories-datasets)

- **SQL Database Interaction**: Configuring connections to Azure SQL Database using pyodbc and SQLAlchemy.
- **Agent Creation**: Setting up an agent with SQL capabilities to interact intelligently with the database based on user queries.
- **Translation and Querying**: Utilizing Azure OpenAI for translating text and executing SQL queries through a zero-shot description-based SQL agent.
- **Query Enhancements**: Improving the quality of system prompts to enhance the precision and relevance of SQL query results.

## Setup and Configuration

Before running the notebooks, ensure that you have the necessary environment variables configured as described in the notebooks. Don't forget to install packages from `requirements.txt`

## Contributions and Issues

Feel free to fork this repository, contribute changes, or submit issues.

Peace!

