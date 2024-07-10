# Retrieval Augmented Generation (RAG)

This repository contains detailed documentation, example notebooks (as well as tips & tricks) for implementing Retrieval Augmented Generation (RAG) systems for various types of data using Python.

## Notebooks Overview

### 0. [Tips & Tricks](0-tips-n-tricks)
Tips & tricks on data processing, cleaning, chunking, and augmentation. It also highlights various approaches, accompanied by code snippets, for evaluating Retrieval-Augmented Generation (RAG) systems. Explore all the effective techniques and methods designed to enhance the accuracy of a RAG system.

**What you can find inside?**
- **Table Processing**: Techniques for processing of complex tables from the PDF documents.
- **RAG Evaluation**: Approaches for evaluation of precision, recall, faithfulness, etc. of your RAG system.
- **Query Extension**: Methodologies for enhanced retrieval (e.g., re-ranking, query extensions, etc.)

### 1. [RAG Contextual Compression](1-rag-contextual-compression)
Methods to compress context for RAG systems to improve performance and accuracy and minimize costs.

**What you can find inside?**
- **Context Summarization**: Techniques to provide only relevant information to retrievers.
- **Embedding Compression**: Reducing the size of embeddings while maintaining information.
- **Cost Optimization**: How to make more with LLMs for less money?

### 2. [RAG for Semi-Structured Data](2-rag-semi-structured)
Implementation of a RAG system for semi-structured data.

**What you can find inside?**
- **Data Retrieval and Processing**: Fetching and processing data from MongoDB.
- **Querying and Chain Integration**: Building end-to-end query-response systems with enhanced retrieval capabilities for semi-structured data (e.g., JSON).

### 3. [RAG for Structured Data](3-rag-structured)
Exploration & implementation of a RAG system for structured data with SQL databases.

**What you can find inside?**
- **SQL Database Interaction**: Configuring connections and interacting with SQL databases using pyodbc and SQLAlchemy.
- **Agent Creation**: Setting up SQL agents for intelligent database interaction & querying.
- **Query Enhancements**: Some ideas (not implemented yet) for information filtering & retrieval.

### 4. [RAG Agent](4-rag-agent)
Detailed implementation and usage of autonomous RAG agents for different retrieval tasks.

**What you can find inside?**
- **Autonomous Agents**: Developing agents that perform retrieval tasks autonomously & efficiently.
- **Integration**: Combining agents with other system components for seamless operation.

### 5. [RAG Routing](5-rag-routing)
Techniques and examples on how to route queries effectively in RAG systems to ensure accurate and relevant information retrieval depending on user's intent.

**What you can find inside?**
- **Intent Detection**: Methods to accurately determine user intent from queries.
- **Query Routing**: Techniques for directing queries to the appropriate retrieval systems.
- **Response Optimization**: Enhancing the relevance and accuracy of generated responses.

### 6. [Corrective RAG](6-rag-corrective)
Techniques and examples on how to use corrective RAG (CRAG) to ensure the pipeline adheres strictly to the information it retrieves.

**What you can find inside?**
- **Fact Verification**: Methods to cross-verify retrieved information with trusted sources.
- **Correction Mechanisms**: Techniques for identifying and correcting inaccuracies in real-time.

### 7. [GraphRAG](7-graphrag)
Implementation & testing of Microsoft's GraphRAG pipeline with global & local searches using GPT-4o.

**What you can find inside?**
- **Graph-Based Retrieval**: Using graph structures to enhance the retrieval process by connecting related pieces of information.
- **Hierarchical Community Structure**: Building hierarchical communities within the graph for better context management and query responses using [Leiden algo](https://arxiv.org/pdf/1810.08473).
- **Local & Global Searches**: Utilizing local searches for detailed, specific queries within smaller subgraphs, and global searches to connect broader themes and insights across the entire knowledge graph.

## Data
Contains sample datasets and scripts for data preprocessing used in the example notebooks.

## Images
Includes images and visual aids used in the documentation and notebooks.

## Setup and Configuration

Before running the notebooks, ensure that you have the necessary environment variables configured as described in the notebooks. Don't forget to install packages from `requirements.txt`.

## Work with Me

If you are interested in:

- Developing custom, production-ready RAG pipelines
- Creating unique chatbot & assistant solutions
- Integrating AI into various aspects of your business operations
- Building prototypes or proofs of concept to validate your next groundbreaking idea

🚩 Then go visit my [services page](https://iamvladyashin.com/services) and witness the magic happen.

Feel free to reach out and share your story with me! I'm just an email away.

## Contributions and Issues

Feel free to fork this repository, contribute changes, or submit issues.

## Looking for the Ideal Hosting Solution for Your Project?

Give [DigitalOcean](https://m.do.co/c/d05f08187ba9) a try! I host all my projects, including RAG pipelines, bots, assistants, and this website, on DigitalOcean.

In addition to receiving __**$200 in credit for over 60 days**__, you'll also enjoy an amazing development and deployment experience!

I highly recommend it! 👍

Peace!
