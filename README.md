# ai-governance-agent
An AI agent that helps policy analysts research and compare AI regulations across jurisdictions using retrieval augmented generation and a ReAct reasoning pattern, grounded in the NIST AI RMF, NIST AI 600-1, and the EU AI Act


## Agent

The embedding model used at query time must match what the index was built with.

Notebook 01 used databricks-bge-large-en. 

use query_text= in your similarity_search call, not query_vector=, and we can drop the SentenceTransformer embedder from notebook 03 entirely. 
