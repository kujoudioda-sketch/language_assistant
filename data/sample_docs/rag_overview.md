# Retrieval-Augmented Generation Overview

Retrieval-Augmented Generation, or RAG, combines information retrieval with language generation. A RAG system first retrieves relevant source passages, then asks a language model to answer using those passages as evidence.

RAG can reduce hallucination because the model is instructed to ground claims in retrieved source text. The answer should cite the passages that support each factual statement so that users can verify the response.

For academic documents, common implementation steps include parsing papers, splitting long text into overlapping chunks, embedding each chunk, indexing vectors, retrieving top-k chunks for a query, and generating a citation-grounded answer.

Retrieval quality can be evaluated with Recall@k and Mean Reciprocal Rank. Generation quality can be evaluated with faithfulness checks that compare the answer against retrieved evidence.
