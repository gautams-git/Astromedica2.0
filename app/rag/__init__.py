"""Reserved RAG chunk store.

The current pipeline does not rank vectors. `db_layer.query_similar_chunks`
still returns an empty list on purpose. Chunk metadata can be stored in
`kb_chunks/{chunkId}` without adding a vector database.
"""
