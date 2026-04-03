from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

text = 'Delhi is the capital of India'

vector = embedding.embed_query(text)
print(str(vector))



# Docs Embeddings

document = [
    "Delhi is capital of India",
    "Rose is Beautiful"
]

result = embedding.embed_documents(document)
print(str(result))