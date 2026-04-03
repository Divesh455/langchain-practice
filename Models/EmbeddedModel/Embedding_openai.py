from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-small',dimensions=32)

result = embedding.embed_query("Delhi is the capital of India")

print(str(result))



# Docs Embeddings

document = [
    "Delhi is capital of India",
    "Rose is Beautiful"
]

result = embedding.embed_documents(document)
print(str(result))