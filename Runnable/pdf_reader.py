from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI

loader = TextLoader('docs.txt')
documents = loader.load()

text_spliter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
docs = text_spliter.split_documents(documents)

vectore_store = FAISS.from_documents(docs,GoogleGenerativeAIEmbeddings())

retriver = vectore_store.as_retriever()

query = "What are the key takeaways from the document?"

retrieved_docs = retriver.get_relevant_documents(query)

# Combine Retrieved Text into a Single Prompt
retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])


llm = GoogleGenerativeAI(model_name="gemini-3.5-flash", temperature=0.7)


prompt = f"""
Based on the following text, answer the question:

{retrieved_text}

Question: {query}
"""

answer = llm.predict(prompt)

print(answer) 