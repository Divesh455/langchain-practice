from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

loader = DirectoryLoader(
    path='../DocumentLoader',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.lazy_load()

spliter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

result = spliter.split_documents(docs)

print(result[0].page_content)