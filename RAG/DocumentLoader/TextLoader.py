from langchain_community.document_loaders import TextLoader

loader  = TextLoader('cricket.txt',encoding='utf-8')

doc = loader.load()

# print(doc)

print(len(doc))

print(doc[0])