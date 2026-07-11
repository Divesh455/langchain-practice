from langchain_community.document_loaders import CSVLoader

loader = CSVLoader('')

doc = loader.load()

print(doc)