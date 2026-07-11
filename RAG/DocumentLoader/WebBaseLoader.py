from langchain_community.document_loaders import WebBaseLoader

url = "https://www.flipkart.com/a/p/itm1692bd8b2fe84?pid=COMHHPK3GMHHGFDD&param=8666&BU=CoreElectronics&pageUID=1783574031663"

loader = WebBaseLoader(url)

doc = loader.load()

print(doc[0].page_content)