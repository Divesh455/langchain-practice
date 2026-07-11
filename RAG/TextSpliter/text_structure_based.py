from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
    The quick brown fox jumps over the lazy dog. Gathering data for software applications requires precise string formats. Most development projects rely on structured dummy data to test user interfaces before launch. 
    
    Ensuring proper alignment and font scaling saves significant time during development. Every sentence should flow naturally to mimic real user input.
"""
spliter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0
)

chunks = spliter.split_text(text)

print(len(chunks))
print(chunks)