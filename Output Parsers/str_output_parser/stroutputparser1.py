from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "do_sample": True,
        "temperature": 0.5,
        "max_new_tokens": 300,
        "return_full_text": False
    }
)

model = ChatHuggingFace(llm=llm)

# Prompt 1
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# Prompt 2
template2 = PromptTemplate(
    template='Write a 5 line summary of the following text:\n{text}',
    input_variables=['text']
)


parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)