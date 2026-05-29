from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel ,Field

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

class Person(BaseModel):
    
    name:str = Field(description='Name of the person')
    age:int = Field(gt=18,description='Age of the person')
    city:str = Field(description='Name of the city the person belongs to')
    
parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

final_result = chain.invoke({'place':'indian'})

print(final_result)