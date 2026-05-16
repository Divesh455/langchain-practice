from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from typing_extensions import TypedDict
import os

os.environ['HF_HOME'] = 'D:/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "do_sample": True,
        "temperature": 0.5,
        "max_new_tokens": 100,
        "return_full_text": False
    }
)

model = ChatHuggingFace(llm=llm)

class Review(TypedDict):
    summary: str
    sentiment: str

structured_model = model.with_structured_output(Review)

result = structured_model.invoke(
    "Samsung Galaxy is amazing with excellent battery and performance."
)

print(result)