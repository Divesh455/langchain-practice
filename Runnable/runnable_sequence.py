from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda ,RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template="Tell me exactly one one-line joke about {topic}. Return only the joke.",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="""
Explain the following joke.

Rules:
- Maximum 3 sentences.
- Use simple English.
- Do not repeat the joke.
- Explain the hidden meaning.

Joke:
{text}
""",
    input_variables=["text"]
)

model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash'
)

parser = StrOutputParser()

chain = RunnableSequence(prompt1,model,parser,prompt2,model,parser)

result = chain.invoke({"topic":"AI"})

print(result)