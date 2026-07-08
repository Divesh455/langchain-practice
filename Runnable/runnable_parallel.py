from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda ,RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template="""
Act as a senior social media marketer.

Create one viral tweet about {topic}.

Requirements:
- Catchy opening
- Valuable insight
- Friendly tone
- 2 emojis maximum
- 2 hashtags
- Under 250 characters
- Call to action
""",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="""
Act as a LinkedIn Top Voice.

Write one LinkedIn post about {topic}.

Structure:
1. Hook
2. Insight
3. Personal perspective
4. Call to action

Rules:
- 150 words
- Professional tone
- 5 hashtags
- No multiple versions
""",
    input_variables=["topic"]
)

model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash'
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "tweet":RunnableSequence(prompt1,model,parser),
        'linkdin':RunnableSequence(prompt2,model,parser)
    }
)

result = parallel_chain.invoke({"topic:'AI"})

print(result)