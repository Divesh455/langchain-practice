from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder


Chat_template = ChatPromptTemplate([
    ('system','You are a helpful customer support agent.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history = []

with open('Chat_History.txt') as f:
    chat_history.extend(f.readlines())
    

prompt = Chat_template.invoke({'chat_history':chat_history,'query':'Where is my refund'})

print(prompt)