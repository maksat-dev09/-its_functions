from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            'Ты преподаватель {language}.'
            'Объясняй материал для уровня {level}.'
            'Всегда приводи пример кода',
        ),
        (
            'human',
            'Объясни тему: {topic}'
        )
    ]
)


model = ChatOllama(
    model='llama3.2:latest',
    temperature=0
)

chain = prompt | model

response = chain.invoke(
    {
        "language": "Python",
        'level': "средний",
        "topic": 'Функция'
    }
)


print(response.content)