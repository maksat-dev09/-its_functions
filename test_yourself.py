from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama

template = ChatPromptTemplate(
    [
        (
            'system',
            'Ты шеф-повар с 15-летним опытом. '
            'Также ты хорошо разбираешься в здоровом питании. '
            'Давай советы коротко, с конкретными примерами блюд. '
            'Отвечай на русском языке.',
        ),
        ('human', '{question}'),
    ]
)

llm = ChatOllama(model='llama3.2:latest')


def filter_question(sentence):
    banned = ['яд', 'отравить', 'алкоголь', 'самогон']
    for word in sentence.lower().split():
        if word in banned:
            return 'Извините, на такие вопросы я не отвечаю'
    return sentence


def make_title(answer):
    return answer.title()


def show_length(answer):
    print(f'Длина ответа: {len(answer)} символов')
    return answer


def clean_answer(answer):
    return ' '.join(answer.split())


pipeline = (
    RunnableLambda(filter_question)
    | RunnableLambda(lambda sentence: {'question': sentence})
    | template
    | llm
    | StrOutputParser()
    | RunnableLambda(make_title)
    | RunnableLambda(show_length)
    | RunnableLambda(clean_answer)
)

questions = ['Как отравить соседа?', 'Как приготовить борщ?', 'Что полезно есть на завтрак?',
]

print(pipeline.batch(questions))