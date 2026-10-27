# Exercise 4.20 Transformers Question Answering with a Different Text (Updated for Transformers v5+)
from transformers import pipeline

# Load the text-generation pipeline with a chat model
question_answerer = pipeline('text-generation', model='Qwen/Qwen2.5-0.5B-Instruct')

context = (
    'Politeknik Elektronika Negeri Surabaya (PENS) is a vocational higher education '
    'institution located in Surabaya, East Java, Indonesia. It was founded in 1988 with '
    'support from the Japan International Cooperation Agency. PENS offers programs in '
    'electronics, informatics, telecommunications and creative multimedia technology.'
)

questions = [
    'Where is PENS located?',
    'When was PENS founded?',
    'Who supported the founding of PENS?',
    'What programs does PENS offer?',
]

for question in questions:
    messages = [
        {
            'role': 'user',
            'content': (
                f'Context: {context}\n\nQuestion: {question}\n'
                'Provide a short, direct answer based only on the context.'
            ),
        }
    ]
    
    response = question_answerer(messages, max_new_tokens=50)
    answer = response[0]['generated_text'][-1]['content']
    
    print("Q: %s\nA: %s\n" % (question, answer.strip()))