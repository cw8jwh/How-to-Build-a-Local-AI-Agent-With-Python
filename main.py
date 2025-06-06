from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever # This is what looks for the functions defined in the vector.py file for the main part of the program to use.

model = OllamaLLM(model="llama3.2:3b")

# This section defines the chat template to be used.

template = """
You are an expert in answering questions about a pizza restaurant

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""

# This section connects the chat template to the LLM as a prompt.

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n------------------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")

    if question == "q":
        break

    else:
        reviews = retriever.invoke(question)
        result = chain.invoke({"reviews": reviews, "question": question})
        print(result)