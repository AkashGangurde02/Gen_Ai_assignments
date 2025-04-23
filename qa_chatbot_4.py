from transformers import pipeline

# Load pre-trained QA model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Context for QA
context = """Artificial Intelligence (AI) is a branch of computer science focused on creating intelligent machines 
that can perform tasks typically requiring human intelligence. It includes machine learning, deep learning, 
natural language processing, and robotics. AI is widely used in various industries such as healthcare, finance, 
automobiles, and customer service. It has the potential to revolutionize the way we work and interact with technology."""

print("Chatbot is ready! Type 'exit' to stop.")

# Chat loop
while True:
    question = input("\nAsk a question: ")
    if question.lower() == "exit":
        print("Goodbye!")
        break

    response = qa_pipeline(question=question, context=context)
    print("Answer:", response["answer"])
