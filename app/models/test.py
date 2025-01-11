from app.models.rag_model import generate_response

question = "Tell me what the notes are for South Australia"
possible_answer = ""
response = generate_response(question, possible_answer)
print(response)
