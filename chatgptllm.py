from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2-medium"  
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Define a function to generate response
def generate_response(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    response_ids = model.generate(input_ids, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response_text

# Example conversation
conversation_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("ChatGPT: Goodbye!")
        break
    conversation_history.append("You: " + user_input)
    
    # Generate response based on conversation history
    chatgpt_input = " ".join(conversation_history)
    response = generate_response(chatgpt_input)
    
    print("ChatGPT:", response)
    conversation_history.append("ChatGPT: " + response)
