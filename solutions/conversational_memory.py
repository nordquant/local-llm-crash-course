from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
)


def get_prompt(instruction: str, history: list[str] | None = None) -> str:
    system = "You are an AI assistant that gives helpful answers. You answer the question in a short and concise way."
    prompt = f"### System:\n{system}\n\n### User:\n"
    if history is not None:
        prompt += f"This is the conversation history: {''.join(history)}. Now answer the question: "
    prompt += f"{instruction}\n\n### Response:\n"
    print(f"Prompt created: {prompt}")
    return prompt


history = []

question = "Which city is the capital of India?"
prompt = get_prompt(question)
answer = ""
for word in llm(prompt, stream=True):
    print(word, end="", flush=True)
    answer += word
print()
history.append(answer)

question = "And of the United States?"
prompt = get_prompt(question, history)
for word in llm(prompt, stream=True):
    print(word, end="", flush=True)
print()
