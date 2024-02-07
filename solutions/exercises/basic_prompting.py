from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
)

# # Probably not that good of a prompt - too vague
# prompt = "The capital of India is"
# print(prompt + llm(prompt))
# print()
# print()

# # A much better prompt, but it might give a long answer
# prompt = "The name of the capital city of India is"
# print(prompt + llm(prompt))
# print()
# print()

# Good prompts are very specific
prompt = "What is the name of the capital city of India, she asked. Please only respond with the city name and then stop talking. He answered: "
print(prompt + llm(prompt))
