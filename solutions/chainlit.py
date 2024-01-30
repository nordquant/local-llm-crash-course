from typing import List

from ctransformers import AutoModelForCausalLM

import chainlit as cl


def get_prompt(instruction: str, history: List[str]) -> str:
    system = "You are an AI assistant that gives helpful answers. You only answer the question in a short and concise way."
    prompt = f"### System:\n{system}\n\n### User: "
    if len(history) > 0:
        prompt += f"Use the following pieces of context to answer the question at the end: {''.join(history)}. This is the end of the context. Now answer this question. Give a short answer:"
    prompt += f"{instruction}\n\n### Response:\n"
    print(prompt)
    return prompt


@cl.on_message
async def main(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    msg = cl.Message(content="")
    await msg.send()

    prompt = get_prompt(message.content, message_history)
    message_history.append("Question: " + message.content + ". Answer: ")
    for word in llm(prompt, stream=True):
        await msg.stream_token(word)
        message_history.append(word)
    await msg.update()


@cl.on_chat_start
async def on_chat_start():
    global llm
    cl.user_session.set("message_history", [])

    await cl.Message("Initializing Model...").send()
    llm = await loading_model()
    await cl.Message("Model initialized. How can I help you?").send()


@cl.step
async def loading_model():
    return AutoModelForCausalLM.from_pretrained(
        "juanjgit/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
    )
