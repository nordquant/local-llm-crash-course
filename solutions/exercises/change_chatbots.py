from typing import List

import chainlit as cl
from ctransformers import AutoModelForCausalLM


def get_prompt_orca(instruction: str, history: List[str] = None) -> str:
    system = "You are an AI assistant that gives helpful answers. You answer the question in a short and concise way."
    prompt = f"### System:\n{system}\n\n### User:\n"
    if history is not None:
        prompt += f"This is the conversation history: {''.join(history)}. Now answer the question: "
    prompt += f"{instruction}\n\n### Response:\n"
    print(f"Prompt created: {prompt}")
    return prompt


def get_prompt_llama2(instruction: str, history: List[str] = None) -> str:
    system = "You are an AI assistant that gives helpful answers. You answer the question in a short and concise way."
    prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n"
    if history is not None:
        prompt += f"This is the conversation history: {''.join(history)}. Now answer the question: "
    prompt += f"{instruction} [/INST]"
    print(f"Prompt created: {prompt}")
    return prompt


def select_llm(llm_name: str):
    global llm, get_prompt
    if llm_name == "llama2":
        llm = AutoModelForCausalLM.from_pretrained(
            "TheBloke/Llama-2-7b-Chat-GGUF", model_file="llama-2-7b-chat.Q5_K_M.gguf"
        )
        get_prompt = get_prompt_llama2
        return "Model changed to Llama"
    elif llm_name == "orca":
        llm = AutoModelForCausalLM.from_pretrained(
            "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
        )
        get_prompt = get_prompt_orca
        return "Model changed to Orca"
    else:
        return "Model not found, keeping old model"


@cl.on_message
async def on_message(message: cl.Message):
    if message.content.lower() in ["use llama2", "use orca"]:
        model_name = message.content.lower().split()[1]
        response = select_llm(model_name)
        await cl.Message(response).send()
        return
    if message.content.lower() == "forget everything":
        cl.user_session.set("message_history", [])
        await cl.Message("Uh oh, I've just forgotten our conversation history").send()
        return

    message_history = cl.user_session.get("message_history")
    msg = cl.Message(content="")
    await msg.send()

    prompt = get_prompt(message.content, message_history)
    answer = ""
    for word in llm(prompt, stream=True):
        await msg.stream_token(word)
        answer += word
    message_history.append(answer)
    await msg.update()


@cl.on_chat_start
async def on_chat_start():
    await cl.Message("Loading model Orca...").send()
    select_llm("orca")
    cl.user_session.set("message_history", [])

    await cl.Message("Model initialized. How can I help you?").send()
