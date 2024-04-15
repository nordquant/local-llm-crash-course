from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

# Load quantized Llama2
llm = CTransformers(
    model="TheBloke/Llama-2-7B-Chat-GGUF",
    model_file="llama-2-7b-chat.Q2_K.gguf",
    model_type="llama2",
    config={
        "max_new_tokens": 2000,
        "temperature": 0.01,
        "context_length": 2000,
        "threads": 8,
        "gpu_layers": 1,
    },
)

# Set up a prompt template
template = """
[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always provide a concise answer and use the Context provided in the question. If you don't know the answer, you can say so.
<</SYS>>
Context:
{context}
Question:
{question}[/INST]"""

# Load documents into a Chroma vector store

prompt_template = PromptTemplate(template=template, input_variables=["text"])
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
report_text_pages = PyPDFLoader("gr-3.pdf").load_and_split()
db = Chroma.from_documents(report_text_pages, embeddings)


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


# Retrieve relevant pages and convert them into plain text
retriever = db.as_retriever()
question = "What is the type of the battery in the GR III camera?"
context = format_docs(retriever.invoke(question))

# Format prompt template, print it and execute query

prompt = prompt_template.format(context=context, question=question)
print(f"PROMPT:\n{prompt}\n\nANSWER:")

for chunk in llm.stream(prompt):
    print(chunk, end="", flush=True)
