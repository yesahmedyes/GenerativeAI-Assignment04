from dotenv import load_dotenv
import os

from pinecone.grpc import PineconeGRPC as Pinecone
from sentence_transformers import SentenceTransformer

from langchain_groq import ChatGroq
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel,
)

from langchain_community.tools import TavilySearchResults

from langgraph.graph import START, END, StateGraph
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

import gradio as gr


load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index = pc.Index("itu")


model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

router_prompt = """Given a user question, classify it into one of these categories:
- FINANCIAL: Questions about NetSol's financial information, revenue, profits, etc.
- CURRENT: Questions about current events, real-time information, or something that may have happened or changed recently
- GENERAL: Anything that does not fall into the other categories

Output only one category: FINANCIAL, CURRENT, or GENERAL

Question: {question}
Category:"""


def route_query(state):
    chat_model = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0,
        max_tokens=8,
    )

    question = state["messages"][0].content

    formatted_prompt = router_prompt.format(question=question)

    category = chat_model.invoke(formatted_prompt)

    return {"category": category.content}


def get_context(query: str, k: int = 3) -> str:
    matches = index.query(vector=model.encode(query), top_k=k, include_metadata=True)

    context_chunks = [match["metadata"]["text"] for match in matches["matches"]]

    context = ""

    for counter, chunk in enumerate(context_chunks):
        context += f"## Chunk {counter}:\n\n{chunk}\n\n"

    return context


retrieval_chain = RunnableParallel(
    {"context": RunnableLambda(get_context), "query": RunnablePassthrough()}
)


def retrieve_documents(state):
    question = state["messages"][-1].content

    context = retrieval_chain.invoke(question)

    return {"context": context}


grade_prompt = """You are trying analyze if the provided documents are relevant to the user question or not. If they are releavant you will output generate, if they are not you will output rewrite.

Output only one word: GENERATE or REWRITE

Provided documents:
{context}                          

Question:
{question}
"""


def grade_documents(state):
    chat_model = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0,
        max_tokens=8,
    )

    question = state["messages"][-1].content
    context = state["context"]

    formatted_prompt = grade_prompt.format(context=context, question=question)

    response = chat_model.invoke(formatted_prompt)

    if response.content == "REWRITE":
        return {"context": None}
    else:
        return {"context": context}


rewrite_prompt = """Look at the input and try to reason about the underlying semantic intent / meaning.

Here is the initial question: {question} 

I ONLY WANT THE REWRITTEN QUESTION, NO OTHER TEXT."""


def rewrite_query(state):
    chat_model = ChatGroq(
        model_name="llama3-8b-8192",
        max_tokens=4096,
    )

    question = state["messages"][-1].content

    formatted_prompt = rewrite_prompt.format(question=question)

    response = chat_model.invoke(formatted_prompt)

    return {"messages": [response.content]}


def handle_financial(state):
    chat_model = ChatGroq(
        model_name="llama3-8b-8192",
        max_tokens=4096,
    )

    question = state["messages"][-1].content

    context = state["context"]

    if context is None:
        return {"messages": ["No context found"]}

    prompt = f"""You are a financial expert assistant. Answer the following question using the provided context.
    If you cannot find the answer in the context, say so.
    
    Question: {question}
    Context: {context}
    Answer:"""

    response = chat_model.invoke(prompt)

    return {"messages": [response.content]}


tavily = TavilySearchResults()


def handle_current(state):
    chat_model = ChatGroq(
        model_name="llama3-8b-8192",
        max_tokens=4096,
    )

    question = state["messages"][-1].content

    search_results = tavily.invoke(question)

    context = ""

    for counter, chunk in enumerate(search_results):
        context += f"## Chunk {counter}:\n\n{chunk}\n\n"

    prompt = f"""You are a helpful assistant. Answer the following question using the search results.
    Base your answer only on the provided search results.
    
    Question: {question}
    Search Results: {context}
    Answer:"""

    response = chat_model.invoke(prompt)

    return {"messages": [response.content], "context": context}


def handle_general(state):
    chat_model = ChatGroq(
        model_name="llama3-8b-8192",
        max_tokens=4096,
    )

    question = state["messages"][-1].content

    prompt = f"""You are an expert on every topic. Answer the following question.
    Question: {question}
    Answer:"""

    response = chat_model.invoke(prompt)

    return {"messages": [response.content]}


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    category: str | None
    context: str | None


workflow = StateGraph(AgentState)

workflow.add_node("router", route_query)
workflow.add_node("financial", handle_financial)
workflow.add_node("current", handle_current)
workflow.add_node("general", handle_general)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("rewrite", rewrite_query)
workflow.add_node("grade_documents", grade_documents)

workflow.add_edge(START, "router")

workflow.add_conditional_edges(
    "router",
    lambda x: x["category"],
    {
        "FINANCIAL": "retrieve",
        "CURRENT": "current",
        "GENERAL": "general",
    },
)

workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    lambda x: "REWRITE" if x["context"] is None else "FINANCIAL",
    {
        "REWRITE": "rewrite",
        "FINANCIAL": "financial",
    },
)

workflow.add_edge("rewrite", "router")

workflow.add_edge("financial", END)
workflow.add_edge("current", END)
workflow.add_edge("general", END)

chain = workflow.compile()


def process_query(question: str) -> str:
    inputs = {
        "messages": [
            ("user", question),
        ]
    }

    response = chain.invoke(inputs)

    return response


def chat_interface(message):
    response = process_query(message)

    category = response["category"]
    answer = response["messages"][-1].content

    return f"Category: {category}\n\nAnswer: {answer}"


demo = gr.Interface(
    fn=chat_interface,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
    outputs=gr.Textbox(lines=10, label="Response"),
    title="NetSol Q&A System",
    description="Ask questions about NetSol's financial information, current events, or general topics.",
    examples=[
        ["What was NetSol's revenue in 2024?"],
        ["What are the current market trends in the automotive industry?"],
        ["Tell me about NetSol's products and services."],
    ],
)

demo.launch(share=True)
