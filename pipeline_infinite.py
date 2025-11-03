import os
import json
from typing import List, TypedDict
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_ollama import OllamaLLM
# from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
# OLD: from langchain_community.vectorstores import Chroma  # Or similar community/core import
from langchain_chroma import Chroma  # <--- NEW DEDICATED PACKAGE
import uuid
# --- CONFIG ---
os.environ["OPENAI_API_KEY"] = ""
# Remove the old os.environ line and replace it with this constant:
DATA_FOLDER = "data"
CHROMA_PATH = "chroma_db"
CHECKPOINT_DB = "infinite_checkpoints.db"
#OLLAMA_MODEL = "llama3"

embeddings = OpenAIEmbeddings()
db = Chroma(collection_name="moengage", embedding_function=embeddings, persist_directory=CHROMA_PATH)
def ingest_data():
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".txt"):
            path = os.path.join(DATA_FOLDER, file)
            docs = TextLoader(path).load()
            chunks = splitter.split_documents(docs)
            db.add_documents(chunks)
# if db._collection.count() == 0:
#     ingest_data()


# local_llm = OllamaLLM(model=OLLAMA_MODEL)
cloud_llm = ChatOpenAI(model="gpt-4o-mini")

def ask_llm(prompt):
    return cloud_llm.invoke(prompt).content

## will code it later with different llms to perform different tasks
#def ask_llm(prompt): return local_llm.invoke(prompt).content if len(prompt.split()) < 15 else cloud_llm.invoke(prompt).content

#checkpointer = SqliteSaver.from_conn_string(CHECKPOINT_DB)
#checkpointer = SqliteSaver.create(CHECKPOINT_DB)

class State(TypedDict):
    messages: List[dict]
    context: str
    proposal: str
    need_feedback: bool

def retrieve(state: State) -> State:
    if state["messages"]:
        query = state["messages"][-1]["content"]
        docs = db.similarity_search(query, k=3)
        state["context"] = "\n".join([d.page_content for d in docs])
    return state

def generate(state: State) -> State:
    if state["messages"]:
        prompt = f"Conversation:\n{json.dumps(state['messages'][-5:])}\nContext: {state['context']}\nRespond helpfully."
        state["proposal"] = ask_llm(prompt)
        state["need_feedback"] = True
    return state

def hitl(state: State) -> State:
    if state["need_feedback"]:
        print("\nAGENT:", state["proposal"])
        user = input("You: ").strip()
        if user:
            state["messages"].append({"role": "assistant", "content": state["proposal"]})
            state["messages"].append({"role": "user", "content": user})
        state["need_feedback"] = False
    return state

graph = StateGraph(State)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_node("hitl", hitl)
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "hitl")
graph.add_edge("hitl", "retrieve")  # ← LOOP FOREVER
graph.set_entry_point("retrieve")
#app = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    print(db._collection.count())
    if db._collection.count() == 0:
        print("Ingest")
        ingest_data()
    print("Ingested")
    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        app = graph.compile(checkpointer=checkpointer)
        thread_id = input("Thread ID (or Enter for new): ").strip() or str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        print(f"Thread: {thread_id}\nType 'exit' to stop.\n")
        state = app.get_state(config)
        messages = state.values.get("messages", []) if state else []
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == "exit":
                break
            messages.append({"role": "user", "content": user_input})
            inputs = {"messages": messages, "context": "", "proposal": "", "need_feedback": False}
            result = app.invoke(inputs, config=config)
            messages = result["messages"]