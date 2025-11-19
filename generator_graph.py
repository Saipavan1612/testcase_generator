from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import Annotated, Sequence
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from pymongo import MongoClient
import operator
import pandas as pd
from tabulate import tabulate
import re
from datetime import datetime
from typing import Union
import os
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    clarifications_asked: bool  # Add flag to control flow


api_key = os.getenv("GEMINI_API_KEY")

print(f"API_KEY: {api_key}")

llm = init_chat_model("gemini-2.5-flash", api_key=api_key, model_provider="google_genai")

mongo_connection = os.getenv("MONGODB_CONNECTION")

print(f"MONGODB_CONNECTION: {mongo_connection}")

try:
    client = MongoClient(mongo_connection, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    print("MongoDB connection successful.")
except Exception as e:
    print("MongoDB connection failed")
    raise Exception(e)

db = client["testcase_generator"]
collection = db["testcase_generator_collection"]

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)

mongo_vector_store = MongoDBAtlasVectorSearch(
    embedding=embeddings,
    collection=collection,
    index_name="testcase_index",
    relevance_score_fn="cosine",
)

def ask_clarifications_node(state: AgentState) -> dict:
    jira_ticket_description = " ".join(
        msg.content for msg in state["messages"] if isinstance(msg, HumanMessage)
    )

    try:
        retrieved_docs = mongo_vector_store.similarity_search(
            jira_ticket_description, k=5
        )
        retrieved_context = "\n".join(doc.page_content for doc in retrieved_docs) if retrieved_docs else ""
    except Exception as e:
        print(f"RAG retrieval failed: {e}. Asking clarifications without context.")
        retrieved_context = ""

    base_prompt = f"""
    You are a QA Test Case Generator.

    RULES:
    1. Always ask clarification questions first.
    2. Ask at least 3-5 detailed questions.
    3. DO NOT generate test cases yet.
    4. Focus on:
       - User workflows and scenarios
       - Edge cases and boundary conditions
       - Expected behavior vs error handling
       - Data validation requirements
       - Security and performance considerations
       - Integration points and dependencies

    ### JIRA Ticket:
    {jira_ticket_description}
    """

    if retrieved_context:
        prompt = f"{base_prompt}\n\n### Retrieved Context from Memory:\n{retrieved_context}\n\nGenerate specific clarification questions based on the ticket and context."
    else:
        prompt = f"{base_prompt}\n\nGenerate clarification questions to understand requirements better."

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content='Generate detailed clarification questions based on the JIRA ticket.')
    ])

    return {
        "messages": [HumanMessage(content=response.content)],
        "clarifications_asked": True  # Mark that clarifications have been asked
    }

def generate_testcases_node(state: AgentState) -> dict:
    try:
        retrieved_docs = mongo_vector_store.similarity_search(
            " ".join(msg.content for msg in state["messages"]), k=5
        )
        retrieved_context = "\n".join(doc.page_content for doc in retrieved_docs) if retrieved_docs else ""
    except Exception as e:
        print(f"RAG retrieval failed: {e}. Proceeding without context.")
        retrieved_context = ""

    messages = state["messages"]
    jira_ticket = messages[0].content if messages else ""

    clarification_history = "\n".join(
        f"Message {i + 1}: {msg.content}"
        for i, msg in enumerate(messages[1:], 1)
    )

    base_prompt = """
    You are a QA Test Case Generator.

    ### OUTPUT FORMAT
    Produce a Markdown table with these columns:

    | Test Case ID | Title | Type | Preconditions | Steps | Expected Result | Priority |

    Include:
    - Positive tests
    - Negative tests
    - Edge cases
    - Security and validation if applicable
    """

    prompt_parts = [base_prompt]

    if retrieved_context:
        prompt_parts.append(f"\n### Retrieved Context from Previous Tests:\n{retrieved_context}")

    prompt_parts.append(f"\n### JIRA Ticket:\n{jira_ticket}")

    if clarification_history:
        prompt_parts.append(f"\n### Clarification Conversation:\n{clarification_history}")
        prompt_parts.append("\n**Generate test cases based on the JIRA ticket AND the clarifications provided.**")
    else:
        prompt_parts.append("\n**Generate test cases based on the JIRA ticket.**")

    final_prompt = "".join(prompt_parts)

    response = llm.invoke([
        SystemMessage(content=final_prompt),
        HumanMessage(content="Generate comprehensive test cases now.")
    ])

    try:
        conversation_summary = "\n".join(msg.content for msg in state["messages"])
        storage_text = f"CONVERSATION:\n{conversation_summary}\n\nGENERATED TEST CASES:\n{response.content}"
        mongo_vector_store.add_texts([storage_text])
        print("Test cases stored successfully in MongoDB.")
    except Exception as e:
        print(f"Failed to store in MongoDB: {e}")

    return {"messages": [HumanMessage(content=response.content)]}


def should_continue(state: AgentState) -> str:
    if state.get("clarifications_asked", False):
        return "generate_testcases"
    else:
        return "ask_clarifications"


def format_markdown_table_for_display(markdown_table: str) -> str:
    lines = [line.strip() for line in markdown_table.strip().split('\n') if line.strip()]
    table_rows = [line for line in lines if '|' in line and not re.match(r'\|[\s\-:]+\|', line)]

    if len(table_rows) < 2:
        return markdown_table

    header = [col.strip() for col in table_rows[0].split('|')[1:-1]]
    data = []

    for row in table_rows[1:]:
        cells = [col.strip() for col in row.split('|')[1:-1]]
        if len(cells) == len(header):
            data.append(cells)

    return tabulate(data, headers=header, tablefmt="grid")


def parse_markdown_table_to_dataframe(markdown_table: str) -> pd.DataFrame:
    lines = [line.strip() for line in markdown_table.strip().split('\n') if line.strip()]

    table_rows = [line for line in lines if '|' in line and not re.match(r'\|[\s\-:]+\|', line)]

    if len(table_rows) < 2:
        return pd.DataFrame()

    header = [col.strip() for col in table_rows[0].split('|')[1:-1]]
    data = []

    for row in table_rows[1:]:
        cells = [col.strip() for col in row.split('|')[1:-1]]
        if len(cells) == len(header):
            data.append(cells)

    return pd.DataFrame(data, columns=header)


def export_testcases_to_excel(test_cases_content: str, filename: str = None) -> Union[str, None]:
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_cases_{timestamp}.xlsx"

    try:
        df = parse_markdown_table_to_dataframe(test_cases_content)

        if df.empty:
            print("Warning: No valid table data found to export.")
            return None

        filepath = os.path.join('exports', filename)

        directory = os.path.dirname(filepath)

        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        df.to_excel(filepath, index=False, engine='openpyxl')
        print(f"Test cases exported to: {filename}")
        return filename
    except Exception as e:
        print(f"Failed to export to Excel: {e}")
        return None


def parse_markdown_table_to_json(markdown_table: str) -> list:
    lines = [line.strip() for line in markdown_table.strip().split('\n') if line.strip()]

    table_rows = [line for line in lines if '|' in line and not re.match(r'\|[\s\-:]+\|', line)]

    if len(table_rows) < 2:
        return []

    header = [col.strip() for col in table_rows[0].split('|')[1:-1]]
    data = []

    for row in table_rows[1:]:
        cells = [col.strip() for col in row.split('|')[1:-1]]
        if len(cells) == len(header):
            row_dict = {header[i]: cells[i] for i in range(len(header))}
            data.append(row_dict)

    return data

state_graph = StateGraph(AgentState)
state_graph.add_node("ask_clarifications", ask_clarifications_node)
state_graph.add_node("generate_testcases", generate_testcases_node)

state_graph.add_conditional_edges(
    START,
    should_continue,
    {
        "ask_clarifications": "ask_clarifications",
        "generate_testcases": "generate_testcases"
    }
)

state_graph.add_edge("ask_clarifications", END)
state_graph.add_edge("generate_testcases", END)

compiled_graph = state_graph.compile()