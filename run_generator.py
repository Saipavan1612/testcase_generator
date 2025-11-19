from generator_graph import compiled_graph, AgentState, format_markdown_table_for_display, export_testcases_to_excel
from langchain_core.messages import HumanMessage

print("=== AI Test Case Generator (LangChain + LangGraph + Gemini) ===")
ticket = input("Paste your JIRA ticket:\n\n")

# Step 1: Ask clarifications (first invocation)
state: AgentState = {
    "messages": [HumanMessage(content=ticket)],
    "clarifications_asked": False
}

print("\n[INFO] Generating clarification questions...")
clarifications_output = compiled_graph.invoke(state)
clarifications = clarifications_output["messages"][-1].content

print("\n=== Clarification Questions ===")
print(clarifications)

# Step 2: Get user's answers to clarifications
user_provided_clarifications = input("\nProvide answers to ALL clarification questions:\n\n")

# Step 3: Build new state with clarifications and flag set
state_with_clarifications: AgentState = {
    "messages": list(clarifications_output["messages"]) + [HumanMessage(content=user_provided_clarifications)],
    "clarifications_asked": True
}

# Step 4: Generate test cases (second invocation)
print("\n[INFO] Generating test cases based on clarifications...")
testcases_output = compiled_graph.invoke(state_with_clarifications)
testcases = testcases_output["messages"][-1].content

print("\n=== FINAL TEST CASES IN TABULAR FORMAT ===\n")
formatted_table = format_markdown_table_for_display(testcases)
print(formatted_table)

# Export to Excel
print("\n" + "=" * 50)
export_choice = input("\nExport to Excel? (y/n): ").strip().lower()

if export_choice == 'y':
    custom_filename = input("Enter filename (press Enter for auto-generated): ").strip()
    filename = custom_filename if custom_filename else None

    if custom_filename and not custom_filename.endswith('.xlsx'):
        filename = f"{custom_filename}.xlsx"

    export_testcases_to_excel(testcases, filename)