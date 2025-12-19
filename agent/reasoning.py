from langchain.agents import create_tool_calling_agent, AgentExecutor

from src.utils import get_llm
from agent.templates import system_prompt
from agent.tools import chunk_embedding_tool, matching_policy_tool, similar_document_tool


def create_policy_compliance_agent(llm_type="openai", model_name="gpt-4o") -> AgentExecutor:
    # Create the LLM and agent
    llm = get_llm(llm_type, model_name=model_name)
    

    agent = create_tool_calling_agent(
        llm=llm,
        prompt=system_prompt,
        tools=[chunk_embedding_tool, matching_policy_tool, similar_document_tool],
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=[chunk_embedding_tool, matching_policy_tool, similar_document_tool],
        verbose=True
    )

    return agent_executor
