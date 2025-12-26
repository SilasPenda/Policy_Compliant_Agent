from typing import List
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser


class PolicyComplianceResponse(BaseModel):
    compliant_policies: List[str] = Field(
        description="Policies fully satisfied by the document"
    )
    violated_policies: List[str] = Field(
        description="Policies violated or not satisfied by the document"
    )
    compliance_status: str = Field(description="Compliant or Non-Compliant")
    reasoning: str = Field(description="Explain why each policy is compliant or violated")
    tools_used: List[str] = Field(default_factory=list)
    similar_documents: List[str] = Field(default_factory=list)

# Output parser
parser = PydanticOutputParser(pydantic_object=PolicyComplianceResponse)

# System prompt
system_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a Compliance Auditor Assistant responsible for reviewing legal contract documents and evaluating them against defined policy rules.

            ⚠️ Respond ONLY with a single valid JSON object. Do NOT include explanations or commentary outside the JSON.

            1. Review the document chunk carefully.

            2. Identify and extract key elements such as:
            - Contract terms and obligations
            - Data handling practices
            - Security and privacy clauses (HIPAA and internal IT security if applicable)
            - Intellectual property clauses
            - Endorsement, sponsorship, or exclusivity conditions
            - Third-party relationships (e.g., distributors, licensees)
            - Prohibited actions and rights limitations

            3. Use `find_matching_policies` to retrieve policies based on:
            - Contract type (e.g., Franchise, License_Agreements, Joint Venture, Manufacturing)
            - Policy domain (contracts, security, privacy, HIPAA, IT security)
            - Severity and category if relevant

            4. Use `find_similar_documents` to gather examples if needed.

            5. Assess whether the document is Compliant or Non-Compliant, clearly citing:
            - compliant_policies
            - violated_policies
            - Supporting clauses
            - Analogies to similar documents

            📄 Document Chunk:
            \"\"\"{chunk}\"\"\"

            Example Compliance Questions:
            - "Does the contract comply with all security and privacy regulations (HIPAA, internal IT security)?"
            - "Are effective dates and termination clauses clearly defined?"
            - "Are governing law and jurisdiction specified?"
            - "Does the document restrict IP usage and sublicensing appropriately?"

            📝 Structure your answer EXACTLY as a valid JSON object following this format:

            {format_instructions}

            Rules:
            - If compliance_status = "Compliant", violated_policies MUST be empty
            - If compliance_status = "Non-Compliant", compliant_policies MUST be empty
            - Every policy mentioned in reasoning MUST appear in exactly one list
            """
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}")
    ]
).partial(format_instructions=parser.get_format_instructions())
