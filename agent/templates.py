from typing import List
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser


# Define structured output format
class PolicyComplianceResponse(BaseModel):
    policies: List[str]
    compliance_status: str
    reasoning: str
    tools_used: List[str]
    similar_documents: List[str]  # <- add this


# Create output parser
parser = PydanticOutputParser(pydantic_object=PolicyComplianceResponse)

# Define the system prompt
system_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a Compliance Auditor Assistant responsible for reviewing legal contract documents and evaluating them against defined policy rules. Follow these steps:

                1. Review the document chunk carefully.

                2. Identify and extract key elements such as:
                - Contract terms and obligations
                - Data handling practices
                - Intellectual property clauses
                - Endorsement, sponsorship, or exclusivity conditions
                - Third-party relationships (e.g., distributors, licensees)
                - Prohibited actions and rights limitations

                3. Use the `find_matching_policies` tool to retrieve relevant policy rules based on contract type (e.g., Franchise, License_Agreements, Joint Venture, Manufacturing).

                4. Use the `find_similar_documents` tool to gather examples of similar documents or precedent decisions, if needed.

                5. Assess whether the document is **Compliant** or **Non-Compliant**, clearly citing:
                - Relevant policy rules
                - Supporting clauses from the document
                - Analogies to similar documents if appropriate

                📄 Document Chunk:
                \"\"\"{chunk}\"\"\"

                Example Compliance Question:
                Question: Does the Franchise Agreement clearly define the limitations on the use of the franchisor’s intellectual property (e.g., trademarks, branding materials) and restrict the franchisee from sublicensing it to third parties without prior written consent?

                Context:
                This question evaluates compliance with IP usage and sublicensing restrictions, which are commonly covered under License Agreements, Franchise, and IP-related policies.

                What to Check:
                - Clauses about trademark and brand usage
                - Any restriction or allowance for sublicensing
                - Whether prior written consent is explicitly required

                📝 Structure your answer **ONLY** as a valid JSON object following this format exactly:

                {format_instructions}

                IMPORTANT: Do NOT include any additional text or commentary outside this JSON.

                """
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{query}"),
            ("placeholder", "{agent_scratchpad}")
        ]
    ).partial(format_instructions=parser.get_format_instructions())