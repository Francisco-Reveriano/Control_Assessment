import warnings
import os
from langchain import hub
from typing import Any, List, Dict
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def risks(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyses the control risks in a payment system by simulating a prompt-based interaction
    with a language model using specified context and inputs. This function dynamically
    generates a list of summarized risks based on the control details provided.

    :param state: A dictionary containing state information used for the analysis.
                  - openai_api_key (str): OpenAI API key for authentication.
                  - original_input (str): The original user-provided input to analyze control risks.
                  - control_risk (Optional[str]): Output of risk analysis will be stored in this key.

    :return: Updated state dictionary with the control risks analysis stored in the "control_risk" field.
    :rtype: Dict[str, Any]

    :raises ValueError: If required keys (e.g., openai_api_key, original_input) are missing from the "state" input.
    """
    risk_prompt = '''
    # Instructions
    - What are the main risks that this control is solving for? \n
    - Provide answer in 3-6 succinct bullet points.\n

    # Context
    - This it analyze the controls placed on a banking payment system
    
    # Control
    {control}
    '''

    # Create a chat prompt template using the detailed prompt
    prompt = ChatPromptTemplate(["system", risk_prompt])

    # Initialize OpenAI Language Model
    llm = ChatOpenAI(model="o3-mini", reasoning_effort="low", api_key=state["openai_api_key"])

    # Combine the prompt, the language model, and the output parser into a processing chain.
    rag_chain = prompt | llm | StrOutputParser()

    generation = rag_chain.invoke({"control": state["original_input"]})
    state["control_risk"] = generation

    return state