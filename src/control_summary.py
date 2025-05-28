import warnings
import os
from langchain import hub
from typing import Any, List, Dict
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def summary(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a succinct summary for financial payment system controls based on the input and updates the state with
    the generated summary.

    :param state: The state dictionary containing the following keys:
                  - "openai_api_key" (str): The API key for OpenAI.
                  - "original_input" (str): The detailed control description to summarize.
    :type state: Dict[str, Any]

    :return: The updated state dictionary with the generated control summary added under the key "control_summary".
    :rtype: Dict[str, Any]
    """
    summary_prompt = '''
    # Instructions
    - The input is the description of a control for a financial payment system
    - Create a succinct summary of the control
    - Summary should be succinct and concise
    - Summary should not have more than 1 sentence

    # Input
    {input}
    '''

    # Create a chat prompt template using the detailed prompt
    prompt = ChatPromptTemplate(["system", summary_prompt])

    # Initialize OpenAI Language Model
    llm = ChatOpenAI(model="o3-mini", reasoning_effort="low", api_key=state["openai_api_key"])

    # Combine the prompt, the language model, and the output parser into a processing chain.
    rag_chain = prompt | llm | StrOutputParser()

    generation = rag_chain.invoke({"input": state["original_input"]})
    state["control_summary"] = generation

    return state