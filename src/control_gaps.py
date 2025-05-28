import warnings
import os
from typing import List
from langchain import hub
from typing import Any, List, Dict
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def gaps(state: Dict[str, Any]) -> Dict[str, Any]:
    gaps_prompt = '''
    # Instructions
    - What are the main operational or technical gaps that this control is solving for? \n
    - Provide answer in 3-6 succinct bullet points.\n

    # Context
    - This it analyze the controls placed on a banking payment system

    # Control
    {control}
    '''

    # Create a chat prompt template using the detailed prompt
    prompt = ChatPromptTemplate(["system", gaps_prompt])

    # Initialize OpenAI Language Model
    llm = ChatOpenAI(model="o3-mini", reasoning_effort="high", api_key=state["openai_api_key"])

    # Combine the prompt, the language model, and the output parser into a processing chain.
    rag_chain = prompt | llm | StrOutputParser()

    generation = rag_chain.invoke({"control": state["original_input"]})
    state["control_gaps"] = generation

    return state