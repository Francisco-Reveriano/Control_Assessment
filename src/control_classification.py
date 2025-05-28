import warnings
import os
from langchain import hub
from typing import Any, List, Dict
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def classify(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify the input control into predefined payment system categories based on provided
    descriptions and return the updated state with the classification result added.

    :param state: A dictionary containing the necessary state information, notably the
                  "openai_api_key" for accessing the OpenAI API and the current input
                  control description to be classified.
    :type state: Dict[str, Any]
    :return: The updated state dictionary with the added control classification result
             under the "control_classification" key.
    :rtype: Dict[str, Any]
    """
    classification_prompt = '''
    # Instructions
    - The input is the description of a control for a financial payement system
    - Your job is to classify the input into the following categories:
        -- Validation
        -- Duplicates
        -- Sanctions
        -- Fraud
        -- Insufficient Funds
        -- High Dollar Escalation
        -- Completed Fields
        -- Travel Rules
    - Trace back your reasoning to the input
    - Return only the classification as a string
    - Make sure you do not return any other information besides the classification

    # Context
    - Validation: Controls that are used to validate the payment account information with OVS
    - Duplicates: Controls that checks if a payment is a duplicate payment
    - Sanctions: Controls that checks if a payment is violation a sanction
    - Control: Controls that checks if a payment is fraudulent
    - Insufficient Funds: Controls that checks if a payment is from an account with insufficient funds
    - High-Dollar Escalation: Controls that checks if a payment is greater than $20 million
    - Completed Fields: Controls that checks if a payment has all fields completed
    - Travel Rules: Controls that checks if a payment is compliant with the Travel Rules

    # Input
    {input}
    '''

    # Create a chat prompt template using the detailed prompt
    prompt = ChatPromptTemplate(["system", classification_prompt])

    # Initialize OpenAI Language Model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=state["openai_api_key"])

    # Combine the prompt, the language model, and the output parser into a processing chain.
    rag_chain = prompt | llm | StrOutputParser()

    state["control_classification"] = rag_chain.invoke({"input": state["original_input"]})

    return state