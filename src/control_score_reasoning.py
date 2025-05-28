import warnings
import os
from typing import List
from langchain import hub
from typing import Any, List, Dict
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def score_reasoning(state: Dict[str, Any]) -> Dict[str, Any]:
    reasoning_score_prompt = '''
    # Instructions
    - The input is the description and score of a control for a financial payment system. \n
    - Your job is to provide the reasoning behind the score  using the provided rubric. \n
    - Provide a succinct and detailed positive and negative bullet point for each rubric section. \n
    
    # Context
    - Provided scores should be either "Low", "Medium", or "High"
    - This it analyze the controls placed on a banking payment system

    # Rubric
    - Each control is rated on a three-point maturity scale (Low/Medium/High) with the following rubric
    - Following is the rubric:

| **Category**                     | **Sub-Category**       | **Low (Below standard¹)** | **Medium (Industry standard²)** | **High (Best practice³)** |
|----------------------------------|------------------------|-----------------------------|----------------------------------|-----------------------------|
| **Control Design & Risk Coverage** | **Risk Coverage**       | Control does not address a risk breakpoint | Control addresses a risk breakpoint | Control addresses a risk breakpoint tied to a specific process |
|                                  | **Control Design**     | Control design lags industry standards | Control design aligns with industry standards | Control design exceeds industry standards |
|                                  | **Documentation**      | Control documentation is incomplete or non-existent | Documentation exists but lacks depth | Control documentation exists and is comprehensive |
|                                  | **Transaction Coverage** | Control does not cover all applicable transactions | Control covers all applicable transactions | N/A |
| **Implementation & Operation**   | **Execution**          | Control is not implemented or executed as expected | Control is generally followed, but there are gaps at times | Control is consistently executed as expected |
|                                  | **Ownership**          | No clear owner accountable | Clear owner accountable for control | N/A |
|                                  | **Dependencies**       | Key system integrations (i.e., control dependencies) are missing or broken | Key system integrations (i.e., control dependencies) are partially working | Key system integrations (i.e., control dependencies) are working fully, ensuring control effectiveness |
| **Monitoring & Reporting**       | **Testing**            | No regular testing or evidence that the control is working as intended | Periodic testing occurs | Suite of controls for a risk breakpoint tested for effectiveness |
|                                  | **Reporting and KPIs** | No reporting to track control performance | Some reporting to track control performance | Reporting is timely and comprehensive, enabling self-identification of issues |
|                                  | **Governance**         | No tiered governance mechanism in place | Ad hoc tiered governance mechanism in place to drive continuous improvement | Consistent tiered governance mechanism in place to drive continuous improvement |

    # Control Description
    {control}
    
    # Control Score
    This control is rated as
    {score}
    '''

    # Create a chat prompt template using the detailed prompt
    prompt = ChatPromptTemplate(["system", reasoning_score_prompt])

    # Initialize OpenAI Language Model
    #llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_api_key)
    llm = ChatOpenAI(model="o3-mini", reasoning_effort="high", api_key=state["openai_api_key"])

    # Combine the prompt, the language model, and the output parser into a processing chain.
    rag_chain = prompt | llm | StrOutputParser()

    generation = rag_chain.invoke({"control": state["original_input"], "score": state["control_score"]})
    state["control_score_reasoning"] = generation

    return state