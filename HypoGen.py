#/usr/bin/python
# -*- coding: utf-8 -*-

"""HypoGen.py: Agent using RAG for hypothesis generation based on scientific data.

The script loads the pdf or text data, create a vector database and analyzes the
findings and creates a hypothesis. This is refined checking on contradiction.
Seems the hypothesis valid a design of experiment for verification is proposed.
"""

__author__ = "Markus Schindler"
__copyright__ = "Copyright 2026"

__license__ = "MIT-License"
__version__ = "0.1.0"
__maintainer__ = "Markus Schindler"
__email__ = "schindlerdrmarkus@gmail.com"
__status__ = "Education"

# -------------------------- #
# Built-in / Generic Imports #
# -------------------------- #

import argparse
import logging
import os
import sys

from dotenv import load_dotenv
from pathlib import Path
from operator import add as add_messages
from pypdf import PdfReader
from typing import Annotated, Sequence, TypedDict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END

# ------------------- #
# Logging information #
# ------------------- #

log = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
log.addHandler(handler)
log.setLevel(logging.INFO)

# --------------------- #
# Reading out .env file #
# --------------------- #

class HypoGenConfig:
    def __init__(self):
        # Load config parameters from .env file
        load_dotenv()
        self.llm_model = os.getenv("LLM")
        self.llm_embedding_model = os.getenv("LLM_EMBEDDING")
        self.retriever_k = os.getenv("RETRIEVER_K")
        self.retriever_fetch_k = os.getenv("RETRIEVER_FETCH_K")
        self.collection_name = os.getenv("COLLECTION_NAME")

# ---------------- #
# Main agent class #
# ---------------- #

class HypoGenAgent:
    def __init__(self, config: HypoGenConfig, document_path: Path, database_path: Path):
        self.config = config
        # Minimization of hallucination - temperature = 0 makes the model output more deterministic
        self.llm = ChatOllama(model = self.config.llm_model, temperature = 0)
        # Embedding model should be compatible with the base LLM
        self.embeddings = OllamaEmbeddings(model = self.config.llm_embedding_model)
        self.retriever_k = int(self.config.retriever_k)
        self.retriever_fetch_k = int(self.config.retriever_fetch_k)
        self.collection_name = self.config.collection_name
        self.documents_dir = document_path
        self.persist_directory = database_path

    def _setup_rag(self):
        # Define the source folder
        all_docs = []

        # Ensure directory exists
        if not os.path.exists(self.documents_dir):
            log.error(f"Directory {self.documents_dir} does not exist.")
            return

        for filename in os.listdir(self.documents_dir):
            file_path = os.path.join(self.documents_dir, filename)
            
            # --- PDF HANDLING ---
            if filename.endswith(".pdf"):
                try:
                    reader = PdfReader(file_path)
                    # Extract text from ALL pages and join into one continuous string
                    # We add a newline between pages to ensure words don't stick together
                    full_text = "\n".join([page.extract_text() for page in 
reader.pages if page.extract_text()])
                    
                    if full_text:
                        # Create ONE document per file instead of one per page
                        all_docs.append(Document(
                            page_content = full_text, 
                            metadata = {"source": filename}
                        ))
                except Exception as e:
                    log.error(f"Could not read PDF {filename}: {e}")

            # --- TXT HANDLING ---
            elif filename.endswith(".txt"):
                try:
                    with open(file_path, "r", encoding = "utf-8") as f:
                        text = f.read()
                        all_docs.append(Document(
                            page_content = text, 
                            metadata = {"source": filename}
                        ))
                except Exception as e:
                    log.error(f"Could not read TXT {filename}: {e}")

        log.info(f"Loaded {len(all_docs)} files for processing.")

        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1000,
                chunk_overlap = 200,
                length_function = len,
                is_separator_regex = False
        )
        chunks = text_splitter.split_documents(all_docs)

        # Check if the persist_directory already contains data to avoid duplicating entries
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            log.info("Loading existing ChromaDB vector store...")
            vectorstore = Chroma(
                persist_directory = self.persist_directory,
                embedding_function = self.embeddings,
                collection_name = self.collection_name
            )
        else:
            log.info("Creating new ChromaDB vector store...")
            if not os.path.exists(self.persist_directory):
                os.makedirs(self.persist_directory)
            try:
                vectorstore = Chroma.from_documents(
                    documents = chunks,
                    embedding = self.embeddings,
                    persist_directory = self.persist_directory,
                    collection_name = self.collection_name
                )
                log.info("Created ChromaDB vector store!")
            except Exception as e:
                log.error(f"Error Setting up ChromaDB: {str(e)}")
                raise

        # Now we create our Retriever - Librarian
        self.retriever = vectorstore.as_retriever(
            #search_type = "similarity",
            #search_kwargs = {"k": 5} # k is the amount of chunks to return
            search_type = "mmr",
            search_kwargs = {
                "k": self.retriever_k, # Number of documents to return
                "fetch_k": self.retriever_fetch_k, # Number of documents to initially fetch for diversity filtering
                "lambda_mult": 0.5 # Diversity parameter: 0 is max diversity, 1 is max similarity
            }
        )
    def _create_retriever_tool(self):
        @tool
        def retriever_tool(query: str) -> str:
            """
            This tool searches and return the information from the provided literature.
            """

            docs = self.retriever.invoke(query)

            if not docs:
                return "I found no relevant information in the data."

            results = []
            for i, doc in enumerate(docs):
                results.append(f"Document {i+1}:\n{doc.page_content}")

            return "\n\n".join(results)

        self.tools = [retriever_tool]
        self.llm = self.llm.bind_tools(self.tools)

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        problem: str
        hypothesis: str
        falsifications: str
        iteration_count: int
        doe: str
    
    def should_continue(self, state: AgentState):
        """Check if the last message contains tool calls."""

        result = state["messages"][-1]
        return hasattr(result, "tool_calls") and len(result.tool_calls) > 0

    # LLM Agent
    def call_llm(self, state: AgentState) -> AgentState:
        """Function to call the LLM with the current state."""
        messages = list(state["messages"])
        messages = [SystemMessage(content = self.system_prompt)] + messages
        messages = self.llm.invoke(messages)
        return {"messages": [messages]}

    # Retriever Agent
    def take_action(self, state: AgentState) -> AgentState:
       """Execute tool calls from the LLM's response."""
       tool_calls = state["messages"][-1].tool_calls
       results = []
       for t in tool_calls:
           log.info(f"Calling Tool: {t["name"]} with query {t["args"].get("query", "No query provided")}")

           if not t["name"] in self.tools_dict: # Checks if a valid tool is present
               log.info(f"\nTool: {t["name"]} does not exist.")
               result = "Incorrect Tool Name, Please Retry and Select tool from List of Available Tools."

           else:
               result = self.tools_dict[t["name"]].invoke(t["args"].get("query", ""))
               log.info(f"Result length: {len(str(result))}")

           # Appends the Tool Message
           results.append(ToolMessage(tool_call_id = t["id"], name = t["name"], content = str(result)))

       log.info("Tools Execution Complete. Back to the model!")
       return {"messages": results}

    # Hypothesizer
    def hypothesizer(self, state: AgentState) -> AgentState:
        log.info("Hypothesizer started!")
        """Function to create only one hypothesis for the selected problem."""
        system_prompt = """
        You are an expert Scientific Researcher. Your task is to synthesize a single, clear, and testable hypothesis based on a specific problem and the provided factual evidence.

        Instructions:
        1. Formulate exactly ONE simple hypothesis.
        2. Use a causal structure: Identify a primary impact factor (independent variable) and describe its effect on the problem.
        3. Explicitly define the relationship by contrasting the outcomes of 'Low severity/level' versus 'High severity/level' of this impact factor.
        4. Ensure the hypothesis is logically grounded in the provided 'Retrieved Facts'.
        
        Format your response as:
        - Hypothesis: [A concise statement of the proposed relationship]
        - Impact Analysis: 
            - Low [Factor]: [Expected result]
            - High [Factor]: [Expected result]
        """
        problem = state.get("problem", "No problem provide")
        facts = list(state.get("messages", []))
        messages = [SystemMessage(content = system_prompt + problem)] + facts
        response = self.llm.invoke(messages)
        return {"hypothesis": response.content}

    # Falsifier
    def falsifier(self, state: AgentState) -> AgentState:
        """
        Performs a rigorous contradiction check between the hypothesis and retrieved facts.
        Improved prompt to ensure consistency for the conditional routing logic.
        """
        log.info("Falsifier started!")
        """ Function to find contradictions for the made hypothesis for the selected problem."""
        system_prompt = """
        You are an expert Scientific Auditor. Your goal is to identify logical contradictions between a proposed hypothesis and the retrieved evidence.
        
        Instructions:
        1. Compare the hypothesis against the 'Retrieved Facts' provided in the conversation history.
        2. If the evidence explicitly contradicts the hypothesis, you MUST start your response with the exact phrase: "Contradiction found!".
        3. Provide a rigorous analysis explaining exactly where the contradiction lies, citing specific points from the retrieved text, and explain why this invalidates the hypothesis.
        4. If no contradiction is found, start your response with: "No contradiction found." and briefly justify why the hypothesis is consistent with the data.
        5. Be concise, objective, and scientifically precise.
        """
        hypothesis = state.get("hypothesis", "No hypothesis provided")
        facts = list(state.get("messages", []))
        messages = [SystemMessage(content = system_prompt + "\n\nHypothesis: " + hypothesis)] + facts
        response = self.llm.invoke(messages)
        return {"falsifications": response.content}

    def problem_redefiner(self, state: AgentState) -> AgentState:
        log.info("Problem Redefiner started!")
        """Redefines the problem statement to resolve contradictions found during the falsifier.
        Improved prompt to ensure a high-quality, narrow scientific objective.
        Function to redefine the problem statement based on the falsifier to avoid contradictions."""
        system_prompt = """
        You are a Principal Investigator specializing in research design. A previous hypothesis was rejected due to contradictions with existing literature.
        
        Your task is to evolve the research question to resolve these conflicts:
        1. Analyze the original problem, the rejected hypothesis, and the specific contradictions identified in the falsifier.
        2. Redefine the problem statement to be more precise, narrow the scope, or pivot the scientific angle to account for the literature's constraints.
        3. Your output must be ONLY the revised problem statement. Do not include introductory text, explanations, or conversational fillers.
        """
        problem = state.get("problem", "No problem provided")
        hypothesis = state.get("hypothesis", "No hypothesis provided")
        falsifications = state.get("falsifications", "No falsifications provided")
    
        messages = [
            SystemMessage(content = system_prompt),
            HumanMessage(content = f"Original Problem: {problem}\nHypothesis: {hypothesis}\nFalsifications: {falsifications}")
        ]
        response = self.llm.invoke(messages)
    
        # Update the problem and increment the iteration count
        current_count = state.get("iteration_count", 0)
        return {"problem": response.content, "iteration_count": current_count + 1}

    def should_redefine(self, state: AgentState):
        """
        Conditional edge that checks whether the falsifier node found a contradiction 
        and ensures the loop does not exceed three cycles. If no contradiction or max 
        iterations reached, it proceeds to the DoE proposal.
        """
        falsifications_text = state.get("falsifications", "")
        iteration_count = state.get("iteration_count", 0)
    
        if "Contradiction found!" in falsifications_text and iteration_count < 3:
            log.info(f"Contradiction detected. Iteration {iteration_count + 1}/3. Redefining problem...")
            return "redefine"
    
        return "doe"

    def doe_proposer(self, state: AgentState) -> AgentState:
        log.info("DoE Proposer started!")
        """Function to propose a design of experiment based on the final hypothesis and retrieved facts."""
        system_prompt = """
        You are a Senior Research Experimentator and expert in Design of Experiments (DoE). Your goal is to translate a theoretical hypothesis and supporting literature into a rigorous, empirical validation framework.

        Based on the number of critical parameters identified in the hypothesis, you must propose the appropriate full factorial design:
        - For 2 parameters: A 2^2 factorial design.
        - For 3 parameters: A 2^3 factorial design.
        - For 4 parameters: A 2^4 factorial design.

        Your proposal must be structured as follows:
        1. Experimental Objective: A precise and concise statement of the scientific goal.
        2. Variables: 
           - Independent Variables: Define each factor and specify the 'High (+)' and 'Low (-)' levels to be tested.
           - Dependent Variables: Define the measurable responses (e.g., yield, purity, crystal size) used to evaluate the outcome.
        3. Methodology: A detailed, step-by-step experimental protocol for synthesis and measurement.
        4. Control Parameters: A list of variables that must remain constant to ensure the validity of the results and eliminate confounding factors.
        5. Validation Criteria: A clear a priori definition of the results that would either support (prove) or reject (disprove) the hypothesis.
        """
    
        hypothesis = state.get("hypothesis", "No hypothesis provided")
        # Extract content from the message history to get the "facts" collected by the retriever agent
        facts_context = "\n".join([msg.content for msg in state.get("messages", []) if hasattr(msg, 'content')])
    
        messages = [
            SystemMessage(content = system_prompt),
            HumanMessage(content = f"Retrieved Facts:\n{facts_context}\n\nFinal Hypothesis: {hypothesis}\n\nProposed Design of Experiment:")
        ]
    
        response = self.llm.invoke(messages)
        return {"doe": response.content}

    def _agent_builder(self):
        self.system_prompt = """
        You are an expert Scientific Literature Analyst. Your primary objective is to synthesize precise, evidence-based technical information from the provided document repository to support the formulation of scientific hypotheses.

        Guidelines:
        1. Utilize the retriever tool to gather factual evidence strictly from the provided papers.
        2. Perform multiple, iterative queries if the initial results are insufficient or if you need 
           to bridge information gaps to provide a comprehensive answer.
        3. Ensure all responses are grounded exclusively in the retrieved text to eliminate hallucinations.
        4. If the available literature does not contain the required information, explicitly state that 
           the data is unavailable rather than inferring an answer.
        """
        self.tools_dict = {our_tool.name: our_tool for our_tool in self.tools} # Creating a dictionary of our Tools
        
        # Initialize the updated graph structure
        graph = StateGraph(self.AgentState)

        # Define nodes
        graph.add_node("llm", self.call_llm)
        graph.add_node("retriever_agent", self.take_action)
        graph.add_node("hypothesizer", self.hypothesizer)
        graph.add_node("falsifier", self.falsifier)
        graph.add_node("problem_redefiner", self.problem_redefiner)
        graph.add_node("doe_proposer", self.doe_proposer)

        # Define edges and flow
        graph.add_conditional_edges(
            "llm",
            self.should_continue,
            {
                True: "retriever_agent", 
                False: "hypothesizer",
            }
        )

        graph.add_edge("retriever_agent", "llm")
        graph.add_edge("hypothesizer", "falsifier")

        # Route from falsifier either back to redefinition or forward to DoE proposal
        graph.add_conditional_edges(
            "falsifier",
            self.should_redefine,
            {
                "redefine": "problem_redefiner",
                "doe": "doe_proposer"
            }
        )

        graph.add_edge("problem_redefiner", "hypothesizer")
        graph.add_edge("doe_proposer", END)

        graph.set_entry_point("llm")
        self.hypothesis_agent = graph.compile()

    def running_agent(self):
        print("\n=== HypoGen AGENT ===\n")
        while True:
            problem = input("\nEnter a scientific problem (or type 'exit' to quit): ")
            
            if problem.lower() == "exit":
                print("Exiting HypoGen Agent. Goodbye!")
                break
            
            if not problem.strip():
                print("Problem cannot be empty. Please try again.")
                continue

            # Initialize messages and start the agent
            messages = [HumanMessage(content=problem)]
            # Passing problem and iteration_count to ensure state is initialized for the nodes
            result = self.hypothesis_agent.invoke({
                "messages": messages, 
                "problem": problem, 
                "iteration_count": 0
            })
            
            # Construct the output text
            text = "\n\n=== FACTS ===\n\n"
            text += str(result["messages"][-1].content)
            text += "\n\n=== HYPOTHESIS ===\n\n"
            text += str(result["hypothesis"])
            text += "\n\n=== CONTRADICTIONS ===\n\n"
            text += str(result["falsifications"])
            text += "\n\n=== DoE ===\n\n"
            text += str(result["doe"])
            
            print(text)
            
            # Handle saving logic
            print("\nDo you want to save the output?")
            save_text = input("Please enter Yes or No: ").strip().lower()
            if save_text == "yes" or save_text == "Yes" or save_text == "YES":
                with open("output.txt", "a") as file: # Using append mode 'a' to keep history of different problems
                    file.write(f"\n{'='*50}\n")
                    file.write(f"Problem: {problem}\n")
                    file.write(f"{text}\n")
                log.info("\nOutput has been saved under output.txt")
       
# ------------------------------- #
# Argument parsing & main routine #
# ------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "HypoGen Agent for scientific hypothesis generation")
    parser.add_argument(
        "--document_path",
        type = Path,
        default = Path("documents"),
        help = "Path to the documents for reading out literature and experiments"
    )
    parser.add_argument(
        "--database_path",
        type = Path,
        default = Path("ChromaDB"),
        help = "Path to the vector database for storing retrieved facts"
    )
    return parser.parse_args()

# ------------ #
# Main Routine #
# ------------ #

def main() -> None:
    args = parse_args()

    # 1. Initialize agent
    hypogen = HypoGenAgent(HypoGenConfig(), args.document_path, args.database_path)

    # 2. Setup RAG
    hypogen._setup_rag()

    # 3. Initialize retriever tool
    hypogen._create_retriever_tool()

    # 4. Build agent
    hypogen._agent_builder()

    # 5. Run agent
    hypogen.running_agent()

if __name__ == "__main__":
    main()
