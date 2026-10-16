import os
import uvicorn

from fastapi import FastAPI
from langserve import add_routes

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda

# -------------------------------------------------------
# Tool 1 - Student Details
# -------------------------------------------------------

@tool
def get_student_details(roll_no: str) -> str:
    """Returns student details using roll number."""

    students = {
        "24R21A66J1": {
            "Name": "Rajesh Prasad Bailwal",
            "Branch": "CSE-AIML",
            "Year": "3rd Year",
            "College": "MLR Institute of Technology",
            CGPA": "7.1"
        },
        "24R21A66J4": {
            "Name": "Scanner",
            "Branch": "CSE-AIML",
            "Year": "3rd Year",
            "College": "MLR Institute of Technology",
            "CGPA": "8.2"
        }
    }

    student = students.get(roll_no.upper())

    if student:
        return str(student)

    return "Student not found."


# -------------------------------------------------------
# Tool 2 - Student Projects
# -------------------------------------------------------

@tool
def get_student_projects(name: str) -> str:
    """Returns student projects."""

    projects = {
        "Rajesh Prasad":
        """
        1. Style Swap
        2. Online Fingerprint Authentication for Banking
        3. Gemini RAG Chatbot
        """,

        "Scanner":
        """
        1. Student Attendance System
        2. Library Management System
        """
    }

    return projects.get(name.lower(), "No projects found.")


# -------------------------------------------------------
# Tool 3 - Student Skills
# -------------------------------------------------------

@tool
def get_student_skills(name: str) -> str:
    """Returns student technical skills."""

    skills = {
        "rajesh":
        """
        Python
        Java
        SQL
        HTML
        CSS
        LangChain
        LangGraph
        FastAPI
        FAISS
        Google Gemini API
        Git
        GitHub
        """,

        "scanner":
        """
        Python
        Java
        MySQL
        HTML
        CSS
        """
    }

    return skills.get(name.lower(), "Skills not available.")


# -------------------------------------------------------
# Tool List
# -------------------------------------------------------

tools = [
    get_student_details,
    get_student_projects,
    get_student_skills
]

# -------------------------------------------------------
# Gemini Model
# -------------------------------------------------------

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GOOGLE_API_KEY,
    temperature=0
)

# -------------------------------------------------------
# Create Agent
# -------------------------------------------------------

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are a Student Information Assistant.

You are allowed to answer ONLY questions related to students,
their details, projects, and technical skills.

If the user asks anything outside student information,
reply exactly:

I am authorized only to answer student-related questions.
"""
)

# -------------------------------------------------------
# Input Schema
# -------------------------------------------------------

class AgentInput(BaseModel):
    input: str = Field(description="Ask a student related question")


# -------------------------------------------------------
# Format User Input
# -------------------------------------------------------

def format_for_agent(data):

    if isinstance(data, dict):
        message = data["input"]
    else:
        message = data.input

    return {
        "messages": [
            ("user", message)
        ]
    }


# -------------------------------------------------------
# Extract Final Response
# -------------------------------------------------------

def extract_response(output):

    if not isinstance(output, dict):
        return str(output)

    messages = output.get("messages")

    if messages is None:
        for value in output.values():
            if isinstance(value, dict) and "messages" in value:
                messages = value["messages"]
                break

    if messages:
        return getattr(messages[-1], "content", str(messages[-1]))

    return str(output)


# -------------------------------------------------------
# Runnable Chain
# -------------------------------------------------------

chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_response)
).with_types(
    input_type=AgentInput,
    output_type=str
)

# -------------------------------------------------------
# FastAPI
# -------------------------------------------------------

app = FastAPI(
    title="Student Information Agent",
    version="1.0",
    description="Single Agent using LangChain + Gemini + LangServe"
)

add_routes(
    app,
    chain,
    path="/agent",
    playground_type="default"
)

# -------------------------------------------------------
# Home Route
# -------------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Student Information Agent Running Successfully 🚀"
    }

# -------------------------------------------------------
# Run Server
# -------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
