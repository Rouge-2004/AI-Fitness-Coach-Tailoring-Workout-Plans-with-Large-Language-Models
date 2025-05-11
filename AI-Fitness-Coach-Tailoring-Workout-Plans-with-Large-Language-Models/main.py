import streamlit as st
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    st.error("❌ Missing Gemini API Key in .env file!")
    st.stop()

# State definition
class State(TypedDict):
    user_data: dict
    fitness_plan: str
    feedback: str
    progress: List[str]
    messages: Annotated[list, add_messages]

# Get Gemini LLM (with ADC disabled)
def get_gemini_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=gemini_api_key,
        temperature=0,
        credentials=None
    )

# User Input Agent (ensures all required fields exist)
def user_input_agent(state: State, llm):
    required_fields = {
        "dietary_preferences": "",
        "workout_days": ["Monday", "Wednesday", "Friday"],
        "workout_duration": 30,
        "workout_preferences": ["Cardio", "Strength Training"],
        "health_conditions": "None"
    }
    
    # Merge user input with default values
    state["user_data"] = {**required_fields, **state["user_data"]}
    
    prompt = ChatPromptTemplate.from_template(
        """Create a JSON profile with these fields:
        {user_input}
        
        Add missing fields with sensible defaults."""
    )
    chain = prompt | llm | StrOutputParser()
    try:
        state["user_data"] = json.loads(chain.invoke({"user_input": json.dumps(state["user_data"])}))
    except json.JSONDecodeError:
        st.error("Failed to parse user profile")
    return state

# Routine Generation Agent (uses all required variables)
def routine_generation_agent(state: State, llm):
    user_data = state["user_data"]
    prompt = ChatPromptTemplate.from_template(
        """Create a fitness plan for:
        - Age: {age}
        - Weight: {weight}kg
        - Goal: {primary_goal}
        - Preferences: {workout_preferences}
        - Days: {workout_days}
        - Duration: {workout_duration} mins
        - Dietary: {dietary_preferences}
        - Health: {health_conditions}"""
    )
    chain = prompt | llm | StrOutputParser()
    state["fitness_plan"] = chain.invoke(user_data)
    return state

# AIFitnessCoach class
class AIFitnessCoach:
    def __init__(self):
        self.llm = get_gemini_llm()
        self.graph = self._build_workflow()

    def _build_workflow(self):
        workflow = StateGraph(State)
        workflow.add_node("user_input", lambda s: user_input_agent(s, self.llm))
        workflow.add_node("generate_plan", lambda s: routine_generation_agent(s, self.llm))
        workflow.add_edge("user_input", "generate_plan")
        workflow.add_edge("generate_plan", END)
        workflow.set_entry_point("user_input")
        return workflow.compile()

    def run(self, user_input):
        return self.graph.invoke({
            "user_data": user_input,
            "fitness_plan": "",
            "feedback": "",
            "progress": [],
            "messages": []
        })

# Streamlit UI
def main():
    st.set_page_config(page_title="AI Fitness Coach", layout="wide")
    st.title("💪 AI Fitness Coach")

    if "coach" not in st.session_state:
        st.session_state.coach = AIFitnessCoach()

    with st.form("user_input_form"):
        st.subheader("Basic Info")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        weight = st.number_input("Weight (kg)", min_value=30, value=70)
        goal = st.selectbox("Primary Goal", ["Weight Loss", "Muscle Gain", "Endurance"])
        
        st.subheader("Preferences")
        workout_days = st.multiselect("Workout Days", 
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            default=["Monday", "Wednesday", "Friday"]
        )
        duration = st.slider("Session Duration (mins)", 15, 120, 45)
        preferences = st.multiselect("Workout Types",
            ["Cardio", "Strength Training", "Yoga", "HIIT"],
            default=["Cardio", "Strength Training"]
        )
        diet = st.text_input("Dietary Preferences", "Balanced")
        health = st.text_input("Health Conditions", "None")

        if st.form_submit_button("Generate Plan"):
            user_data = {
                "age": age,
                "weight": weight,
                "primary_goal": goal,
                "workout_days": workout_days,
                "workout_duration": duration,
                "workout_preferences": preferences,
                "dietary_preferences": diet,
                "health_conditions": health
            }
            
            result = st.session_state.coach.run(user_data)
            st.session_state.plan = result["fitness_plan"]
            st.success("Plan generated!")

    if "plan" in st.session_state:
        st.subheader("Your Fitness Plan")
        st.write(st.session_state.plan)

if __name__ == "__main__":
    main()