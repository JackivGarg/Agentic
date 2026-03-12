import sys
import os
import httpx
import streamlit as st
from src.config import VALID_CATEGORIES

st.set_page_config(page_title="Benny AI - Unified Bot", page_icon="🤖", layout="wide")

# API Configuration
API_URL = "http://localhost:8000"

# Admin Credentials (as requested)
ADMIN_CREDENTIALS = {
    "name": "JACKIV GARG",
    "email": "jackiv@gmail.com",
    "password": "admin@123"
}

if "human_review_toggle" not in st.session_state:
    st.session_state.human_review_toggle = False
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "pending_interpreted" not in st.session_state:
    st.session_state.pending_interpreted = None
if "pending_session_id" not in st.session_state:
    st.session_state.pending_session_id = None
if "human_review_waiting" not in st.session_state:
    st.session_state.human_review_waiting = False
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False
if "admin_user" not in st.session_state:
    st.session_state.admin_user = None

def clear_human_review_state():
    st.session_state.pending_query = None
    st.session_state.pending_interpreted = None
    st.session_state.pending_interpreted_lc = None
    st.session_state.pending_session_id = None
    st.session_state.human_review_waiting = False
    st.session_state.pending_mode = None
    if 'comp_lc_res' in st.session_state:
        del st.session_state.comp_lc_res

def call_chat_api(query, session_id, mode, use_human_review=False, edited_query=None):
    """Generator that yields text chunks from the FastAPI stream."""
    try:
        payload = {
            "query": query,
            "session_id": session_id,
            "mode": mode,
            "use_human_review": use_human_review,
            "edited_query": edited_query
        }
        with httpx.stream("POST", f"{API_URL}/chat", json=payload, timeout=60.0) as response:
            response.raise_for_status()
            for chunk in response.iter_text():
                if chunk:
                    yield chunk
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        yield f" Error connecting to backend: {str(e)}"

def get_rewritten_query(query: str, session_id: str) -> str:
    try:
        with httpx.Client(timeout=10.0) as client:
            res = client.post(f"{API_URL}/rewrite", json={"query": query, "session_id": session_id})
            if res.status_code == 200:
                return res.json().get("rewritten_query", query)
    except Exception as e:
        st.error(f"Error fetching rewrite: {e}")
    return query

def display_query_box(original: str, interpreted: str):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Your Query**")
        st.info(original)
    with col2:
        st.markdown("**Rewritten Question**")
        st.success(interpreted)

def show_human_review_ui(mode: str):
    prompt = st.session_state.pending_query
    interpreted_q = st.session_state.pending_interpreted
    session_id = st.session_state.pending_session_id or "default"
    
    with st.chat_message("assistant"):
        st.markdown(f"**{mode} Mode**")
        display_query_box(prompt, interpreted_q)
        
        edited_q = st.text_area("Edit query if needed:", value=interpreted_q, height=80, key="edit_query")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            proceed_btn = st.button("Proceed", key="proceed_btn")
        with col2:
            cancel_btn = st.button("Cancel", key="cancel_btn")
        
        if cancel_btn:
            clear_human_review_state()
            st.rerun()
        
        if proceed_btn:
            run_langgraph_respond(prompt, session_id, edited_q)
            clear_human_review_state()
            st.rerun()

def render_comparison_review(session_id):
    prompt = st.session_state.pending_query
    interpreted_q_lg = st.session_state.pending_interpreted
    
    st.markdown("---")
    st.markdown("### Query Analysis")
    display_query_box(prompt, interpreted_q_lg)
    
    col1, col2 = st.columns(2)
    
    if 'comp_lc_res' not in st.session_state:
        st.session_state.comp_lc_res = ""
        with col1:
            st.markdown("### LangChain Response")
            response_container_lc = st.empty()
            for chunk in call_chat_api(prompt, session_id + "_lc", "LangChain"):
                st.session_state.comp_lc_res += chunk
                response_container_lc.markdown(st.session_state.comp_lc_res + "▌")
            response_container_lc.markdown(st.session_state.comp_lc_res)
    else:
        with col1:
            st.markdown("### LangChain Response")
            st.markdown(st.session_state.comp_lc_res)
            
    full_res_lc = st.session_state.comp_lc_res

    with col2:
        st.warning("Human Review Required")
        edited_q = st.text_area("Edit query if needed:", value=interpreted_q_lg, height=80, key="comp_edit")
        
        if st.button("Proceed", key="comp_proceed"):
            st.markdown("### LangGraph Response")
            response_container_lg = st.empty()
            full_res_lg = ""
            for chunk in call_chat_api(prompt, session_id + "_lg", "LangGraph", use_human_review=True, edited_query=edited_q):
                full_res_lg += chunk
                response_container_lg.markdown(full_res_lg + "▌")
            response_container_lg.markdown(full_res_lg)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"**Comparison Mode:**\n\n**LangChain:** {full_res_lc}\n\n---\n\n**LangGraph:** {full_res_lg}"
            })
            clear_human_review_state()
            st.rerun()

def run_langchain(prompt: str, session_id: str):
    with st.chat_message("assistant"):
        st.markdown("**LangChain Mode**")
        response_container = st.empty()
        full_response = ""
        for chunk in call_chat_api(prompt, session_id, "LangChain"):
            full_response += chunk
            response_container.markdown(full_response + "▌")
        response_container.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": f"**LangChain Mode:**\n\n{full_response}"})

def run_langgraph_respond(prompt: str, session_id: str, interpreted_query: str):
    with st.chat_message("assistant"):
        st.markdown("**LangGraph Mode**")
        response_container = st.empty()
        full_response = ""
        for chunk in call_chat_api(prompt, session_id, "LangGraph", use_human_review=True, edited_query=interpreted_query):
            full_response += chunk
            response_container.markdown(full_response + "▌")
        response_container.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": f"**LangGraph Mode:**\n\n{full_response}"})

def run_comparison(prompt: str, session_id: str):
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    full_res_lc = ""
    with col1:
        st.markdown("### LangChain")
        response_container_lc = st.empty()
        for chunk in call_chat_api(prompt, session_id + "_lc", "LangChain"):
            full_res_lc += chunk
            response_container_lc.markdown(full_res_lc + "▌")
        response_container_lc.markdown(full_res_lc)
    
    full_res_lg = ""
    with col2:
        st.markdown("### LangGraph")
        response_container_lg = st.empty()
        for chunk in call_chat_api(prompt, session_id + "_lg", "LangGraph"):
            full_res_lg += chunk
            response_container_lg.markdown(full_res_lg + "▌")
        response_container_lg.markdown(full_res_lg)
        
    st.session_state.messages.append({
        "role": "assistant", 
        "content": f"**Comparison Mode:**\n\n**LangChain:** {full_res_lc}\n\n---\n\n**LangGraph:** {full_res_lg}"
    })

def admin_sidebar():
    st.sidebar.title("🔐 Admin Portal")
    
    if not st.session_state.admin_logged_in:
        email = st.sidebar.text_input("Email")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if email == ADMIN_CREDENTIALS["email"] and password == ADMIN_CREDENTIALS["password"]:
                st.session_state.admin_logged_in = True
                st.session_state.admin_user = ADMIN_CREDENTIALS["name"]
                st.success(f"Welcome {st.session_state.admin_user}")
                st.rerun()
            else:
                st.sidebar.error("Invalid credentials")
    else:
        st.sidebar.success(f"Logged in as: {st.session_state.admin_user}")
        if st.sidebar.button("Logout"):
            st.session_state.admin_logged_in = False
            st.session_state.admin_user = None
            st.rerun()
            
        st.sidebar.markdown("---")
        st.sidebar.subheader("Add Knowledge")
        category = st.sidebar.selectbox("Category", VALID_CATEGORIES)
        new_content = st.sidebar.text_area("Content to add", height=150)
        
        if st.sidebar.button("Add Content"):
            try:
                payload = {
                    "category": category,
                    "content": new_content,
                    "email": ADMIN_CREDENTIALS["email"],
                    "password": ADMIN_CREDENTIALS["password"]
                }
                with httpx.Client(timeout=120.0) as client:
                    res = client.post(f"{API_URL}/admin/add_content", json=payload)
                    if res.status_code == 200:
                        st.sidebar.success("Content added successfully!")
                    else:
                        st.sidebar.error(f"Failed: {res.text}")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
        
        if st.sidebar.button("Refresh Vector Store"):
            try:
                payload = {
                    "email": ADMIN_CREDENTIALS["email"],
                    "password": ADMIN_CREDENTIALS["password"]
                }
                with httpx.Client(timeout=120.0) as client:
                    res = client.post(f"{API_URL}/admin/refresh?category={category}", json=payload)
                    if res.status_code == 200:
                        st.sidebar.success(f"{category} store refreshed!")
                    else:
                        st.sidebar.error(f"Failed: {res.text}")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")

def main():
    st.title("🤖 Benny AI - Unified Bot")
    admin_sidebar()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        mode = st.radio("Select Mode:", ["LangChain", "LangGraph", "Comparison"], horizontal=True)
    with col2:
        if mode != "LangChain":
            st.session_state.human_review_toggle = st.toggle("Human Review", value=False)

    session_id = st.session_state.get("session_id", "default")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if st.session_state.human_review_waiting:
        if st.session_state.pending_mode == "Comparison":
            render_comparison_review(session_id)
        else:
            show_human_review_ui(mode)
        return
    
    if prompt := st.chat_input("Ask me anything about Bennett University..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        if mode == "LangChain":
            run_langchain(prompt, session_id)
        elif mode == "LangGraph":
            if st.session_state.human_review_toggle:
                interpreted_q = get_rewritten_query(prompt, session_id)
                st.session_state.pending_query = prompt
                st.session_state.pending_interpreted = interpreted_q
                st.session_state.pending_session_id = session_id
                st.session_state.pending_mode = "LangGraph"
                st.session_state.human_review_waiting = True
                st.rerun()
            else:
                run_langgraph_respond(prompt, session_id, prompt)
        else:
            if st.session_state.human_review_toggle:
                interpreted_q = get_rewritten_query(prompt, session_id)
                st.session_state.pending_query = prompt
                st.session_state.pending_interpreted = interpreted_q
                st.session_state.pending_session_id = session_id
                st.session_state.pending_mode = "Comparison"
                st.session_state.human_review_waiting = True
                st.rerun()
            else:
                run_comparison(prompt, session_id)

if __name__ == "__main__":
    main()
