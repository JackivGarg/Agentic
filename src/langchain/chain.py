import time
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from operator import itemgetter

from src.llm import llm
from src.langchain.prompts import LANCHAIN_PROMPT
from src.langchain.history import get_session_history
from src.vectorstores import load_vector_store
from src.config import VALID_CATEGORIES
from src.embeddings import embedding
from src.langchain.utils import rewrite_query
from src.agent.states.states import Route1




def route_to_category(query: str, history_str: str):
    try:
        router_chain = router_template | llm.with_structured_output(Route1)
        route_result = router_chain.invoke({
            "user_input": query,
            "history": history_str,
            "context": ""
        })
        category = route_result.category if route_result.action == "VECTOR_STORE" else "general"
        if category not in VALID_CATEGORIES or category is None:
            category = "general"
        return category, route_result.action
    except Exception as e:
        return "general", "VECTOR_STORE"


def search_specific_vectorstore(query: str, category: str, k: int = 3):
    try:
        vs = load_vector_store(category)
        docs = vs.similarity_search(query, k=k)
        return docs
    except Exception as e:
        return []


def langchain_mode(query: str, session_id: str):
    start_time = time.time()
    
    history = get_session_history(session_id)
    history_messages = history.messages
    
    history_str = "\n".join(
        f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}"
        for m in history_messages
    )
    
    original_query = query
    interpreted_query = rewrite_query(query, history_str)
    
    category, action = route_to_category(interpreted_query, history_str)
    
    docs = search_specific_vectorstore(interpreted_query, category, k=3)
    
    if not docs:
        full_response = "Sorry, I couldn't find any relevant information."
        for chunk in [full_response]:
            yield chunk
        
        history.add_user_message(query)
        history.add_ai_message(full_response)
        
        response_time = time.time() - start_time
        yield {"__stats__": {
            "original_query": original_query,
            "interpreted_query": interpreted_query,
            "action": action,
            "category": category,
            "response_time": round(response_time, 2),
            "word_count": len(full_response.split()),
            "char_count": len(full_response)
        }}
        return
    
    context = "\n\n".join(doc.page_content for doc in docs)
    
    parser = StrOutputParser()
    
    pro_chain = (
        {
            "context": lambda x: context,
            "user_input": itemgetter("user_input"),
            "history": itemgetter("history")
        }
        | LANCHAIN_PROMPT
        | llm
        | parser
    )
    
    full_response = ""
    for chunk in pro_chain.invoke({"user_input": interpreted_query, "history": history_str}):
        full_response += chunk
        yield chunk
    
    history.add_user_message(query)
    history.add_ai_message(full_response)
    
    response_time = time.time() - start_time
    
    stats = {
        "original_query": original_query,
        "interpreted_query": interpreted_query,
        "action": action,
        "category": category,
        "response_time": round(response_time, 2),
        "word_count": len(full_response.split()),
        "char_count": len(full_response)
    }
    
    yield {"__stats__": stats}
