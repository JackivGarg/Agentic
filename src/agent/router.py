import time
from langchain_core.output_parsers import StrOutputParser

from src.llm import llm
from src.agent.prompts import router_template, generate_template
from src.agent.states.states import Route1
from src.langchain.history import get_session_history
from src.vectorstores import load_vector_store
from src.config import VALID_CATEGORIES


from src.langchain.utils import rewrite_query


def langgraph_route_and_respond(query: str, session_id: str, use_human_review: bool = False, edited_query: str = None):
    start_time = time.time()
    
    history = get_session_history(session_id)
    history_messages = history.messages
    
    history_str = "\n".join(
        f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}"
        for m in history_messages
    )
    
    original_query = query
    interpreted_query = rewrite_query(query, history_str) if not use_human_review else (edited_query or query)
    
    if use_human_review and edited_query:
        interpreted_query = edited_query
    
    try:
        router_chain = router_template | llm.with_structured_output(Route1)
        route_result = router_chain.invoke({
            "user_input": interpreted_query,
            "history": history_str,
            "context": ""
        })
        route_result_action = route_result.action
        category = route_result.category if route_result.action == "VECTOR_STORE" else "general"
        
        if category not in VALID_CATEGORIES or category is None:
            category = "general"
    except Exception as e:
        route_result_action = "VECTOR_STORE"
        category = "general"
    
    try:
        vs = load_vector_store(category)
        docs = vs.similarity_search(interpreted_query, k=3)
        context = "\n\n".join(doc.page_content for doc in docs)
    except Exception as e:
        context = f"Could not retrieve from {category}: {str(e)}"
    
    generate_chain = generate_template | llm | StrOutputParser()
    
    full_response = ""
    for chunk in generate_chain.invoke({
        "context": context,
        "history": history_str,
        "user_input": interpreted_query
    }):
        full_response += chunk
        yield chunk
    
    history.add_user_message(query)
    history.add_ai_message(full_response)
    
    response_time = time.time() - start_time
    
    stats = {
        "original_query": original_query,
        "interpreted_query": interpreted_query,
        "action": route_result_action,
        "category": category,
        "response_time": round(response_time, 2),
        "word_count": len(full_response.split()),
        "char_count": len(full_response),
        "was_edited": use_human_review and edited_query is not None,
        "human_approved": use_human_review
    }
    
    yield {"__stats__": stats}
