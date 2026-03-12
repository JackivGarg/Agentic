from pydantic import BaseModel, Field
from src.llm import llm
from src.langchain.prompts import query_rewriter_template

class RewrittenQuery(BaseModel):
    query: str = Field(description="The standalone rewritten question without any preamble.")

def rewrite_query(query: str, history_str: str) -> str:
    """
    Rewrites the user query into a standalone question using conversation history.
    """
    if not history_str.strip():
        return query

    structured_llm = llm.with_structured_output(RewrittenQuery)
    rewriter_chain = query_rewriter_template | structured_llm
    
    try:
        interpreted = rewriter_chain.invoke({
            "user_input": query,
            "history": history_str
        })
        
        result = interpreted.query.strip()
        
        # Fallback to original query if rewriter returns empty or weirdly small string
        if not result or len(result) < 2:
            return query
            
        return result
    except Exception as e:
        print(f"Error rewriting query: {e}")
        return query
