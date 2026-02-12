from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Useful for calculating math expressions. Input should be a mathematical expression string like '2 + 2'."""
    try:
        # Warning: eval is dangerous in production, using for simple demo purposes only
        return str(eval(expression))
    except Exception as e:
        return f"Error calculating: {e}"