from langchain_core.tools import tool

@tool
def get_system_time() -> str:
    """Returns the current system time."""
    import datetime
    return str(datetime.datetime.now())