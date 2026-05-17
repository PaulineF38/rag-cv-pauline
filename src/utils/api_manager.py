def get_llm(provider: str, api_key: str):
    """
    Initializes and returns the LLM configured according to the chosen provider.
    """
    if provider == "OpenAI":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.3)
        
    elif provider == "MistralAI":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model="mistral-large-latest", api_key=api_key, temperature=0.3)
        
    elif provider == "Google Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.3)
        
    else:
        raise ValueError(f"Provider '{provider}' is not supported.")