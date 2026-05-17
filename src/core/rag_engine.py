import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate
from src.utils.api_manager import get_llm


def query_rag(user_question: str, provider: str, api_key: str, persist_directory: str = "data/chroma_db") -> str:
    """
    Executes the complete RAG chain using the local vector database.
    """
    if not os.path.exists(persist_directory):
        return "Error: The vector database could not be found."
    
    try:

        # load embeding model (the one use for ingestion)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # load database
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 3})
        
        # =========== DEBUG ===============
        
        docs = retriever.invoke(user_question)
        print("\n" + "="*60, flush=True)
        print(" DEBUG RAG : CONTENU ENVOYÉ AU LLM DEPUIS LA BDD", flush=True)
        print("="*60, flush=True)
        if not docs:
            print("Database return zero document, BDD empty", flush=True)
        for i, doc in enumerate(docs):
            print(f"📄 Chunk {i+1} (Source: {doc.metadata.get('source', 'Inconnue')}):", flush=True)
            print(f"{doc.page_content[:300]}...", flush=True)
            print("-" * 40, flush=True)
        print("="*60 + "\n", flush=True)
        # ====================================
        
        # Fetch the configured LLM from the utility manager
        llm = get_llm(provider, api_key)

        # prompt
        system_prompt = (
            "You are Pauline's virtual assistant. Your role is to answer recruiters' questions "
            "based exclusively on the provided context below (her CV and professional experiences).\n"
            "CRITICAL: Respond in the SAME language used by the recruiter in their question (e.g., if they ask in English, "
            "respond in English; if they ask in French, respond in French). Professionally translate the facts from the CV if necessary.\n"
            "Be professional, friendly, clear, and concise. If the answer cannot be found in the context, "
            "politely state that you do not have this information.\n\n"
            "Context (Pauline's CV):\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # Assemble and invoke the RAG chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        response = rag_chain.invoke({"input": user_question})
        return response["answer"]

    except Exception as e:
        return f"An error occurred while generating the response: {str(e)}"