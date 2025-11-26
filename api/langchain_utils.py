from typing import List, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class LlamaIndexRetrieverWrapper(BaseRetriever):
    """Wraps a LlamaIndex retriever to work with LangChain."""
    index: Any
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query."""
        # Use the LlamaIndex retriever
        retriever = self.index.as_retriever(similarity_top_k=3)
        response = retriever.retrieve(query)
        
        # Convert LlamaIndex nodes to LangChain documents
        documents = []
        for node in response:
            # Extract content and metadata
            content = node.get_content()
            metadata = node.metadata
            documents.append(Document(page_content=content, metadata=metadata))
            
        return documents

def get_conversational_rag_chain(llm, index):
    """
    Creates a conversational RAG chain using LangChain.
    
    Args:
        llm: The LangChain LLM instance (e.g., ChatGroq).
        index: The LlamaIndex index instance.
        
    Returns:
        A LangChain Runnable that takes {"input": str, "chat_history": List[BaseMessage]} 
        and returns a dictionary with "answer" and "context".
    """
    
    # 1. Define the Retriever
    retriever = LlamaIndexRetrieverWrapper(index=index)
    
    # 2. Contextualize Question Prompt (Rephrase Query)
    # This prompt rephrases the follow-up question into a standalone question
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    
    # Create the history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # 3. Answer Question Prompt
    # This prompt uses the retrieved context to answer the question
    qa_system_prompt = (
        "Bạn là Lucfin, trợ lý dinh dưỡng thông minh. "
        "Sử dụng các đoạn ngữ cảnh được truy xuất sau đây để trả lời câu hỏi. "
        "Nếu bạn không biết câu trả lời, hãy nói rằng bạn không biết. "
        "Luôn trả lời bằng tiếng Việt.\n\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    
    # Create the document chain (stuff documents into prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # 4. Create the final Retrieval Chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain
