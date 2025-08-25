import streamlit as st
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser



@st.cache_resource
def load_components():
    """
    Load all the necessary components for the RAG chain.
    This includes the vector database, the retriever, and the LLM.
    """
    print("Loading components...")
    
    # 1. Initialize the embedding model
    embedding_function = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

    # 2. Load the ChromaDB vector database
    db_client = chromadb.PersistentClient(path="./chroma_db")
    vector_store = Chroma(
        client=db_client,
        collection_name="supreme_court_judgments",
        embedding_function=embedding_function,
    )

    # 3. Create a retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 most relevant chunks

    # 4. Initialize the Groq LLM
    llm = ChatGroq(
        temperature=0, 
        model_name="llama3-8b-8192",
    )

    print("Components loaded successfully.")
    return retriever, llm

# --- THE RAG CHAIN ---


def create_rag_chain(retriever, llm):
    """
    Create the RAG chain using LangChain Expression Language (LCEL).
    """
    prompt_template = """
    You are a legal assistant. Your task is to answer the user's question based ONLY on the following context from Supreme Court of India judgments.
    If the context does not contain the answer, state that you cannot answer with the provided information.
    Do not add any information that is not in the context. Be concise and accurate.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    prompt = PromptTemplate.from_template(prompt_template)

    # The RAG chain pipeline
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


st.set_page_config(page_title="Legal Document Assistant", layout="wide")
st.title("⚖️ Legal Document Assistant with RAG")
st.write("Ask a question about a legal scenario, and the assistant will find relevant information from 10 years of Supreme Court of India judgments.")


try:
    retriever, llm = load_components()
    rag_chain = create_rag_chain(retriever, llm)
except Exception as e:
    st.error(f"Failed to load components: {e}")
    st.stop()


user_query = st.text_input("Enter your legal question:", placeholder="E.g., What are the principles for granting anticipatory bail?")

if st.button("Get Answer"):
    if not user_query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching for relevant judgments and generating an answer..."):
            
            answer = rag_chain.invoke(user_query)
            st.success("Answer Generated!")
            st.write(answer)
            st.write("---")
            st.subheader("Sources:")
            source_documents = retriever.get_relevant_documents(user_query)
            for i, doc in enumerate(source_documents):
                with st.expander(f"Source {i+1}: {doc.metadata.get('case_name', 'Unknown Case')}"):
                    st.write(f"**Section:** {doc.metadata.get('section', 'N/A')}")
                    st.write(f"**Source File:** {doc.metadata.get('source_file', 'N/A')}")
                    st.write("---")
                    st.write(doc.page_content)