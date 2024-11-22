# Import python packages
import pandas as pd
import streamlit as st
from snowflake.snowpark.context import get_active_session
from snowflake.cortex import Complete
import snowflake.snowpark.functions as F
import snowflake.snowpark.types as T

# Set Streamlit page configuration
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title(":truck: Tasty Bytes Support: Customer Q&A Assistant  :truck:")
st.caption(
    f"""Welcome! This application suggests answers to customer questions based 
    on corporate documentation and previous agent responses in support chats.
    """
)

# Get current credentials
session = get_active_session()

# Constants
CHAT_MEMORY = 20
DOC_TABLE = "app.vector_store"


# Reset chat conversation
def reset_conversation():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "What question do you need assistance answering?",
        }
    ]


##########################################
#       Select LLM
##########################################
with st.expander(":gear: Settings"):
    model = st.selectbox(
        "Change chatbot model:",
        [
            "mistral-large",
            "reka-flash",
            "llama2-70b-chat",
            "gemma-7b",
            "mixtral-8x7b",
            "mistral-7b",
        ],
    )
    st.button("Reset Chat", on_click=reset_conversation)


##########################################
#       RAG
##########################################
def get_context(chat, DOC_TABLE):
    chat_summary = summarize(chat)
    return find_similar_doc(chat_summary, DOC_TABLE)


def summarize(chat):
    summary = Complete(
        model,
        "Provide the most recent question with essential context from this support chat: "
        + chat,
    )
    return summary.replace("'", "")


def find_similar_doc(text, DOC_TABLE):
    doc = session.sql(f"""Select input_text,
                        source_desc,
                        VECTOR_COSINE_SIMILARITY(chunk_embedding, SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2', '{text.replace("'", "''")}')) as dist
                        from {DOC_TABLE}
                        order by dist desc
                        limit 1
                        """).to_pandas()
    st.info("Selected Source: " + doc["SOURCE_DESC"].iloc[0])
    return doc["INPUT_TEXT"].iloc[0]

##########################################
#       Prompt Construction
##########################################
if "background_info" not in st.session_state:
    st.session_state.background_info = (
        session.table("app.documents")
        .select("raw_text")
        .filter(F.col("relative_path") == "tasty_bytes_who_we_are.pdf")
        .collect()[0][0]
    )


def get_prompt(chat, context):
    prompt = f"""Answer this new customer question sent to our support agent
        at Tasty Bytes Food Truck Company. Use the background information
        and provided context taken from the most relevant corporate documents
        or previous support chat logs with other customers.
        Be concise and only answer the latest question.
        The question is in the chat.
        Chat: <chat> {chat} </chat>.
        Context: <context> {context} </context>.
        Background Info: <background_info> {st.session_state.background_info} </background_info>."""
    return prompt.replace("'", "")


##########################################
#       Chat with LLM
##########################################
if "messages" not in st.session_state:
    reset_conversation()

if user_message := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_message})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    chat = str(st.session_state.messages[-CHAT_MEMORY:]).replace("'", "")
    with st.chat_message("assistant"):
        with st.status("Answering..", expanded=True) as status:
            st.write("Finding relevant documents & support chat logs...")
            # Get relevant information
            context = get_context(chat, DOC_TABLE)
            st.write("Using search results to answer your question...")
            # Ask LLM
            prompt = get_prompt(chat, context)
            response = Complete(model, prompt)
            status.update(label="Complete!", state="complete", expanded=False)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})