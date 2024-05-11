import streamlit as st
import uuid
from query import executor
from query.generator import generate_sql

st.title("NL TO SQL Using AWS Bedtock and Athena")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

if prompt := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    sql, thinking_steps = generate_sql(prompt=prompt)

    data = executor.execute_sql(query=sql, report_id=str(uuid.uuid4()))

    st.write(str(thinking_steps))
    st.write(str(sql))

    st.dataframe(data=data)

    prompt = ""
