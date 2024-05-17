import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate

# Set the environment variable for the Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCkLcgdJwKwGOydaL3dxfJIcopEc7PXYWI"

# Initialize Model
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Streamlit UI
st.title("Ask me")

url = st.text_input("Enter the URL of the website:")
user_query = st.text_input("Enter your query:")

if url and user_query:
    with st.spinner('Loading website content and generating response...'):
        try:
            # Load the website content
            loader = WebBaseLoader(url)
            docs = loader.load()

            # Extract the text content from the document
            text_content = "\n".join([doc.page_content for doc in docs])

            # Define the Query Chain
            template = """Based on the following content, answer the query:
            Content: "{text}"
            Query: "{query}"
            Answer:"""

            prompt = PromptTemplate(
                input_variables=["text", "query"],
                template=template
            )

            llm_chain = LLMChain(llm=llm, prompt=prompt)

            # Prepare documents and query
            context = {"text": text_content, "query": user_query}

            # Invoke Chain
            response = llm_chain.invoke(context)
            answer = response["text"]

            # Display the answer
            st.subheader("Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")
