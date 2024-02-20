from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter

def ai_template(data):
    import streamlit as st
    from langchain.callbacks.base import BaseCallbackHandler

    class StreamHandler(BaseCallbackHandler):
        
        def __init__(self, container, initial_text=""):
            self.container = container
            self.text = initial_text

        def on_llm_new_token(self, token: str, **kwargs):
            self.text += token
            self.container.markdown(self.text)

    
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    class CustomDataChatbot:

        def __init__(self):
            self.openai_api_key = st.secrets['openai_api_key']
            self.openai_model = "gpt-3.5-turbo"

        @st.spinner('Analyzing documents..')
        def setup_qa_chain(self, data):
            # Load documents
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200
            )
            # transform data to text for each row join details and venue
            data['text'] = data['Details'] + ' ' + data['Reservation_Venue']
            # now as a big string
            data = data['text'].str.cat(sep=' ')
            splits = text_splitter.split_text(data)

            # Create embeddings and store in vectordb
            embeddings = OpenAIEmbeddings(
                openai_api_key=self.openai_api_key
            )            
            vectordb = DocArrayInMemorySearch.from_texts(splits, embeddings)

            # Define retriever
            retriever = vectordb.as_retriever(
                search_type='mmr',
                search_kwargs={'k':2, 'fetch_k':4}
            )

            # Setup memory for contextual conversation        
            memory = ConversationBufferMemory(
                memory_key='chat_history',
                return_messages=True
            )

            # Setup LLM and QA chain
            llm = ChatOpenAI(model_name=self.openai_model, temperature=0, streaming=True, openai_api_key=self.openai_api_key)
            qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, verbose=True)
            return qa_chain

        def main(self):
            #decorator
            def enable_chat_history(func):
                # to clear chat history after swtching chatbot
                current_page = func.__qualname__
                if "current_page" not in st.session_state:
                    st.session_state["current_page"] = current_page
                if st.session_state["current_page"] != current_page:
                    try:
                        st.cache_resource.clear()
                        del st.session_state["current_page"]
                        del st.session_state["messages"]
                    except:
                        pass

                # to show chat history on ui
                if "messages" not in st.session_state:
                    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
                for msg in st.session_state["messages"]:
                    st.chat_message(msg["role"]).write(msg["content"])

                def execute(*args, **kwargs):
                    func(*args, **kwargs)
                return execute

            def display_msg(msg, author):
                """Method to display message on the UI

                Args:
                    msg (str): message to display
                    author (str): author of the message -user/assistant
                """
                st.session_state.messages.append({"role": author, "content": msg})
                st.chat_message(author).write(msg)

            
            # User Inputs
            enable_chat_history(self.main)

            # create 3 buttons for automatic questions
            button_1 = st.sidebar.button('Generate a Report', use_container_width=True)
            button_2 = st.sidebar.button('Show me the best reviews', use_container_width=True)
            button_3 = st.sidebar.button('Show me the worst reviews', use_container_width=True)
            
            user_query = st.chat_input(placeholder="Ask me anything!")

            if user_query and data is not None:
                with st.spinner('Thinking...'):
                    qa_chain = self.setup_qa_chain(data)
                    display_msg(user_query, 'user')
                    with st.chat_message("assistant"):
                        st_cb = StreamHandler(st.empty())
                    response = qa_chain.run(user_query, callbacks=[st_cb])
                    st.session_state.messages.append({"role": "assistant", "content": response})
            
            elif button_1:
                user_query = 'Generate an in depth report, highlight important points and patterns that emerge in the reviews. Give back a list of positive points and negative points'
                qa_chain = self.setup_qa_chain(data)

                display_msg(user_query, 'user')

                with st.chat_message("assistant"):
                    st_cb = StreamHandler(st.empty())
                    with st.spinner('Thinking...'):
                        response = qa_chain.run(user_query, callbacks=[st_cb])
                        st.session_state.messages.append({"role": "assistant", "content": response})

            elif button_2:
                user_query = 'Generate a list of the best reviews, found in the data.'
                qa_chain = self.setup_qa_chain(data)

                display_msg(user_query, 'user')

                with st.chat_message("assistant"):
                    st_cb = StreamHandler(st.empty())
                    with st.spinner('Thinking...'):
                        response = qa_chain.run(user_query, callbacks=[st_cb])
                        st.session_state.messages.append({"role": "assistant", "content": response})

            elif button_3:
                user_query = 'Generate a list of the worst reviews, found in the data.'

                qa_chain = self.setup_qa_chain(data)

                display_msg(user_query, 'user')

                with st.chat_message("assistant"):
                    st_cb = StreamHandler(st.empty())
                    with st.spinner('Thinking...'):
                        response = qa_chain.run(user_query, callbacks=[st_cb])
                        st.session_state.messages.append({"role": "assistant", "content": response})

            # create button to restart chat
            restart = st.sidebar.button('New Chat', use_container_width=True, type = 'primary')
            if restart:
                st.session_state['messages'] = []
                st.session_state['current_page'] = None

    obj = CustomDataChatbot()
    obj.main()

def ai_template(data):
    import os
    import tempfile
    import streamlit as st
    from langchain.chat_models import ChatOpenAI
    from langchain.document_loaders import PyPDFLoader
    from langchain.memory import ConversationBufferMemory
    from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain.chains import ConversationalRetrievalChain
    from langchain.vectorstores import DocArrayInMemorySearch
    from langchain.text_splitter import RecursiveCharacterTextSplitter


    @st.cache_resource(ttl="1h")
    def configure_retriever(uploaded_files):
        # Read documents
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_text(uploaded_files.to_string())

        # Create embeddings and store in vectordb
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = DocArrayInMemorySearch.from_texts(splits, embeddings)

        # Define retriever
        retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})

        return retriever


    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
            self.container = container
            self.text = initial_text
            self.run_id_ignore_token = None

        def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
            # Workaround to prevent showing the rephrased question as output
            if prompts[0].startswith("Human"):
                self.run_id_ignore_token = kwargs.get("run_id")

        def on_llm_new_token(self, token: str, **kwargs) -> None:
            if self.run_id_ignore_token == kwargs.get("run_id", False):
                return
            self.text += token
            self.container.markdown(self.text)


    class PrintRetrievalHandler(BaseCallbackHandler):
        def __init__(self, container):
            self.status = container.status("**Context Retrieval**")

        def on_retriever_start(self, serialized: dict, query: str, **kwargs):
            self.status.write(f"**Question:** {query}")
            self.status.update(label=f"**Context Retrieval:** {query}")

        def on_retriever_end(self, documents, **kwargs):
            for idx, doc in enumerate(documents):
                source = os.path.basename(doc.metadata["source"])
                self.status.write(f"**Document {idx} from {source}**")
                self.status.markdown(doc.page_content)
            self.status.update(state="complete")


    openai_api_key = st.secrets["openai_api_key"]
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    uploaded_files = data

    retriever = configure_retriever(uploaded_files)

    # Setup memory for contextual conversation
    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

    # Setup LLM and QA chain
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0, streaming=True
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=True
    )

    if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
        msgs.clear()
        msgs.add_ai_message("How can I help you?")

    avatars = {"human": "user", "ai": "assistant"}
    for msg in msgs.messages:
        st.chat_message(avatars[msg.type]).write(msg.content)

    if user_query := st.chat_input(placeholder="Ask me anything!"):
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            retrieval_handler = PrintRetrievalHandler(st.container())
            stream_handler = StreamHandler(st.empty())
            response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])