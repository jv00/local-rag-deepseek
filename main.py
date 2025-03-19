import streamlit as st
from typing import List, Dict, Optional
from utils import upload_files_to_db
from graph import graph

# Constants
APP_TITLE = "DeepSeek R1"
ALLOWED_FILE_TYPES = ["pdf"]
LAYOUT_CONFIG = {"page_title": "DeepSeek Local", "layout": "wide"}

class ChatState:
    """Manages session state initialization and updates"""
    
    @staticmethod
    def initialize():
        """Initialize session state variables if they don't exist"""
        defaults = {
            "selected_files_ready": False,
            "files_upload_complete": False,
            "messages": [],
            "uploader_key": 0
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @staticmethod
    def clear_messages():
        """Clear chat messages"""
        st.session_state.messages = []

def generate_response(user_input: str) -> Dict[str, str]:
    """Generate AI response based on user input"""
    initial_state = {"question": user_input}
    response = graph.invoke(initial_state)
    return response["answer"]

def render_header() -> None:
    """Render the application header and controls"""
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title(APP_TITLE)
    with col2:
        if st.button("Clear Chat", use_container_width=True):
            ChatState.clear_messages()
            st.rerun()

def handle_file_upload() -> Optional[List]:
    """Handle file upload functionality in sidebar"""
    selected_files = st.sidebar.file_uploader(
        "Upload PDF files",
        type=ALLOWED_FILE_TYPES,
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if selected_files:
        st.session_state.selected_files_ready = True
        st.session_state.files_upload_complete = False
        return selected_files
    return None

def process_uploaded_files(selected_files: List) -> None:
    """Process and upload selected files"""
    if (st.session_state.selected_files_ready and 
        not st.session_state.files_upload_complete):
        
        upload_button_placeholder = st.sidebar.empty()
        
        with upload_button_placeholder.container():
            if st.button("Upload Files", use_container_width=True):
                with st.status("Uploading files...", expanded=False) as status:
                    if upload_files_to_db(selected_files):
                        st.session_state.files_upload_complete = True
                        st.session_state.selected_files_ready = False
                        st.session_state.uploader_key += 1
                    status.update(
                        label="Files uploaded successfully!",
                        state="complete",
                        expanded=False
                    )

def display_chat_history() -> None:
    """Display chat message history"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def handle_user_input() -> None:
    """Handle user input and display responses"""
    if user_input := st.chat_input("Type your question here..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Generate and display assistant response
        assistant_response = generate_response(user_input)
        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_response
        })

        with st.chat_message("assistant"):
            st.write("#### Reasoning:")
            st.write(assistant_response.get('reasoning', 'No reasoning provided'))
            
            st.write("#### Response:")
            st.write(assistant_response.get('response', 'No response generated'))

def main() -> None:
    """Main application function"""
    st.set_page_config(**LAYOUT_CONFIG)
    ChatState.initialize()
    
    render_header()
    selected_files = handle_file_upload()
    
    if selected_files:
        process_uploaded_files(selected_files)
    
    display_chat_history()
    handle_user_input()

if __name__ == "__main__":
    main()