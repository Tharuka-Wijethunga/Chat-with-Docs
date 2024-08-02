import panel as pn
from chatbot import cbfs
import os

# Check for Mistral AI API key
if "MISTRAL_API_KEY" not in os.environ:
    raise ValueError("Please set the MISTRAL_API_KEY environment variable")

# Initialize Panel
pn.extension()

# Create chatbot instance
cb = cbfs()

# Create input widgets
file_input = pn.widgets.FileInput(accept='.pdf')
load_button = pn.widgets.Button(name="Load PDF", button_type='primary')
clear_button = pn.widgets.Button(name="Clear History", button_type='warning')
text_input = pn.widgets.TextInput(placeholder='Enter your question here...')

# Create output panes
conversation_pane = pn.pane.Markdown("")
db_response_pane = pn.pane.Markdown("")

# Define callback functions
def load_pdf(event):
    if file_input.value is not None:
        file_input.save("temp.pdf")
        result = cb.load_db("temp.pdf")
        conversation_pane.object = result
    else:
        conversation_pane.object = "Please select a PDF file first."

def clear_history(event):
    cb.clr_history()
    conversation_pane.object = "Chat history cleared."
    db_response_pane.object = ""

def send_message(event):
    question = text_input.value
    if question:
        if cb.vectorstore is None:
            conversation_pane.object += "\nPlease load a PDF file first."
        else:
            result = cb(question)
            conversation_pane.object += f"\nUser: {question}\nChatBot: {result['answer']}\n"
            db_response_pane.object = "Source Documents:\n" + "\n".join([doc.page_content[:200] + "..." for doc in result['source_documents']])
        text_input.value = ''

# Connect callbacks
load_button.on_click(load_pdf)
clear_button.on_click(clear_history)
text_input.param.watch(send_message, 'value')

# Create layout
layout = pn.Column(
    pn.pane.Markdown("# ChatWithDocs Bot (using Mistral AI)"),
    pn.Row(file_input, load_button),
    clear_button,
    text_input,
    pn.Tabs(
        ("Conversation", conversation_pane),
        ("Source Documents", db_response_pane)
    )
)

# Serve the application
layout.servable()