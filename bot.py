import os
import json
import openai
import faiss
import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document

load_dotenv()

# Define some helper functions
def extract_restjson_content(text):
    start_tag = "<RESTJSON>"
    end_tag = "</RESTJSON>"

    # Find the positions of the start and end tags
    start_index = text.find(start_tag) + len(start_tag)
    end_index = text.find(end_tag)

    # If both tags are found, extract the substring
    if start_index != -1 and end_index != -1 and start_index < end_index:
        return text[start_index:end_index].strip()
    else:
         raise ValueError("The input text does not contain valid <RESTJSON> and </RESTJSON> tags.")

# Function to load recipes from the 'rezepte' folder and convert to documents
def load_recipes_as_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r") as file:
                try:
                    recipe = json.load(file)
                    recipe = json.loads(recipe)
                    # Ensure the recipe is a dictionary
                    if not isinstance(recipe, dict):
                        print(f"Skipping {filename}: Not a valid JSON object")
                        continue

                    # Ensure instructions are a list, if not convert it to a list
                    instructions = recipe.get('properties').get('instructions', [])
                    if isinstance(instructions, str):
                        instructions = [instructions]  # Convert string to list

                    # Convert each recipe to a document for embedding
                    content = (
                        f"Beschreibung: {recipe.get('properties').get('description', 'No description')}\n\n"
                        f"Portionen: {recipe.get('properties').get('servings', 'Unknown')}\n\n"
                        f"Zutaten: {', '.join([ingredient.get('name', 'Unknown') for ingredient in recipe.get('properties').get('ingredients', [])])}\n\n"
                        f"Anweisungen: {' '.join(instructions)}"
                    )
                    doc = Document(page_content=content, metadata={"name": recipe.get('properties').get('name', 'Unknown')})
                    documents.append(doc)
                except json.JSONDecodeError:
                    print(f"Error reading {filename}, skipping.")
    
    # Print the number of documents loaded
    print(f"Loaded {len(documents)} documents from {folder_path}")
    return documents

# Load and convert recipes into documents
if "documents" not in st.session_state:
    st.session_state.documents = load_recipes_as_documents("rezepte")

# Check if documents are loaded before creating the vector store
if len(st.session_state.documents) == 0:
    st.error("No recipes found. Please check if the 'rezepte' folder contains valid JSON files.")
else:
    # Initialize OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Initialize FAISS vector store and embeddings
    if "vector_store" not in st.session_state:
        # Embed the documents
        embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
        
        # Embed only if documents exist
        try:
            vector_store = FAISS.from_documents(st.session_state.documents, embeddings)
            st.session_state.vector_store = vector_store
        except IndexError as e:
            st.error(f"Error embedding documents: {e}")
            print(f"Error embedding documents: {e}")

# Initialize Streamlit session state for memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=False)

# Initialize the ChatGPT model with LangChain
chat_model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai.api_key)

# Define a prompt template for querying the model
prompt_template = ChatPromptTemplate.from_template(
    '''You are a bot that takes all ingredients of a meal as input and outputs a json. 
        The input will be in German. Do not tell the User you're making a JSON. The JSON Response will be stripped before the user can see it. You're here to help them get all the information for a meal. If the description of the meal is missing, create a new description yourself.
        Ask about missing information, confirm that all info is taken then output the JSON with the prefix <RESTJSON> and suffix </RESTJSON>. 
        Here's what happened so far: \n {history} \n
        Every REST json must follow the following structure:
        
        {{
        "title": "Meal",
        "type": "object",
        "required": ["name", "ingredients", "servings"],
        "properties": {{
            "name": {{
            "type": "string",
            "description": "The name of the meal."
            }},
            "description": {{
            "type": "string",
            "description": "A brief description of the meal."
            }},
            "servings": {{
            "type": "integer",
            "minimum": 1,
            "description": "The number of servings this meal makes."
            }},
            "ingredients": {{
            "type": "array",
            "minItems": 1,
            "items": {{
                "type": "object",
                "required": ["name", "quantity", "unit"],
                "properties": {{
                "name": {{
                    "type": "string",
                    "description": "The name of the ingredient."
                }},
                "quantity": {{
                    "type": "number",
                    "description": "The quantity of the ingredient."
                }},
                "unit": {{
                    "type": "string",
                    "description": "The unit of measurement for the ingredient."
                }}
                }}
            }}
            }},
            "instructions": {{
            "type": "array",
            "description": "List of instructions for preparing the meal.",
            "items": {{
                "type": "string"
            }}
            }}
        }}
        }}
        
        Here's the user input: {user_input}'''
)
# Prompt to detect if input is new recipe or not
input_prompt = ChatPromptTemplate.from_template(
'''Analyze the following statement: {user_input}
If it seems that the user wants to search old recipes with questions like "what was that spaghetti dish" or "How much of an ingredient for the recipe xy" then return "search"
If it seems that the user wants to create a new recipe with input like "I just cooked something" or "I want to add a new recipe" or just a list of ingredients and instructions then return "recipe"
Do only return "search" or "recipe" do not give any other output. If you are uncertain, return "search".'''
)


# Create a chain with the prompt and chat model
conversation_chain = LLMChain(
    llm=chat_model,
    memory=st.session_state.memory,  # Use the memory from session state
    prompt=prompt_template,
    verbose=True
)

input_chain = LLMChain(
    llm=chat_model,
    prompt=input_prompt
)

# Streamlit app layout and logic
st.title("Rezepte-Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if u_input := st.chat_input("Was hast du gekocht oder suchst du nach einem Rezept?"):
    with st.chat_message("user"):
        st.markdown(u_input)
    st.session_state.messages.append({"role": "user", "content": u_input})

    input_val = input_chain.run({"user_input": u_input})

    # First, check if the user is asking about recipes in the vector store
    if "vector_store" in st.session_state:
        vector_search_results = st.session_state.vector_store.similarity_search(u_input)

        if vector_search_results and input_val == "search":
            # If recipes are found, display the result
            response = "Hier sind einige Rezepte, die zu Ihrer Anfrage passen:\n"
            for result in vector_search_results:
                response += f"- {result.metadata['name']}: {result.page_content}\n"
        else:
            # Otherwise, continue with the regular conversation chain
            response = conversation_chain.run({"user_input": u_input})
            if "<RESTJSON>" in response:
                try:
                    recipe = extract_restjson_content(response)

                    random_id = str(uuid.uuid4())

                    with open(f"rezepte/{random_id}.json", "w") as json_file:
                        json.dump(recipe, json_file, indent=4)
                    response = response.replace(recipe, "").strip()
                    response = response.replace("<RESTJSON>", "").strip()
                    response = response.replace("</RESTJSON>", "").strip()
                    response = "Danke, das Rezept wurde gespeichert! \n" + response
                except ValueError as e:
                    print(f"Error: {e}")

        # Display the assistant's response in the chat
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.error("Vector store not initialized. Please check your recipe folder.")
