import streamlit as st
# COMPONENT - VIEWING HTML FOR WORKFLOW MONITOR
import streamlit.components.v1 as components

# HELPERS
import uuid
from pathlib import Path

# STORAGE
from pinecone import Pinecone
import llama_index
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext, Document, VectorStoreIndex, set_global_handler

# LLM
from llama_index.llms.anthropic import Anthropic
from anthropic import Anthropic as AnthropicClient, HUMAN_PROMPT, AI_PROMPT
ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]

# WORKFLOW
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
)

# WORKFLOW OBSERVABILITY
from llama_index.utils.workflow import draw_all_possible_flows

# OBSERVABILILITY & EVALUATION USING LLAMATRACE WITH ARIZE PHOENIX
PHOENIX_API_KEY = st.secrets["PHOENIX_API_KEY"]
OTEL_EXPORTER_OTLP_HEADERS = st.secrets["OTEL_EXPORTER_OTLP_HEADERS"]
PHOENIX_CLIENT_HEADERS = st.secrets["PHOENIX_CLIENT_HEADERS"]
PHOENIX_COLLECTOR_ENDPOINT = st.secrets["PHOENIX_COLLECTOR_ENDPOINT"]
llama_index.core.set_global_handler( #"arize_phoenix",
    project_name="llamaindex-workflow-pinecone-observability", endpoint="https://llamatrace.com/v1/traces"
)

# DEFINE EVENTS FOR WORKFLOW
class Brainstorming(Event):
    first_output: str

class DesignBrief(Event):
    second_output: str

class Flowchart(Event):
    third_output: str

class UserResearch(Event):
    fourth_output: str

class Wireframe(Event):
    fifth_output: str

class Design(Event):
    sixth_output: str  

class Prototype(Event):
    seventh_output: str

class Deploy(Event):
    eighth_output: str

# CUSTOM WORKFLOW
class ProductionWorkflow(Workflow):
    @step
    async def step_one(query, ev: StartEvent) -> Brainstorming:
        #st.write(ev.first_input)
        # Create a temporary document from the query
        temp_doc = Document(text=query)
        temp_index = VectorStoreIndex.from_documents([temp_doc], storage_context=storage_context)
        
        # Perform a similarity search
        retriever = temp_index.as_retriever(similarity_top_k=3)
        similar_docs = retriever.retrieve(query)
        
        if similar_docs:
            return "\n\n".join([f"Similar Project: {doc.text}" for doc in similar_docs])
        else:
            return "No similar projects found."
        
    @step
    async def step_two(self, ev: Brainstorming) -> DesignBrief:
        st.write(ev.first_output)
        return DesignBrief(second_output="Second step complete.")

    @step
    async def step_three(self, ev: DesignBrief) -> Flowchart:
        st.write(ev.second_output)
        return Flowchart(result="Third step complete.")
    
    @step
    async def step_four(self, ev: Flowchart) -> UserResearch:
        st.write(ev.third_output)
        return UserResearch(result="Fourth step complete.")

    @step
    async def step_five(self, ev: UserResearch) -> Wireframe:
        st.write(ev.fourth_output)
        return Wireframe(result="Fifth step complete.")

    @step
    async def step_six(self, ev: Wireframe) -> Design:
        st.write(ev.fifth_output)
        return Design(result="Sixth step complete.")
    
    @step
    async def step_seven(self, ev: Design) -> Prototype:
        st.write(ev.sixth_output)
        return Prototype(result="Seventh step complete.")

    @step
    async def step_eight(self, ev: Prototype) -> Deploy:
        st.write(ev.seventh_output)
        return Deploy(result="Eighth step complete.")

    @step
    async def step_nine(self, ev: Deploy) -> StopEvent:
        st.write(ev.eighth_output)
        return StopEvent(result="Workflow complete.")

# EXPORT WORKFLOW TO GRAPH
async def workflow():
    w = ProductionWorkflow(timeout=10, verbose=True)
    result = await w.run(first_input="Start the workflow.")
    st.sidebar.write(result)

if __name__ == "__workflow__":
    import asyncio
    asyncio.run(workflow())

# Load and display the HTML file
URL = "workflow.html"
st.sidebar.write("Workflow Graph")
with st.sidebar:
    with open(Path(URL), 'r') as f:
        html_content = f.read()
        draw_all_possible_flows(ProductionWorkflow, filename=URL)
        components.html(html_content, height=500, scrolling=True)

# ==========
# EMBEDDINGS
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

# Initialize Pinecone vector databasee with API Key and OpenAI embed model
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

embed_model = OpenAIEmbedding(
    model="text-embedding-3-large",
    dimensions=1536,
)

index_name = "llamaindex-docs"
if not PINECONE_API_KEY:
    st.error("Pinecone API key is not set. Please check your secrets file.")
    st.stop()









# ============
# Streamlit UI
st.title("LlamaIndex Workflow with Pinecone and Arize Phoenix")
st.write("Using OpenAI Embeddings as well as Anthopic's Claude and Mermaid to support workflow output.")

# Initialize components
@st.cache_resource
def init_components():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_name)
    
    # Initialize Anthropic LLM for text generation
    llm = Anthropic(api_key=ANTHROPIC_API_KEY)
    
    # Setup the vector store and storage context
    vector_store = PineconeVectorStore(pinecone_index=index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Initialize Anthropic client
    anthropic_client = AnthropicClient(api_key=ANTHROPIC_API_KEY)
    
    # Return for the initialization function
    return pc, index, llm, vector_store, storage_context, anthropic_client

# Call the initialization function
pc, index, llm, vector_store, storage_context, anthropic_client = init_components()

# Setup tabs for visual display of our workflow output
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Description", "Similar Projects", "Design Brief", "Flowchart", "User Research", "Journey Map"])

# Streamlit UI for user input
tab1.subheader("Enter your project idea")
query = tab1.text_area("Example: A streamlit app for tracking cryptocurrency prices", height=100)

# Custom Mermaid rendering function
with tab4:
    def render_mermaid(code: str) -> None:
        components.html(
            f"""
            <pre class="mermaid">
                {code}
            </pre>
            <script type="module">
                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                mermaid.initialize({{ startOnLoad: true }});
            </script>
            """,
            height=500,
        )

# Define the workflow function
def main_workflow(query):
    similar_projects = find_similar_projects(query)
    tab2.subheader("Similarity Search Results")
    tab2.write(similar_projects)

    project_brief = brainstorm_step(query)
    tab3.subheader("Project Design Brief")
    tab3.write(project_brief)

    flowchart = flowchart_step(project_brief)
    tab4.subheader("Flowchart and Recommendations")
    with tab4:
        st.write(flowchart)
        # Extract and render Mermaid diagram to tab 4
        mermaid_start = flowchart.find("```mermaid")
        mermaid_end = flowchart.find("```", mermaid_start + 10)
        if mermaid_start != -1 and mermaid_end != -1:
            mermaid_code = flowchart[mermaid_start+10:mermaid_end].strip()    
            render_mermaid(mermaid_code)
        else:
            st.error("Mermaid flowchart not found in the generated response.")
        return flowchart

    user = user_step(project_brief)
    tab5.subheader("Persona, Scenario, and User Interview Questions")
    tab5.write(user)

    journey = user_step(project_brief)
    tab6.subheader("User Journey")
    tab6.write(journey)
    with tab6:
        st.write(journey)
        # Extract and render Mermaid diagram to tab 4
        mermaid_start = journey.find("```mermaid")
        mermaid_end = journey.find("```", mermaid_start + 10)
        if mermaid_start != -1 and mermaid_end != -1:
            mermaid_code = journey[mermaid_start+10:mermaid_end].strip()    
            render_mermaid(mermaid_code)
        else:
            st.error("Mermaid journey map not found in the generated response.")
        return journey

    wireframe = user_step(project_brief, flowchart)
    tab7.subheader("Wireframe")
    tab7.write(wireframe)

    design = user_step(query, project_brief, flowchart, user, wireframe)
    tab8.subheader("Visual Design")
    tab8.write(design)

    prototype = user_step(project_brief, flowchart, user, wireframe, design)
    tab9.subheader("Prototype")
    tab9.write(prototype)
    

# WORKFLOW STEP FUNCTIONS
# Function to find similar projects
# START WORKFLOW | TAB 2 - SIMILAR PROJECTS
def find_similar_projects(query):
    # Create a temporary document from the query
    temp_doc = Document(text=query)
    temp_index = VectorStoreIndex.from_documents([temp_doc], storage_context=storage_context)
    
    # Perform a similarity search
    retriever = temp_index.as_retriever(similarity_top_k=3)
    similar_docs = retriever.retrieve(query)
    
    if similar_docs:
        tab2.write("Your idea might not be as original as you thought...")
        return "\n\n".join([f"- {doc.text}" for doc in similar_docs])
    else:
        tab2.write("Wow! Your idea seems like it might be quite original...")
        return "No similar projects found."


# STEP 2 | TAB 3 - BRAINSTORM FOR DESIGN BRIEF
def brainstorm_step(query):
    brainstorm_prompt = f"{HUMAN_PROMPT} Use '{query}' as the problem and define the solution by outlining experience highlighting pain points and explaining how your solutions resolve an issue, conflict or problem. Create and output a Project Design Brief with the following sections:\n\n1. Target Market\n2. Target Audience\n3. Competitors\n4. Project Description\n5. Technical Requirements\n6. Expected Outcome from using the product\n7. Estimated number of potential users\n8. Estimated number of potential business partners\n9. Expected revenue for first year in operation\n10. Explanation of monetization strategy\n\nPlease format your response as a structured document with clear headings for each section.{AI_PROMPT}"

    with tab3:
        try:
            # Generate a response using Anthropic LLM
            response = anthropic_client.completions.create(
                model="claude-2",
                prompt=brainstorm_prompt,
                max_tokens_to_sample=100
            )
            project_design_brief = response.completion
            
            # Store the project brief in Pinecone
            doc_id = str(uuid.uuid4())
            doc = Document(text=project_design_brief, id_=doc_id)
            VectorStoreIndex.from_documents([doc], storage_context=storage_context)
            return project_design_brief
        except Exception as e:
            st.error(f"An error occurred while generating the project brief: {str(e)}")
            return None

# STEP 3 | TAB 4 - FLOWCHART
def flowchart_step(project_design_brief):
    flowchart_prompt = f"{HUMAN_PROMPT} Based on the following Project Design Brief, please:\n\n1. Create a Mermaid flowchart describing the basic architecture of the project.\n2. Provide recommendations or suggestions on other features or considerations that might be useful.\n\nProject Design Brief:\n{project_design_brief}\n\nPlease format your response in two sections:\n1. Mermaid Flowchart\n2. Recommendations and Suggestions\n\nFor the Mermaid flowchart, use the following syntax:\n```mermaid\ngraph TD\n    A[Start] --> B[Process]\n    B --> C[End]\n```\n\nReplace the example with an appropriate flowchart for the project.{AI_PROMPT}"

    with tab4:
        try:
            # Generate flowchart response
            response = anthropic_client.completions.create(
                model="claude-2",
                prompt=flowchart_prompt,
                max_tokens_to_sample=1000
            )
            return response.completion
        except Exception as e:
            st.error(f"An error occurred while generating the flowchart: {str(e)}")
            return None

# STEP 4  | TAB 5 - USER RESEARCH
def user_step(project_design_brief):
    user_prompt = f"{HUMAN_PROMPT} Based on the following Project Design Brief, please:\n\n1. Create a Persona\n\n2. Create a day in the life scenario for the Persona to describe the problem the application will solve highlighting the pain points of the experience.\n\n3. Create a list of 10 questions for a user interview for the persona. Ask these questions to strategically balance both quantitative and qualitative aspects of user research principles.\n\nProject Design Brief:\n{project_design_brief}\n\nPlease format your response in three sections:\n1. Persona\n2. Scenario\n3. Interview\n\n"

    with tab5:
        try:
            # Generate response
            response = anthropic_client.completions.create(
                model="claude-2",
                prompt=user_prompt,
                max_tokens_to_sample=1000
            )
            return response.completion
        except Exception as e:
            st.error(f"An error occurred while generating the persona, scenario, and user interview questions: {str(e)}")
            return None

# STEP 5  | TAB 6 - USER JOURNEY
def journey_step(project_design_brief):
    journey_prompt = f"{HUMAN_PROMPT} Based on the following Project Design Brief, please:\n\n1. Create a user journey\n\nProject Design Brief:\n{project_design_brief}\n\n"

    with tab6:
        try:
            # Generate response
            response = anthropic_client.completions.create(
                model="claude-2",
                prompt=journey_prompt,
                max_tokens_to_sample=300
            )
            return response.completion
        except Exception as e:
            st.error(f"An error occurred while generating the user journey: {str(e)}")
            return None

# RUN WORKFLOW BUTTON
with tab1:
    if st.button("Run Workflow"):
        with st.spinner("Running workflow..."):
            result = main_workflow(query)
        if result:
            st.success("Workflow completed successfully!")
        else:
            st.error("Workflow failed to complete. Please check the error messages.")

