# # app.py

import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from nltk.tokenize import sent_tokenize
from prompts import *
import nltk
import json
from db_operations import *
from clustering import *
import re
import openai
import numpy as np

from sklearn.decomposition import PCA
import plotly.express as px

import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from collections import defaultdict
import numpy as np

nltk.download('punkt')

# Initialize session state variables
if 'nodes_data' not in st.session_state:
    st.session_state['nodes_data'] = None
if 'edges_data' not in st.session_state:
    st.session_state['edges_data'] = None
if 'chat_messages' not in st.session_state:
    st.session_state['chat_messages'] = []
if 'top_n_interact' not in st.session_state:
    st.session_state['top_n_interact'] = 5  # Default value

# Initialize session state variables for Tabs 4 and 5
if 'synthesis_report' not in st.session_state:
    st.session_state['synthesis_report'] = ""
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []
if 'synthesis_report_tab5' not in st.session_state:
    st.session_state['synthesis_report_tab5'] = ""

# Split text into manageable sentence-based chunks
def split_text_into_chunks(text, max_length=6000):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 > max_length:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
        else:
            current_chunk += sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# Function to clean up raw output for parsing
def clean_output(output):
    match = re.search(r'{.*}', output, re.DOTALL)
    if match:
        return match.group(0)
    return output  # Return as-is if no match found

# Initialize API Key securely
API_KEY = st.secrets['API_KEY']

st.title('IdeaTree')

# Add tabs, inserting "Natural Language Hierarchies" between "Embedding Visualization" and "Interact with a Tree"
tab1, tab2, tab_new, tab3, tab4 = st.tabs(["Idea Extraction", "Embedding Visualization", "Natural Language Hierarchies", "Interact with a Tree", "Have Two Trees Interact"])

### **Tab 1: Idea Extraction**
with tab1:
    st.header("Enter Sources")

    source_data = []
    num_sources = st.number_input("Number of Sources", min_value=1, value=1, step=1)
    
    # Collect user input for each source
    for i in range(int(num_sources)):
        with st.expander(f"Source {i + 1}", expanded=True):
            entity = st.text_input(f"Entity for Source {i + 1}", key=f"entity_{i}")
            title_url = st.text_input(f"Title/URL for Source {i + 1}", key=f"title_url_{i}")
            text_content = st.text_area(f"Text Content for Source {i + 1}", height=150, key=f"text_content_{i}")
            source_data.append({
                "entity": entity.strip(),
                "title_url": title_url.strip(),
                "text_content": text_content.strip()
            })

    if st.button('Generate'):
        if not source_data:
            st.error("Please enter at least one source.")
        else:
            # Initialize the LLM
            try:
                chat_model = ChatOpenAI(openai_api_key=API_KEY, model_name='gpt-4o-2024-08-06')  # Changed to a valid model name
            except Exception as e:
                st.error(f"Error initializing ChatOpenAI: {e}")
                st.stop()
            
            all_sources_ideas = {}
            total_chunks = sum(len(split_text_into_chunks(source["text_content"])) for source in source_data)
            progress_bar = st.progress(0)
            current_chunk_count = 0

            for source in source_data:
                entity = source["entity"]
                title = source["title_url"]
                text_content = source["text_content"]
                source_ideas = {}

                text_chunks = split_text_into_chunks(text_content)
                for idx, chunk in enumerate(text_chunks):
                    try:
                        chain = LLMChain(
                            prompt=PromptTemplate.from_template(idea_extractor),
                            llm=chat_model
                        )
                        
                        output = chain.run({"RAW_TEXT": chunk, "IDEAS_SO_FAR": source_ideas})
                        cleaned_output = clean_output(output)

                        ideas = json.loads(cleaned_output)
                        if isinstance(ideas, dict):
                            source_ideas.update(ideas)
                        else:
                            st.error(f"Output in chunk {idx + 1} of source '{title}' is not a dictionary.")
                    except json.JSONDecodeError as e:
                        st.error(f"Error parsing output in chunk {idx + 1} of source '{title}': {e}")
                    except Exception as e:
                        st.error(f"Error processing chunk {idx + 1} of source '{title}': {e}")
                    
                    current_chunk_count += 1
                    progress_bar.progress(current_chunk_count / total_chunks)
            
                all_sources_ideas[title] = source_ideas
                
                # Generate embeddings for each idea and save to database
                for idea_text in source_ideas.values():
                    try:
                        embedding = get_embedding_for_idea(idea_text)
                        add_idea(entity, title, title, text_content, idea_text, embedding)
                    except Exception as e:
                        st.error(f"Error adding idea '{idea_text}': {e}")
                
                st.subheader(f'Ideas for Source: {title}')
                st.json(source_ideas)
            
            st.success("Ideas generated and saved to the database successfully.")
            st.subheader('All Extracted Ideas Across Sources')
            st.json(all_sources_ideas)

### **Tab 2: Embedding Visualization**
with tab2:
    st.header("Hierarchical model of ideas")

    # Retrieve the list of entities
    all_ideas = get_ideas()
    entities = list(set([idea[1] for idea in all_ideas]))
    if not entities:
        st.info("No entities found in the database. Please generate some ideas first.")
    else:
        # Use session state to store selected_entity
        if 'selected_entity' not in st.session_state:
            st.session_state['selected_entity'] = entities[0]

        # Select an entity
        selected_entity = st.selectbox("Select Entity", entities, index=entities.index(st.session_state['selected_entity']), key='selected_entity')

        # Set clustering parameters
        # Removed the user input for Q and hardcoded it as 5
        min_cluster_size = 5  # Hardcoded value

        # Add a checkbox to toggle hierarchical config option
        hierarchical_config_toggle = st.checkbox("Enable Hierarchical Configuration", value=True, key='hierarchical_config_toggle')

        # Option to toggle node labels
        show_node_labels = st.checkbox("Show Node Labels", value=True)

        # Function to load hierarchy from database and update session state
        def load_hierarchy():
            hierarchy, markdown_text = load_cluster_hierarchy(selected_entity)
            if hierarchy:
                st.session_state['nodes_data'] = hierarchy['nodes']
                st.session_state['edges_data'] = hierarchy['edges']
                st.session_state['markdown_text'] = markdown_text
            else:
                st.session_state['nodes_data'] = None
                st.session_state['edges_data'] = None
                st.session_state['markdown_text'] = None

        # Load hierarchy when the app starts or when selected_entity changes
        if 'selected_entity_prev' not in st.session_state or st.session_state['selected_entity_prev'] != selected_entity:
            st.session_state['selected_entity_prev'] = selected_entity
            load_hierarchy()

        # Add a 'Run Clustering' button
        if st.button('Run Clustering', key='run_clustering'):
            # Remove existing data from session_state if override is selected
            if 'override_existing' in st.session_state and st.session_state['override_existing']:
                st.session_state.pop('nodes_data', None)
                st.session_state.pop('edges_data', None)
                st.session_state.pop('markdown_text', None)
                delete_cluster_hierarchy(selected_entity)

            # Perform clustering
            # Retrieve ideas and embeddings for the selected entity
            ideas = get_ideas(entity=selected_entity)
            if not ideas:
                st.error(f"No ideas found for entity '{selected_entity}'.")
            else:
                # Collect embeddings and corresponding idea texts
                idea_texts = [idea[5] for idea in ideas if idea[6] is not None]
                embeddings = [idea[6] for idea in ideas if idea[6] is not None]

                if embeddings:
                    # Ensure all embeddings are of the same length
                    embedding_lengths = [len(embed) for embed in embeddings]
                    if len(set(embedding_lengths)) != 1:
                        st.write(set(embedding_lengths))
                        st.error("Embeddings have varying lengths. Cannot perform clustering.")
                    else:
                        try:
                            embeddings_array = np.array(embeddings)

                            # Initialize the LLM
                            try:
                                chat_model = ChatOpenAI(openai_api_key=API_KEY, model_name='gpt-4o-2024-08-06')  # Changed to a valid model name
                            except Exception as e:
                                st.error(f"Error initializing ChatOpenAI: {e}")
                                st.stop()

                            # Perform recursive clustering
                            nodes_data, edges_data, _ = recursive_clustering(
                                embeddings_array.tolist(),
                                idea_texts,
                                chat_model,
                                min_cluster_size=min_cluster_size,
                                node_id=None,  # Start with no parent node
                                current_node_id=0  # Start node ID counter at 0
                            )

                            # Save the cluster hierarchy to the database
                            hierarchy = {'nodes': nodes_data, 'edges': edges_data}

                            # Reconstruct Node and Edge objects from saved data
                            nodes = []
                            nodes_dict = {}  # For accessing node attributes by ID
                            for node_data in nodes_data:
                                # Save full_text and description separately
                                full_text = node_data.get('full_text', node_data.get('label', ''))
                                description = node_data.get('description', '')
                                node_obj = Node(**node_data)
                                # Truncate label for visualization
                                node_obj.label = node_obj.label[:50]
                                # Add full_text and description attributes to node_obj
                                node_obj.full_text = full_text
                                node_obj.description = description
                                nodes.append(node_obj)
                                nodes_dict[node_obj.id] = node_obj

                            edges = []
                            for edge_data in edges_data:
                                source = edge_data.get('source')
                                to = edge_data.get('to')
                                if source is None or to is None:
                                    st.error("Edge data missing 'source' or 'to'")
                                    continue
                                # Pass 'target' instead of 'to' to the Edge constructor
                                edge_obj = Edge(source=source, target=to)
                                edges.append(edge_obj)

                            # Build the tree structure
                            tree = defaultdict(list)
                            for edge in edges:
                                tree[edge.source].append(edge.to)

                            # Identify root nodes
                            all_node_ids = set(node.id for node in nodes)
                            child_node_ids = set(edge.to for edge in edges)
                            root_ids = all_node_ids - child_node_ids

                            # Function to build the hierarchical markdown
                            def build_hierarchical_markdown(node_id, tree, nodes_dict, level=0):
                                node = nodes_dict[node_id]
                                label = getattr(node, 'full_text', node.label).strip()
                                description = getattr(node, 'description', '').strip()
                                markdown = ''
                                indent = '    ' * level
                                if node.shape == 'ellipse':  # Cluster node
                                    markdown += f"{indent}- **{label}**\n"
                                    if description:
                                        markdown += f"{indent}    — *{description}*\n"
                                else:
                                    markdown += f"{indent}- {label}\n"
                                for child_id in tree.get(node_id, []):
                                    markdown += build_hierarchical_markdown(child_id, tree, nodes_dict, level + 1)
                                return markdown

                            if root_ids:
                                markdown = ''
                                for root_id in root_ids:
                                    markdown += build_hierarchical_markdown(root_id, tree, nodes_dict)
                                # Save markdown to session state
                                st.session_state['markdown_text'] = markdown
                            else:
                                st.error("Could not determine the root node for hierarchical display.")
                                st.session_state['markdown_text'] = ''

                            # Save the cluster hierarchy and markdown text to the database
                            save_cluster_hierarchy(selected_entity, hierarchy, st.session_state['markdown_text'])

                            # Update session_state
                            st.session_state['nodes_data'] = nodes_data
                            st.session_state['edges_data'] = edges_data

                        except Exception as e:
                            st.error(f"Error during clustering: {e}")
                            # Remove nodes_data and edges_data from session_state if clustering fails
                            st.session_state.pop('nodes_data', None)
                            st.session_state.pop('edges_data', None)
                            st.session_state.pop('markdown_text', None)
                else:
                    st.error("No embeddings found to visualize.")
                    st.session_state.pop('nodes_data', None)
                    st.session_state.pop('edges_data', None)
                    st.session_state.pop('markdown_text', None)

        # Now, check if 'nodes_data' and 'edges_data' are in st.session_state
        if 'nodes_data' in st.session_state and 'edges_data' in st.session_state and st.session_state['nodes_data'] and st.session_state['edges_data']:
            nodes_data = st.session_state['nodes_data']
            edges_data = st.session_state['edges_data']
            if nodes_data and edges_data:
                # Reconstruct Node and Edge objects from saved data
                nodes = []
                nodes_dict = {}  # For accessing node attributes by ID
                for node_data in nodes_data:
                    # Save full_text and description separately
                    full_text = node_data.get('full_text', node_data.get('label', ''))
                    description = node_data.get('description', '')
                    node_obj = Node(**node_data)
                    # Truncate label for visualization
                    node_obj.label = node_obj.label[:50]
                    # Add full_text and description attributes to node_obj
                    node_obj.full_text = full_text
                    node_obj.description = description
                    nodes.append(node_obj)
                    nodes_dict[node_obj.id] = node_obj

                edges = []
                for edge_data in edges_data:
                    source = edge_data.get('source')
                    to = edge_data.get('to')
                    if source is None or to is None:
                        st.error("Edge data missing 'source' or 'to'")
                        continue
                    # Pass 'target' instead of 'to' to the Edge constructor
                    edge_obj = Edge(source=source, target=to)
                    edges.append(edge_obj)

                # Build the tree structure
                tree = defaultdict(list)
                for edge in edges:
                    tree[edge.source].append(edge.to)

                # Identify root nodes
                all_node_ids = set(node.id for node in nodes)
                child_node_ids = set(edge.to for edge in edges)
                root_ids = all_node_ids - child_node_ids

                # Function to build the hierarchical markdown
                # Load markdown from session_state if available
                markdown = st.session_state.get('markdown_text', '')

                if markdown:
                    # Display the markdown
                    with st.expander("Hierarchical Ideas List"):
                        st.markdown(markdown)
                else:
                    st.error("No markdown text available.")

                # Create the Config object with your specified settings
                config = Config(
                    width=1900,  # Increased width
                    height=800,
                    directed=True,
                    physics= not hierarchical_config_toggle,
                    hierarchical=hierarchical_config_toggle,
                    levelSeparation=2000,
                    nodeSpacing=500,
                    treeSpacing=10,
                    blockShifting=True,
                    edgeMinimization=True,
                    parentCentralization=True,
                    direction='DU',
                    sortMethod='directed',
                    shakeTowards='roots',
                    node={'labelProperty': 'label'},
                    link={'labelProperty': 'label', 'renderLabel': False},
                    font={'size': 16}
                )

                # Optionally remove labels from nodes
                if not show_node_labels:
                    for node in nodes:
                        node.label = ''

                # Display the graph
                agraph(nodes=nodes, edges=edges, config=config)
            else:
                st.error("No cluster hierarchy to display.")
        else:
            st.info("No cluster hierarchy loaded. Please run clustering first.")

### **Tab New: Natural Language Hierarchies**
with tab_new:
    st.header("Natural Language Hierarchies")

    # Retrieve the list of entities
    all_ideas = get_ideas()
    entities = list(set([idea[1] for idea in all_ideas]))
    if not entities:
        st.info("No entities found in the database. Please generate some ideas first.")
    else:
        # Select an entity
        selected_entity = st.selectbox("Select Entity for Natural Language Hierarchy", entities, key='natural_hierarchy_select')

        # Load the cluster hierarchy and markdown
        hierarchy, markdown_text = load_cluster_hierarchy(selected_entity)
        if not hierarchy:
            st.info(f"No cluster hierarchy found for entity '{selected_entity}'. Please run clustering first.")
        else:
            # Display the hierarchical ideas list without an expander
            # st.subheader("Hierarchical Ideas List")
            st.markdown(markdown_text)

            # Optionally, display the graph similar to Embedding Visualization
            nodes_data = hierarchy['nodes']
            edges_data = hierarchy['edges']

### **Tab 3: Interact with Tree**
with tab3:
    st.header("Interact with Tree")

    # Retrieve the list of entities
    all_ideas = get_ideas()
    entities = list(set([idea[1] for idea in all_ideas]))
    if not entities:
        st.info("No entities found in the database. Please generate some ideas first.")
    else:
        # Select an entity
        selected_entity = st.selectbox("Select Entity for Interaction", entities, key='interact_selected_entity')

        # Load the cluster hierarchy and markdown
        hierarchy, markdown_text = load_cluster_hierarchy(selected_entity)
        if not hierarchy:
            st.info(f"No cluster hierarchy found for entity '{selected_entity}'. Please run clustering first.")
        else:
            # Display the hierarchical ideas list
            with st.expander("Hierarchical Ideas List"):
                st.markdown(markdown_text)

            st.session_state['top_n_interact'] = 4

            # Button to clear and refresh chat history
            if st.button("Clear Chat History"):
                st.session_state['chat_messages'] = []
                st.rerun()

            # Chat interface
            st.subheader("Chat with the Tree")

            # Display chat messages from history
            for message in st.session_state['chat_messages']:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Accept user input (ensure it's not inside any prohibited container)
            user_input = st.chat_input("Ask about the tree...")

            if user_input:
                # Display user message
                with st.chat_message("user"):
                    st.markdown(user_input)
                # Add user message to chat history
                st.session_state['chat_messages'].append({"role": "user", "content": user_input})

                # Compute embedding for the user query
                try:
                    query_embedding = get_embedding_for_query(user_input)
                except Exception as e:
                    st.write(e)
                    st.error(f"Error generating embedding for the query: {e}")
                    st.session_state['chat_messages'].append({"role": "assistant", "content": f"Sorry, I couldn't process your request: {e}"})
                    st.rerun()

                # Retrieve the number of closest ideas from session state
                top_n = st.session_state['top_n_interact']

                # Find top N closest ideas
                try:
                    closest_ideas = find_closest_ideas(selected_entity, query_embedding, top_n=top_n)
                except Exception as e:
                    st.error(f"Error finding closest ideas: {e}")
                    closest_ideas = []

                # Format the closest ideas for the system message
                closest_ideas_text = "\n".join(f"{idx + 1}. {idea[5]}" for idx, idea in enumerate(closest_ideas))  # Assuming idea_text is in the 6th column

                # Format the system message using the template
                system_message = system_message_template.format(
                    markdown_text=markdown_text,
                    closest_ideas=closest_ideas_text
                )

                # Prepare messages for OpenAI
                messages = [
                    {"role": "system", "content": system_message},
                ]

                # Append the chat history
                for msg in st.session_state['chat_messages']:
                    messages.append({"role": msg["role"], "content": msg["content"]})

                # Initialize OpenAI Chat API
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-2024-08-06",  # Changed to a valid model name
                        messages=messages,
                        temperature=0.3,
                        frequency_penalty=0.5,  # Added to reduce repetition
                        presence_penalty=0.5     # Added to encourage diversity
                    )
                    assistant_message = response.choices[0].message.content
                except Exception as e:
                    st.error(f"Error communicating with OpenAI: {e}")
                    assistant_message = "Sorry, I encountered an error while processing your request."

                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(assistant_message)

                # Add assistant message to chat history
                st.session_state['chat_messages'].append({"role": "assistant", "content": assistant_message})
                st.rerun()

### **Tab 4: Entity Conversation**
with tab4:
    st.header("Entity Conversation")
    
    # Step 1: Select two entities
    all_ideas = get_ideas()
    entities = list(set([idea[1] for idea in all_ideas]))
    
    if len(entities) < 2:
        st.error("At least two entities are required to initiate a conversation.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            entity1 = st.selectbox("Select Entity 1", entities, key='entity1_select')
        with col2:
            entity2 = st.selectbox("Select Entity 2", entities, key='entity2_select')
        
        if entity1 == entity2:
            st.warning("Selected entities are the same. Please select two different entities for the conversation.")
        else:
            # Step 2: Set Conversation Objective
            conversation_objective = st.text_input(
                "Set Conversation Objective/Topic/Prompt", 
                value="Comprehensively explore the conceptual relationship between your ideas, drawing heavily on your idea trees.", 
                key='conversation_objective_input'
            )
            
            # Step 3: Define Number of Rounds
            num_rounds = st.number_input(
                "Number of Back-and-Forth Rounds", 
                min_value=1, 
                max_value=20, 
                value=3, 
                step=1, 
                key='num_rounds_input'
            )
            
            # Step 4: Start Conversation
            if st.button("Start Conversation"):
                st.session_state['conversation_history'] = []
                st.session_state['synthesis_report'] = ""
                
                # Retrieve hierarchical ideas for both entities
                hierarchy1, markdown_text1 = load_cluster_hierarchy(entity1)
                hierarchy2, markdown_text2 = load_cluster_hierarchy(entity2)
                
                if not hierarchy1 or not hierarchy2:
                    st.error("Both entities must have their cluster hierarchies loaded. Please ensure both have been clustered.")
                else:
                    # Initialize conversation history
                    conversation = []
                    
                    # Determine total responses (each back-and-forth has two responses)
                    total_responses = num_rounds * 2
                    
                    # Start the conversation with Entity1
                    current_entity = entity1
                    other_entity = entity2
                    current_markdown = markdown_text1
                    other_markdown = markdown_text2
                    
                    for response_num in range(1, total_responses + 1):
                        round_num = (response_num + 1) // 2
                        entity_turn = current_entity
                        
                        st.write(f"**Round {round_num}: {entity_turn} responds**")
                        
                        # Find the most recent message to get context for embeddings
                        if conversation:
                            last_message = conversation[-1]['content']
                            related_ideas = find_closest_ideas(
                                entity=current_entity, 
                                query_embedding=get_embedding_for_query(last_message), 
                                top_n=3  # Adjust as needed
                            )
                            closest_ideas_text = "\n".join(f"{idx + 1}. {idea[5]}" for idx, idea in enumerate(related_ideas))
                        else:
                            # If no prior message, no related ideas
                            closest_ideas_text = "No prior messages to derive related ideas from."
                        
                        # Prepare system message for the current entity
                        system_message = entity_system_message_template.format(
                            entity_name=current_entity,
                            other_entity_name=other_entity,
                            conversation_objective=conversation_objective,
                            responses_remaining=(num_rounds - ((response_num + 1) // 2)),
                            your_markdown_text=current_markdown,
                            closest_ideas=closest_ideas_text
                        )
                        
                        # Prepare messages for OpenAI
                        messages = [
                            {"role": "system", "content": system_message},
                        ]
                        
                        # Append the conversation history up to now, adjusting roles
                        for msg in conversation:
                            if msg['entity'] == other_entity:
                                # Messages from the other entity are 'user' messages
                                messages.append({"role": "user", "content": msg["content"]})
                            else:
                                # Messages from the current entity are 'assistant' messages
                                messages.append({"role": "assistant", "content": msg["content"]})

                        # Create the chat completion
                        try:
                            response = client.chat.completions.create(
                                model="gpt-4o-2024-08-06",  # Changed to a valid model name
                                messages=messages,
                                temperature=0.3,
                                frequency_penalty=0.5,
                                presence_penalty=0.5
                            )
                            assistant_message = response.choices[0].message.content
                        except Exception as e:
                            st.error(f"Error communicating with OpenAI for {current_entity}: {e}")
                            assistant_message = f"Error: {e}"
                        
                        # Display the message
                        if current_entity == entity1:
                            # Represent Entity1 as 'user'
                            with st.chat_message("user"):
                                st.markdown(assistant_message)
                        else:
                            # Represent Entity2 as 'assistant'
                            with st.chat_message("assistant"):
                                st.markdown(assistant_message)
                        
                        # Append the message to conversation history with the entity
                        conversation.append({"entity": current_entity, "content": assistant_message})
                        
                        # Swap entities for next response
                        current_entity, other_entity = other_entity, current_entity
                        current_markdown, other_markdown = other_markdown, current_markdown
                    
                    # Save the conversation history
                    st.session_state['conversation_history'] = conversation
                    
                    # Synthesize and evaluate the conversation
                    st.write("**Synthesizing Conversation Report...**")
                    
                    # Prepare the synthesizer system message
                    conversation_text = "\n".join([f"{msg['entity']}: {msg['content']}" for msg in conversation])
                    
                    synthesizer_system_message = synthesizer_system_message_template.format(
                        entity1_name=entity1,
                        entity2_name=entity2,
                        conversation_history=conversation_text
                    )
                    
                    # Prepare messages for synthesizer
                    synthesizer_messages = [
                        {"role": "system", "content": synthesizer_system_message},
                        {"role": "user", "content": "Please provide an overview and evaluation of the above conversation."}
                    ]
                    
                    # Create the chat completion for synthesis
                    try:
                        synth_response = client.chat.completions.create(
                            model="gpt-4o-2024-08-06",  # Changed to a valid model name
                            messages=synthesizer_messages,
                            temperature=0.3,
                            frequency_penalty=0.5,
                            presence_penalty=0.5
                        )
                        synthesis = synth_response.choices[0].message.content
                    except Exception as e:
                        st.error(f"Error communicating with OpenAI for synthesis: {e}")
                        synthesis = "Sorry, I encountered an error while processing the conversation synthesis."
                    
                    # Save the synthesis report
                    st.session_state['synthesis_report'] = synthesis
                    
                    # Display the synthesis report in an expander
                    with st.expander("Synthesis Report"):
                        st.markdown(synthesis)
