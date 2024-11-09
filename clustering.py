# clustering.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st

def recursive_clustering(embeddings, idea_texts, chat_model, min_cluster_size=5, node_id=None, level=0, parent_labels=None, current_node_id=0):
    """
    Perform recursive clustering on embeddings and generate labels using LLM.

    Returns:
        nodes_data: List of node dictionaries.
        edges_data: List of edge dictionaries.
        current_node_id: The next available node ID (integer).
    """
    nodes_data = []
    edges_data = []

    # Define colors for different levels
    level_colors = ["#FF7F50", "#87CEFA", "#32CD32", "#BA55D3",
                    "#FFD700", "#FF69B4", "#CD5C5C", "#4B0082"]

    n_samples = len(embeddings)

    if node_id is None:
        # Root node
        cluster_label, cluster_description = generate_cluster_label(idea_texts, chat_model, parent_labels)
        cluster_node_id = f"cluster_{current_node_id}"
        node_data = {
            'id': cluster_node_id,
            'title': cluster_description.replace('\n', '<br>'),
            'label': cluster_label,
            'shape': 'ellipse',
            'color': level_colors[level % len(level_colors)],
            'font': {'size': 16, 'multi': True},
            'full_text': cluster_label,  # Store full text for markdown
            'description': cluster_description  # Store description
        }
        nodes_data.append(node_data)
        current_node_id += 1
        node_id = cluster_node_id  # Set node_id to this root node ID
    else:
        # Generate label and description for this cluster using LLM
        cluster_label, cluster_description = generate_cluster_label(idea_texts, chat_model, parent_labels)
        cluster_node_id = f"cluster_{current_node_id}"
        node_data = {
            'id': cluster_node_id,
            'title': cluster_description.replace('\n', '<br>'),
            'label': cluster_label,
            'shape': 'ellipse',
            'color': level_colors[level % len(level_colors)],
            'font': {'size': 16, 'multi': True},
            'full_text': cluster_label,  # Store full text for markdown
            'description': cluster_description  # Store description
        }
        nodes_data.append(node_data)
        current_node_id += 1
        # Add edge from parent node to this cluster node
        edge_data = {
            'source': node_id,
            'to': cluster_node_id  # Use 'to' instead of 'target'
        }
        edges_data.append(edge_data)
        node_id = cluster_node_id  # Update node_id to current cluster's node ID

    if n_samples <= min_cluster_size or n_samples <= 2:
        # Base case: create nodes for ideas
        for i, idea in enumerate(idea_texts):
            idea_node_id = f"idea_{current_node_id}"
            node_data = {
                'id': idea_node_id,
                'title': idea.replace('\n', '<br>'),
                'label': idea,
                'shape': 'box',
                'color': level_colors[level % len(level_colors)],
                'font': {'size': 16, 'multi': True},
                'full_text': idea  # Store full text for markdown
                # No description for idea nodes
            }
            nodes_data.append(node_data)
            edge_data = {
                'source': node_id,
                'to': idea_node_id  # Use 'to' instead of 'target'
            }
            edges_data.append(edge_data)
            current_node_id += 1
        return nodes_data, edges_data, current_node_id
    else:
        # Determine the optimal number of clusters using silhouette score
        optimal_k = determine_optimal_clusters(
            embeddings, min_k=2, max_k=min(10, n_samples - 1))
        if optimal_k < 2:
            # Cannot cluster further, proceed to base case
            for i, idea in enumerate(idea_texts):
                idea_node_id = f"idea_{current_node_id}"
                node_data = {
                    'id': idea_node_id,
                    'title': idea.replace('\n', '<br>'),
                    'label': idea,
                    'shape': 'box',
                    'color': level_colors[level % len(level_colors)],
                    'font': {'size': 16, 'multi': True},
                    'full_text': idea  # Store full text for markdown
                    # No description for idea nodes
                }
                nodes_data.append(node_data)
                edge_data = {
                    'source': node_id,
                    'to': idea_node_id  # Use 'to' instead of 'target'
                }
                edges_data.append(edge_data)
                current_node_id += 1
            return nodes_data, edges_data, current_node_id

        # Perform KMeans clustering
        try:
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            labels = kmeans.fit_predict(embeddings)
        except Exception as e:
            # If clustering fails, proceed to base case
            for i, idea in enumerate(idea_texts):
                idea_node_id = f"idea_{current_node_id}"
                node_data = {
                    'id': idea_node_id,
                    'title': idea.replace('\n', '<br>'),
                    'label': idea,
                    'shape': 'box',
                    'color': level_colors[level % len(level_colors)],
                    'font': {'size': 16, 'multi': True},
                    'full_text': idea  # Store full text for markdown
                    # No description for idea nodes
                }
                nodes_data.append(node_data)
                edge_data = {
                    'source': node_id,
                    'to': idea_node_id  # Use 'to' instead of 'target'
                }
                edges_data.append(edge_data)
                current_node_id += 1
            return nodes_data, edges_data, current_node_id

        # Update parent labels
        current_parent_labels = parent_labels.copy() if parent_labels else []
        current_parent_labels.append(cluster_label)

        # Group embeddings and idea_texts by cluster
        unique_labels = np.unique(labels)
        for cluster_num in unique_labels:
            cluster_indices = [
                idx for idx, label in enumerate(labels) if label == cluster_num]
            cluster_embeddings = [embeddings[idx] for idx in cluster_indices]
            cluster_idea_texts = [idea_texts[idx]
                                  for idx in cluster_indices]

            # Recursively cluster
            child_nodes_data, child_edges_data, current_node_id = recursive_clustering(
                cluster_embeddings,
                cluster_idea_texts,
                chat_model,
                min_cluster_size=min_cluster_size,
                node_id=node_id,  # Pass current cluster node ID as parent
                level=level + 1,
                parent_labels=current_parent_labels,
                current_node_id=current_node_id  # Pass the updated node ID
            )

            # Add child nodes and edges to the main lists
            nodes_data.extend(child_nodes_data)
            edges_data.extend(child_edges_data)

        return nodes_data, edges_data, current_node_id

def determine_optimal_clusters(embeddings, min_k=2, max_k=10):
    """
    Determine the optimal number of clusters using silhouette score.

    Args:
        embeddings: List of embeddings.
        min_k: Minimum number of clusters to try.
        max_k: Maximum number of clusters to try.

    Returns:
        The optimal number of clusters.
    """
    n_samples = len(embeddings)
    if n_samples <= 2:
        return 1  # Cannot cluster further

    min_k = max(2, min_k)
    max_k = min(max_k, n_samples - 1)

    if min_k > max_k:
        return 1  # Cannot cluster further

    best_k = 1
    best_score = -1
    for k in range(min_k, max_k + 1):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            if len(np.unique(labels)) < 2:
                continue  # Need at least 2 clusters for silhouette score
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_k = k
                best_score = score
        except Exception as e:
            # Catch exceptions such as when n_samples < n_clusters
            continue
    return best_k if best_score > -1 else 1  # Return 1 if no better k found

def generate_cluster_label(idea_texts_list, chat_model, parent_labels=None):
    """
    Generate a label and description for a cluster using the LLM.

    Args:
        idea_texts_list: List of idea texts in the cluster.
        chat_model: The LLM model to use.
        parent_labels: List of parent cluster labels.

    Returns:
        A tuple of (label string, description string) generated by the LLM.
    """
    # Prepare the prompt
    ideas_concatenated = '\n'.join(idea_texts_list[:50])  # Limit to first 50 ideas to control prompt length
    parent_labels_text = ', '.join(parent_labels) if parent_labels else ''

    prompt_template = """
You are an assistant that labels clusters of ideas. Given the following ideas:

{IDEAS}

Parent cluster labels: {PARENT_LABELS}

Considering the parent labels to avoid redundancy, provide:
1. A concise label (a few words) that summarizes the main theme of these ideas and is distinct from the parent labels.
2. A brief description (2-3 sentences) that captures the essence of this cluster.

Return your answer in JSON format **without any additional text**, and ensure that the output is ONLY the JSON and nothing else.

Example output:
{{
    "label": "Cluster Label",
    "description": "Cluster Description"
}}
"""

    # Use the LLM to generate the label and description
    chain = LLMChain(
        prompt=PromptTemplate.from_template(prompt_template),
        llm=chat_model
    )
    response = chain.run(IDEAS=ideas_concatenated, PARENT_LABELS=parent_labels_text)

    # Clean up the response and parse the JSON
    import json
    import re
    try:
        # Extract JSON object from the response using regex
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            response_json = json.loads(json_str)
            label = response_json.get('label', 'No Label')
            description = response_json.get('description', 'No Description')
        else:
            # If no JSON found, set default values
            label = "No Label"
            description = "No Description"
            st.error("LLM did not return JSON format as expected.")
            st.write("LLM Response:")
            st.write(response)
    except json.JSONDecodeError as e:
        label = "No Label"
        description = "No Description"
        st.error(f"Error parsing LLM response: {e}")
        st.write("LLM Response:")
        st.write(response)

    return label.strip(), description.strip()
