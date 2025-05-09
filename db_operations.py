import psycopg2
import json
from openai import OpenAI
import streamlit as st
from typing import List, Tuple, Optional
import numpy as np

# Initialize OpenAI client with API key from Streamlit secrets
client = OpenAI(api_key=st.secrets['API_KEY'])

def get_connection():
    """
    Establishes a connection to the Supabase PostgreSQL database using the URI.
    Utilizes Streamlit's session state to cache the connection for reuse.
    """
    if 'db_connection' not in st.session_state:
        try:
            connection = psycopg2.connect(st.secrets['DB'])
            st.session_state['db_connection'] = connection
        except psycopg2.Error as e:
            st.error(f"Database connection error: {e}")
            st.stop()
    return st.session_state['db_connection']

def add_idea(entity: str, source_title: str, source_url: str, source_text: str, idea_text: str, embedding: Optional[List[float]] = None):
    """
    Adds a new idea to the 'ideas' table in the database.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Convert embedding to JSON string to store it
        embedding_str = json.dumps(embedding) if embedding else None

        cursor.execute('''
            INSERT INTO ideas (entity, source_title, source_url, source_text, idea_text, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
        ''', (entity, source_title, source_url, source_text, idea_text, embedding_str))

        conn.commit()
    except psycopg2.Error as e:
        st.error(f"Error adding idea: {e}")
    finally:
        cursor.close()

def get_embedding_for_idea(idea_text: str) -> List[float]:
    """
    Generates an embedding for the given idea text using OpenAI's API.
    """
    try:
        response = client.embeddings.create(
            input=idea_text,
            model="text-embedding-3-small" 
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return []

@st.cache_data(ttl=120)
def get_ideas(entity: Optional[str] = None, with_embeddings: bool = False) -> List[Tuple]:
    """
    Efficiently retrieves ideas from the database. Can filter by entity and optionally include embeddings.
    """
    conn = None
    cursor = None
    results = []

    try:
        conn = psycopg2.connect(st.secrets['DB'], sslmode='require')
        cursor = conn.cursor()

        # Construct dynamic SELECT clause
        fields = "id, entity, source_title, source_url, source_text, idea_text"
        if with_embeddings:
            fields += ", embedding"

        if entity:
            cursor.execute(f"SELECT {fields} FROM ideas WHERE entity = %s", (entity,))
        else:
            cursor.execute(f"SELECT {fields} FROM ideas LIMIT 200")  # Safety limit

        rows = cursor.fetchall()

        for row in rows:
            if with_embeddings:
                # Reconstruct embedding from JSON string
                *main_fields, embedding_json = row
                embedding = json.loads(embedding_json) if embedding_json else None
                results.append((*main_fields, embedding))
            else:
                results.append(row)

    except Exception as e:
        st.error(f"Database error in get_ideas: {e}")

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

    return results

def delete_idea_by_id(idea_id: int):
    """
    Deletes an idea from the 'ideas' table based on its ID.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute('DELETE FROM ideas WHERE id = %s', (idea_id,))
        conn.commit()
        st.success(f"Idea ID {idea_id} deleted successfully.")
    except psycopg2.Error as e:
        st.error(f"Error deleting Idea ID {idea_id}: {e}")
    finally:
        cursor.close()

def update_embedding(idea_id: int, embedding: Optional[List[float]]):
    """
    Updates the embedding of a specific idea in the 'ideas' table.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Convert embedding to JSON string to store it
        embedding_str = json.dumps(embedding) if embedding else None

        cursor.execute('UPDATE ideas SET embedding = %s WHERE id = %s', (embedding_str, idea_id))
        conn.commit()
        st.success(f"Embedding for Idea ID {idea_id} updated successfully.")
    except psycopg2.Error as e:
        st.error(f"Error updating embedding for Idea ID {idea_id}: {e}")
    finally:
        cursor.close()

def save_cluster_hierarchy(entity: str, hierarchy: dict, markdown_text: str):
    """
    Saves or updates the cluster hierarchy and its corresponding markdown text for a given entity.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()

        hierarchy_json = json.dumps(hierarchy)

        cursor.execute('''
            INSERT INTO cluster_hierarchies (entity, hierarchy_json, markdown_text)
            VALUES (%s, %s, %s)
            ON CONFLICT (entity) DO UPDATE SET hierarchy_json = EXCLUDED.hierarchy_json, markdown_text = EXCLUDED.markdown_text
        ''', (entity, hierarchy_json, markdown_text))

        conn.commit()
    except psycopg2.Error as e:
        st.error(f"Error saving cluster hierarchy for entity '{entity}': {e}")
    finally:
        cursor.close()

@st.cache_data(ttl=120)
def load_cluster_hierarchy(entity: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Loads the cluster hierarchy and its corresponding markdown text for a given entity.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT hierarchy_json, markdown_text FROM cluster_hierarchies WHERE entity = %s', (entity,))
        row = cursor.fetchone()

        if row:
            hierarchy_json, markdown_text = row
            hierarchy = json.loads(hierarchy_json)
            return hierarchy, markdown_text
        else:
            st.info(f"No cluster hierarchy found for entity '{entity}'.")
            return None, None
    except psycopg2.Error as e:
        st.error(f"Error loading cluster hierarchy for entity '{entity}': {e}")
        return None, None
    finally:
        cursor.close()

def delete_cluster_hierarchy(entity: str):
    """
    Deletes the cluster hierarchy and its corresponding markdown text for a given entity.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM cluster_hierarchies WHERE entity = %s', (entity,))
        conn.commit()
        st.success(f"Cluster hierarchy for entity '{entity}' deleted successfully.")
    except psycopg2.Error as e:
        st.error(f"Error deleting cluster hierarchy for entity '{entity}': {e}")
    finally:
        cursor.close()

def get_embedding_for_query(query: str) -> List[float]:
    """
    Generates an embedding for a user's query using OpenAI's API.
    """
    # Reuse the embedding function for ideas
    return get_embedding_for_idea(query)

@st.cache_data(ttl=120)
def find_closest_ideas(entity: str, query_embedding: List[float], top_n: int = 5) -> List[Tuple]:
    """
    Finds the top N ideas closest to the query embedding, for a specific entity only.
    """
    ideas = get_ideas(entity=entity, with_embeddings=True)
    if not ideas:
        return []

    query_vec = np.array(query_embedding)
    similarities = []

    for idea in ideas:
        idea_embedding = idea[6]
        if not idea_embedding:
            continue
        idea_vec = np.array(idea_embedding)
        try:
            cosine_sim = np.dot(query_vec, idea_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(idea_vec))
            similarities.append((idea, cosine_sim))
        except ZeroDivisionError:
            continue

    if not similarities:
        return []

    similarities.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in similarities[:top_n]]
