import requests
import streamlit as st

st.set_page_config(layout="wide")

def handle_query_vectorization(query):
    """Vectorize the query using the Vectorize endpoint."""
    api_endpoint = st.secrets["api"]["endpoint"]
    response = requests.get(
        f"{api_endpoint}/Vectorize",
        params={"text": query},
        timeout=10,
        verify=False
    )
    return response.text  # We'll just return the raw text, e.g. a JSON list of floats

def handle_vector_search(query_vector, max_results=5, minimum_similarity_score=0.8):
    """Perform a vector search using the VectorSearch endpoint."""
    api_endpoint = st.secrets["api"]["endpoint"]
    headers = {"Content-Type": "application/json"}

    # 'query_vector' here is a string from the handle_query_vectorization call.
    # The endpoint expects a JSON float[] array, so just pass the string directly
    # if it's already in the correct format or do any needed parsing.

    response = requests.post(
        f"{api_endpoint}/VectorSearch",
        data=query_vector,  # The body is the vector string
        params={
            "max_results": max_results,
            "minimum_similarity_score": minimum_similarity_score
        },
        headers=headers,
        timeout=10,
        verify=False
    )
    return response

def main():
    """Main function for the Vector Search over Maintenance Requests Streamlit page."""

    st.write(
        """
        # Vector Search for Maintenance Requests

        This Streamlit dashboard demonstrates how to perform a vector search
        to find maintenance requests similar to a user query.

        ## Enter a Maintenance Request query
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        query = st.text_input("Search query:", key="query")
    with col2:
        # 'max_results' is a text input, so parse to int (default 5 if blank)
        max_results_str = st.text_input("Max results (<=0 will return all results):", key="max_results", value="0")
        try:
            max_results = int(max_results_str)
        except ValueError:
            max_results = 0

    minimum_similarity_score = st.slider("Minimum Similarity Score:", min_value=0.0, max_value=1.0, value=0.8, step=0.01)

    if st.button("Submit"):
        with st.spinner("Performing vector search..."):
            if query:
                # TODO #4: Vectorize the query text
                query_vector = handle_query_vectorization(query)

                # TODO #5: Get the vector search results
                vector_search_results = handle_vector_search(query_vector, max_results, minimum_similarity_score)

                # Display the results
                st.write("## Results")
                # TODO #6: Display the results as a table
                st.table(vector_search_results.json())
            else:
                st.warning("Please enter a query.")

if __name__ == "__main__":
    main()