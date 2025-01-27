import requests
import streamlit as st

st.set_page_config(layout="wide")

@st.cache_data
def get_hotels():
    """Return a list of hotels from the API."""
    api_endpoint = st.secrets["api"]["endpoint"]
    response = requests.get(f"{api_endpoint}/Hotels", timeout=30)
    
    # Debug info: show status code and raw text if not 200
    if response.status_code != 200:
        st.write(f"DEBUG (get_hotels): Error {response.status_code}")
        st.write("Raw response text:", response.text)
        return response  # or raise an exception

    return response


@st.cache_data
def get_hotel_bookings(hotel_id):
    """Return a list of bookings for the specified hotel."""
    api_endpoint = st.secrets["api"]["endpoint"]
    response = requests.get(f"{api_endpoint}/Hotels/{hotel_id}/Bookings", timeout=10)
    
    if response.status_code != 200:
        st.write(f"DEBUG (get_hotel_bookings): Error {response.status_code}")
        st.write("Raw response text:", response.text)
        return response

    return response


@st.cache_data
def invoke_chat_endpoint(question):
    """Invoke the chat endpoint with the specified question."""
    api_endpoint = st.secrets["api"]["endpoint"]
    response = requests.post(f"{api_endpoint}/Chat", data={"message": question}, timeout=10)
    
    if response.status_code != 200:
        st.write(f"DEBUG (invoke_chat_endpoint): Error {response.status_code}")
        st.write("Raw response text:", response.text)
        return response

    return response


def main():
    """Main function for the Chat with Data Streamlit app."""

    st.write(
    """
    # API Integration via Semantic Kernel

    This Streamlit dashboard is intended to demonstrate how we can use
    the Semantic Kernel library to generate SQL statements from natural language
    queries and display them in a Streamlit app.

    ## Select a Hotel
    """
    )

    # Display the list of hotels as a drop-down list
    hotels_response = get_hotels()
    if hotels_response.status_code == 200:
        try:
            hotels_json = hotels_response.json()
        except Exception as e:
            st.error(f"Failed to parse hotels JSON: {e}")
            return
    else:
        st.error("Failed to retrieve hotels list.")
        return

    # Reshape hotels to an object with hotelID and hotelName
    hotels = [{"id": hotel["hotelID"], "name": hotel["hotelName"]} for hotel in hotels_json]
    
    selected_hotel = st.selectbox("Hotel:", hotels, format_func=lambda x: x["name"])

    # Display the list of bookings for the selected hotel as a table
    if selected_hotel:
        hotel_id = selected_hotel["id"]
        bookings_response = get_hotel_bookings(hotel_id)
        if bookings_response.status_code == 200:
            try:
                bookings = bookings_response.json()
            except Exception as e:
                st.error(f"Failed to parse bookings JSON: {e}")
                return
            st.write("### Bookings")
            st.table(bookings)
        else:
            st.error("Failed to retrieve hotel bookings.")

    st.write(
        """
        ## Ask a Bookings Question

        Enter a question about hotel bookings in the text box below.
        Then select the "Submit" button to call the Chat endpoint.
        """
    )

    question = st.text_input("Question:", key="question")
    if st.button("Submit"):
        with st.spinner("Calling Chat endpoint..."):
            if question:
                chat_response = invoke_chat_endpoint(question)
                if chat_response.status_code == 200:
                    st.write(chat_response.text)
                    st.success("Chat endpoint called successfully.")
                else:
                    st.error("Failed to call chat endpoint.")
            else:
                st.warning("Please enter a question.")


if __name__ == "__main__":
    main()