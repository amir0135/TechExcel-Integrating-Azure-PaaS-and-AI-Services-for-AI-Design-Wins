import json
import time
import re
import uuid
import streamlit as st
from scipy.io import wavfile
import azure.cognitiveservices.speech as speechsdk
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import ExtractiveSummaryAction, AbstractiveSummaryAction
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import openai


st.set_page_config(layout="wide")


############################
# New function for vector search in Cosmos
############################
def make_cosmos_db_vector_search_request(query_embedding, max_results=5, minimum_similarity_score=0.5):
    """Perform a vector distance query in Cosmos DB, returning transcripts
    with VectorDistance(c.request_vector, @request_vector) above a threshold."""
    
    # Retrieve Cosmos secrets
    cosmos_client_id = st.secrets["cosmos"]["client_id"]
    cosmos_credentials = DefaultAzureCredential(managed_identity_client_id=cosmos_client_id)

    cosmos_endpoint = st.secrets["cosmos"]["endpoint"]
    cosmos_database_name = st.secrets["cosmos"]["database_name"]
    cosmos_container_name = "CallTranscripts"

    # Create CosmosClient with MSI credentials
    client = CosmosClient(url=cosmos_endpoint, credential=cosmos_credentials)
    
    # Load the database and container
    database = client.get_database_client(cosmos_database_name)
    container = database.get_container_client(cosmos_container_name)

    # Prepare the vector distance query
    query = f"""
        SELECT TOP {max_results}
            c.id,
            c.call_id,
            c.call_transcript,
            c.request_vector,
            VectorDistance(c.request_vector, @query_vector) AS SimilarityScore
        FROM c
        WHERE VectorDistance(c.request_vector, @query_vector) > {minimum_similarity_score}
        ORDER BY VectorDistance(c.request_vector, @query_vector)
    """

    # Run the query with your query embedding
    results = container.query_items(
        query=query,
        parameters=[
            {"name": "@query_vector", "value": query_embedding}
        ],
        enable_cross_partition_query=True
    )

    return list(results)  # Convert iterator to a list for easy consumption


@st.cache_data
def create_transcription_request(audio_file, speech_recognition_language="en-US"):
    """Transcribe the contents of an audio file. Key assumptions:
    - The audio file is in WAV format.
    - The audio file is mono.
    - The audio file has a sample rate of 16 kHz.
    - Speech key and region are stored in Streamlit secrets."""

    speech_key = st.secrets["speech"]["key"]
    speech_region = st.secrets["speech"]["region"]

    # Create an instance of a speech config with specified subscription key and service region.
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_recognition_language = speech_recognition_language

    # Prepare audio settings for the wave stream
    channels = 1
    bits_per_sample = 16
    samples_per_second = 16000

    # Create audio configuration using the push stream
    wave_format = speechsdk.audio.AudioStreamFormat(samples_per_second, bits_per_sample, channels)
    stream = speechsdk.audio.PushAudioInputStream(stream_format=wave_format)
    audio_config = speechsdk.audio.AudioConfig(stream=stream)

    transcriber = speechsdk.transcription.ConversationTranscriber(speech_config, audio_config)
    all_results = []

    def handle_final_result(evt):
        all_results.append(evt.result.text)

    done = False

    def stop_cb(evt):
        print(f'CLOSING on {evt}')
        nonlocal done
        done = True

    # 1) Subscribe to the events fired by the conversation transcriber
    transcriber.transcribed.connect(handle_final_result)
    transcriber.session_started.connect(lambda evt: print(f'SESSION STARTED: {evt}'))
    transcriber.session_stopped.connect(lambda evt: print(f'SESSION STOPPED {evt}'))
    transcriber.canceled.connect(lambda evt: print(f'CANCELED {evt}'))

    # 2) Stop continuous transcription on either session stopped or canceled events
    transcriber.session_stopped.connect(stop_cb)
    transcriber.canceled.connect(stop_cb)

    # Start transcription
    transcriber.start_transcribing_async()

    # Read the entire WAV file at once and stream it to the SDK
    _, wav_data = wavfile.read(audio_file)
    stream.write(wav_data.tobytes())
    stream.close()

    while not done:
        time.sleep(.5)

    transcriber.stop_transcribing_async()

    return all_results

def make_azure_openai_chat_request(system, call_contents):
    """Create and return a new chat completion request. Key assumptions:
    - Azure OpenAI endpoint, key, and deployment name stored in Streamlit secrets."""

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    
    aoai_endpoint = st.secrets["aoai"]["endpoint"]
    aoai_deployment_name = st.secrets["aoai"]["deployment_name"]

    client = openai.AzureOpenAI(
        azure_ad_token_provider=token_provider,
        api_version="2024-06-01",
        azure_endpoint=aoai_endpoint
    )
    # Create and return a new chat completion request
    return client.chat.completions.create(
        model=aoai_deployment_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": call_contents}
        ],
    )

@st.cache_data
def is_call_in_compliance(call_contents, include_recording_message, is_relevant_to_topic):
    """Analyze a call for relevance and compliance."""

    joined_call_contents = ' '.join(call_contents)
    if include_recording_message:
        include_recording_message_text = "2. Was the caller aware that the call was being recorded?"
    else:
        include_recording_message_text = ""

    if is_relevant_to_topic:
        is_relevant_to_topic_text = "3. Was the call relevant to the hotel and resort industry?"
    else:
        is_relevant_to_topic_text = ""

    system = f"""
        You are an automated analysis system for Contoso Suites.
        Contoso Suites is a luxury hotel and resort chain with locations
        in a variety of Caribbean nations and territories.

        You are analyzing a call for relevance and compliance.

        You will only answer the following questions based on the call contents:
        1. Was there vulgarity on the call?
        {include_recording_message_text}
        {is_relevant_to_topic_text}
    """

    response = make_azure_openai_chat_request(system, joined_call_contents)
    return response.choices[0].message.content

@st.cache_data
def generate_extractive_summary(call_contents):
    """Generate an extractive summary of a call transcript. Key assumptions:
    - Azure AI Services Language service endpoint and key stored in Streamlit secrets."""

    language_endpoint = st.secrets["language"]["endpoint"]
    language_key = st.secrets["language"]["key"]

    joined_call_contents = ' '.join(call_contents)

    # 1) Create a TextAnalyticsClient
    client = TextAnalyticsClient(language_endpoint, AzureKeyCredential(language_key))

    # 2) Call the begin_analyze_actions with ExtractiveSummaryAction
    poller = client.begin_analyze_actions(
        [joined_call_contents],
        actions=[
            ExtractiveSummaryAction(max_sentence_count=2)
        ]
    )

    # 3) Extract the summary sentences
    extractive_summary = ""
    for result in poller.result():
        summary_result = result[0]
        if summary_result.is_error:
            st.error(f'Extractive summary resulted in an error with code "{summary_result.code}" and message "{summary_result.message}"')
            return ''
        extractive_summary = " ".join([sentence.text for sentence in summary_result.sentences])

    # 4) Return the summary as a JSON object in the shape '{"call-summary": extractive_summary}'
    return json.loads('{"call-summary":"' + extractive_summary + '"}')

@st.cache_data
def generate_abstractive_summary(call_contents):
    """Generate an abstractive summary of a call transcript. Key assumptions:
    - Azure AI Services Language service endpoint and key stored in Streamlit secrets."""

    language_endpoint = st.secrets["language"]["endpoint"]
    language_key = st.secrets["language"]["key"]

    joined_call_contents = ' '.join(call_contents)

    # 1) Create a TextAnalyticsClient
    client = TextAnalyticsClient(language_endpoint, AzureKeyCredential(language_key))

    # 2) Call the begin_analyze_actions with AbstractiveSummaryAction
    poller = client.begin_analyze_actions(
        [joined_call_contents],
        actions=[
            AbstractiveSummaryAction(sentence_count=2)
        ]
    )

    # 3) Extract the summary
    abstractive_summary = ""
    for result in poller.result():
        summary_result = result[0]
        if summary_result.is_error:
            st.error(f'Abstractive summary error: code "{summary_result.code}" - {summary_result.message}')
            return ''
        abstractive_summary = " ".join([summary.text for summary in summary_result.summaries])

    # 4) Return the summary as JSON
    return json.loads('{"call-summary":"' + abstractive_summary + '"}')

@st.cache_data
def generate_query_based_summary(call_contents):
    """Generate a query-based summary of a call transcript."""

    joined_call_contents = ' '.join(call_contents)

    # system prompt for short (5 word) summary and two-sentence summary
    system = """
        Write a five-word summary and label it as call-title.
        Write a two-sentence summary and label it as call-summary.

        Output the results in JSON format.
    """

    response = make_azure_openai_chat_request(system, joined_call_contents)
    return response.choices[0].message.content

@st.cache_data
def create_sentiment_analysis_and_opinion_mining_request(call_contents):
    """Analyze the sentiment of a call transcript and mine opinions. Key assumptions:
    - Azure AI Services Language service endpoint and key stored in Streamlit secrets."""

    language_endpoint = st.secrets["language"]["endpoint"]
    language_key = st.secrets["language"]["key"]

    joined_call_contents = ' '.join(call_contents)

    # 1) Create a TextAnalyticsClient
    client = TextAnalyticsClient(language_endpoint, AzureKeyCredential(language_key))

    # 2) Analyze sentiment with opinion mining
    result = client.analyze_sentiment([joined_call_contents], show_opinion_mining=True)

    doc_result = [doc for doc in result if not doc.is_error]
    sentiment = {}

    for document in doc_result:
        sentiment["sentiment"] = document.sentiment
        sentiment["sentiment-scores"] = {
            "positive": document.confidence_scores.positive,
            "neutral": document.confidence_scores.neutral,
            "negative": document.confidence_scores.negative
        }

        sentences = []
        for s in document.sentences:
            sentence = {
                "text": s.text,
                "sentiment": s.sentiment,
                "sentiment-scores": {
                    "positive": s.confidence_scores.positive,
                    "neutral": s.confidence_scores.neutral,
                    "negative": s.confidence_scores.negative
                },
                "mined_opinions": []
            }

            for mined_opinion in s.mined_opinions:
                opinion = {
                    "target-text": mined_opinion.target.text,
                    "target-sentiment": mined_opinion.target.sentiment,
                    "sentiment-scores": {
                        "positive": mined_opinion.target.confidence_scores.positive,
                        "negative": mined_opinion.target.confidence_scores.negative
                    },
                    "assessments": []
                }
                for assessment in mined_opinion.assessments:
                    opinion_assessment = {
                        "text": assessment.text,
                        "sentiment": assessment.sentiment,
                        "sentiment-scores": {
                            "positive": assessment.confidence_scores.positive,
                            "negative": assessment.confidence_scores.negative
                        }
                    }
                    opinion["assessments"].append(opinion_assessment)

                sentence["mined_opinions"].append(opinion)

            sentences.append(sentence)

        sentiment["sentences"] = sentences

    return sentiment

def make_azure_openai_embedding_request(text):
    """Create and return a new embedding request. Key assumptions:
    - Azure OpenAI endpoint, key, and embedding deployment name stored in Streamlit secrets."""
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    aoai_endpoint = st.secrets["aoai"]["endpoint"]
    aoai_embedding_deployment_name = st.secrets["aoai"]["embedding_deployment_name"]

    client = openai.AzureOpenAI(
        azure_ad_token_provider=token_provider,
        api_version="2024-06-01",
        azure_endpoint=aoai_endpoint
    )
    # Create and return a new embedding request
    return client.embeddings.create(
        model=aoai_embedding_deployment_name,
        input=text
    )

def normalize_text(s):
    """Normalize text for tokenization."""
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r". ,","",s)
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.strip()
    return s

def generate_embeddings_for_call_contents(call_contents):
    """Generate embeddings for call contents. Key assumptions:
    - call_contents is a single string.
    - Azure OpenAI endpoint, key, and embedding deployment name stored in Streamlit secrets."""
    normalized_content = normalize_text(call_contents)
    response = make_azure_openai_embedding_request(normalized_content)
    return response.data[0].embedding

def save_transcript_to_cosmos_db(transcript_item):
    """Save embeddings to Cosmos DB vector store. Key assumptions:
    - transcript_item is a JSON object containing call_id (int), 
      call_transcript (string), and request_vector (list).
    - Cosmos DB endpoint, client_id, and database name stored in Streamlit secrets."""
    cosmos_client_id = st.secrets["cosmos"]["client_id"]
    cosmos_credentials = DefaultAzureCredential(managed_identity_client_id=cosmos_client_id)

    cosmos_endpoint = st.secrets["cosmos"]["endpoint"]
    cosmos_database_name = st.secrets["cosmos"]["database_name"]
    cosmos_container_name = "CallTranscripts"

    # Create a CosmosClient
    client = CosmosClient(url=cosmos_endpoint, credential=cosmos_credentials)
    # Load the Cosmos database and container
    database = client.get_database_client(cosmos_database_name)
    container = database.get_container_client(cosmos_container_name)

    # Insert the call transcript
    container.create_item(body=transcript_item)


####################### HELPER FUNCTIONS FOR MAIN() #######################
def perform_audio_transcription(uploaded_file):
    """Generate a transcription of an uploaded audio file."""
    st.audio(uploaded_file, format='audio/wav')
    with st.spinner("Transcribing the call..."):
        all_results = create_transcription_request(uploaded_file)
        return all_results

def perform_compliance_check(call_contents, include_recording_message, is_relevant_to_topic):
    """Perform a compliance check on a call transcript."""
    with st.spinner("Checking for compliance..."):
        if 'file_transcription_results' in st.session_state:
            call_contents = st.session_state.file_transcription_results
            if call_contents is not None and len(call_contents) > 0:
                st.session_state.compliance_results = is_call_in_compliance(
                    call_contents, include_recording_message, is_relevant_to_topic)
            st.success("Compliance check complete!")
        else:
            st.write("Please upload an audio file before checking for compliance.")

def perform_extractive_summary_generation():
    """Generate an extractive summary of a call transcript."""
    if 'file_transcription_results' in st.session_state:
        with st.spinner("Generating extractive summary..."):
            if 'extractive_summary' in st.session_state:
                extractive_summary = st.session_state.extractive_summary
            else:
                ftr = st.session_state.file_transcription_results
                extractive_summary = generate_extractive_summary(ftr)
                st.session_state.extractive_summary = extractive_summary

            if extractive_summary is not None:
                st.success("Extractive summarization complete!")
    else:
        st.error("Please upload an audio file before attempting to generate a summary.")

def perform_abstractive_summary_generation():
    """Generate an abstractive summary of a call transcript."""
    if 'file_transcription_results' in st.session_state:
        with st.spinner("Generating abstractive summary..."):
            ftr = st.session_state.file_transcription_results
            abstractive_summary = generate_abstractive_summary(ftr)
            st.session_state.abstractive_summary = abstractive_summary

            if abstractive_summary is not None:
                st.success("Abstractive summarization complete!")
    else:
        st.error("Please upload an audio file before attempting to generate a summary.")

def perform_openai_summary():
    """Generate a query-based summary of a call transcript."""
    if 'file_transcription_results' in st.session_state:
        with st.spinner("Generating Azure OpenAI summary..."):
            summary = generate_query_based_summary(st.session_state.file_transcription_results)
            st.session_state.openai_summary = summary

            if summary is not None:
                st.success("Azure OpenAI query-based summarization complete!")
    else:
        st.error("Please upload an audio file before attempting to generate a summary.")

def perform_sentiment_analysis_and_opinion_mining():
    """Analyze the sentiment of a call transcript and mine opinions."""
    if 'file_transcription_results' in st.session_state:
        with st.spinner("Analyzing transcript sentiment and mining opinions..."):
            ftr = st.session_state.file_transcription_results
            smo = create_sentiment_analysis_and_opinion_mining_request(ftr)
            st.session_state.sentiment_and_mined_opinions = smo

            if smo is not None:
                st.success("Sentiment analysis and opinion mining complete!")
    else:
        st.error("Please upload an audio file before attempting to analyze sentiment.")

def perform_save_embeddings_to_cosmos_db():
    """Save embeddings to Cosmos DB vector store."""
    if 'file_transcription_results' in st.session_state:
        with st.spinner("Saving embeddings to Cosmos DB..."):
            ftr = ' '.join(st.session_state.file_transcription_results)
            call_id = abs(hash(ftr)) % (10 ** 8)
            embeddings = generate_embeddings_for_call_contents(ftr)
            transcript_item = {
                "id": f'{call_id}_{uuid.uuid4()}',
                "call_id": call_id,
                "call_transcript": ftr,
                "request_vector": embeddings
            }
            save_transcript_to_cosmos_db(transcript_item)
            st.session_state.embedding_status = "Transcript and embeddings saved for this audio."
            st.success("Embeddings saved to Cosmos DB!")
    else:
        st.error("Please upload an audio file before attempting to save embeddings.")


##################
# Extra function to do "Vector Search" in the UI
##################
def perform_vector_search():
    """Search for relevant transcripts in Cosmos DB by generating an embedding of user input."""
    query_text = st.text_input("Enter search text:")
    max_results = st.number_input("Max results:", min_value=1, max_value=10, value=5)
    min_score = st.slider("Minimum similarity score:", 0.0, 1.0, 0.5, 0.01)

    if st.button("Run Vector Search"):
        if query_text.strip():
            with st.spinner("Generating embedding and searching transcripts..."):
                # generate embedding from query text
                embedding_resp = make_azure_openai_embedding_request(query_text)
                query_embedding = embedding_resp.data[0].embedding

                # query cosmos
                results = make_cosmos_db_vector_search_request(query_embedding, max_results, min_score)

                if len(results) == 0:
                    st.info("No transcripts matched above that similarity score.")
                else:
                    for doc in results:
                        st.write(f"**ID**: {doc['id']}")
                        st.write(f"**Call ID**: {doc.get('call_id')}")
                        st.write(f"**Transcript**: {doc.get('call_transcript')}")
                        st.write(f"**Similarity Score**: {doc.get('SimilarityScore')}")
                        st.write("---")
            st.success("Vector search complete.")
        else:
            st.warning("Enter some text for searching.")


def main():
    """Main function for the call center dashboard."""
    call_contents = []
    st.write(
    """
    # Call Center

    This Streamlit dashboard is intended to replicate some of the functionality
    of a call center monitoring solution. It is not intended to be a
    production-ready application.
    """
    )

    st.write("## Upload a Call")

    uploaded_file = st.file_uploader("Upload an audio file", type="wav")
    if uploaded_file is not None and ('file_transcription_results' not in st.session_state):
        st.session_state.file_transcription_results = perform_audio_transcription(uploaded_file)
        st.success("Transcription complete!")

    if 'file_transcription_results' in st.session_state:
        st.write(st.session_state.file_transcription_results)

    st.write("## Transcription Operations")

    tabs = st.tabs([
        "Compliance",
        "Extractive Summary",
        "Abstractive Summary",
        "Azure OpenAI Summary",
        "Sentiment and Opinions",
        "Save to DB",
        "Vector Search"  # new tab for vector searching
    ])

    comp = tabs[0]
    esum = tabs[1]
    asum = tabs[2]
    osum = tabs[3]
    sent = tabs[4]
    db = tabs[5]
    vsrch = tabs[6]

    with comp:
        st.write("## Is Your Call in Compliance?")
        include_recording_message = st.checkbox("Call needs an indicator we are recording it")
        is_relevant_to_topic = st.checkbox("Call is relevant to the hotel and resort industry")

        if st.button("Check for Compliance"):
            perform_compliance_check(call_contents, include_recording_message, is_relevant_to_topic)

        if 'compliance_results' in st.session_state:
            st.write(st.session_state.compliance_results)

    with esum:
        if st.button("Generate extractive summary"):
            perform_extractive_summary_generation()

        if 'extractive_summary' in st.session_state:
            st.write(st.session_state.extractive_summary)

    with asum:
        if st.button("Generate abstractive summary"):
            perform_abstractive_summary_generation()

        if 'abstractive_summary' in st.session_state:
            st.write(st.session_state.abstractive_summary)

    with osum:
        if st.button("Generate query-based summary"):
            perform_openai_summary()

        if 'openai_summary' in st.session_state:
            st.write(st.session_state.openai_summary)

    with sent:
        if st.button("Analyze sentiment and mine opinions"):
            perform_sentiment_analysis_and_opinion_mining()

        if 'sentiment_and_mined_opinions' in st.session_state:
            st.write(st.session_state.sentiment_and_mined_opinions)

    with db:
        if st.button("Save embeddings to Cosmos DB"):
            perform_save_embeddings_to_cosmos_db()

        if 'embedding_status' in st.session_state:
            st.write(st.session_state.embedding_status)

    # The new Vector Search tab
    with vsrch:
        st.write("## Vector Search for Transcripts")
        perform_vector_search()


if __name__ == "__main__":
    main()