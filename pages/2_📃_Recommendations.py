import requests
import replicate
import streamlit as st
import re
import nltk
nltk.download("stopwords")
from youtube_transcript_api import YouTubeTranscriptApi
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

# Replace 'YOUR_API_KEY' with your actual API key
api_key = "ADD YOUR YOUTUBE API KEY"


# URL of your Google Apps Script web app
apps_script_url = "ADD YOUR GOOGLE APPS SCRIPT WEB APP URL"


def get_video_title(api_key, video_id):
    base_url = "https://www.googleapis.com/youtube/v3/videos"
    params = {"part": "snippet", "id": video_id, "key": api_key}

    response = requests.get(base_url, params=params)
    video_data = response.json()

    try:
        video_title = video_data["items"][0]["snippet"]["title"]
        return str(video_title)
    except (KeyError, IndexError):
        return None


def get_video_id(url):
    regex = r"(?<=watch\?v=)([\w-]+)"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    else:
        return None


def remove_non_alphabetic(input_string):
    # Use regular expression to remove non-alphabetic characters
    cleaned_string = re.sub(r"[^a-zA-Z0-9\s]", "", input_string)
    return cleaned_string


def remove_stopwords(sentence):
    stop_words = set(stopwords.words("english"))
    words = sentence.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)


# transcription
def generate_transcription(video_id):
    # Get transcript
    transcript = YouTubeTranscriptApi.get_transcript(video_id)

    # Extract text from the transcript
    text = ""
    for entry in transcript:
        text += entry["text"] + " "

    return text



def get_course_info(subject):
    # Construct the Coursera search URL
    url = f"https://www.coursera.org/search?query={subject}"

    # Send a GET request to the URL
    response = requests.get(url)

    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all course elements
    course_elements = soup.find_all(
        "li", class_="cds-9 css-0 cds-11 cds-grid-item cds-56 cds-64 cds-76"
    )

    # Extract course information for the top 5 courses
    course_info = []
    for title in course_elements[:5]:
        title_class = title.find(
            class_="cds-119 cds-CommonCard-title css-e7lgfl cds-121"
        )
        link_class = title.find(
            class_="cds-119 cds-113 cds-115 cds-CommonCard-titleLink css-si869u cds-142"
        )
        image_class = title.find("div", class_="cds-CommonCard-previewImage").find(
            "img"
        )["src"]
        ratings_class = title.find(class_="cds-119 css-11uuo4b cds-121")

        if title_class and link_class and image_class and ratings_class:
            link = link_class["href"]
            ratings = ratings_class.text.strip()

            course_info.append(
                {
                    "title": title_class.text.strip(),
                    "link": link,
                    "image": image_class,
                    "ratings": ratings,
                }
            )

    return course_info


def get_topic(video_title):
    # REPLICATE_API_TOKEN = "r8_2xXDCVqk0FuIsLZEQXpqLwp6xOD2MRK1DvkjP"

    pre_prompt = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    # prompt_input = video_title
    prompt = (
        "Act as a program that gives a specific output. your output should only consist of the topic of this sentence- ",
        video_title,
    )
    # Generate LLM response
    output = replicate.run(
        "replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781",  # LLM model
        input={
            "prompt": f" {prompt} ",  # Prompts
            "temperature": 0.1,
            "top_p": 0.9,
            "max_length": 128,
            "repetition_penalty": 1,
        },
    )  # Model parameters
    # print("___________-----____________",output,"________________-------")

    full_response = ""
    for item in output:
        full_response += item
    print("full response", full_response)
    chunks = full_response.split('"')
    print(chunks)
    print(chunks[1])
    return chunks[1]



# Set title of the browser tab
youtube_url = st.session_state["youtube_url"]
# st.video(youtube_url)
# print(youtube_url)
video_id = get_video_id(youtube_url)
# transcript = generate_transcription(video_id)
user_input = remove_non_alphabetic(str(get_video_title(api_key, video_id)))
sentence = user_input
user_input = remove_stopwords(sentence)
print("=====================")
print(user_input)

# Set the title of the Streamlit app
st.title("Sensei Extension")

# Get query parameters
if youtube_url:

    # Use st.video to embed the YouTube video
    st.video(youtube_url)

    # Check if a YouTube URL is provided

    with st.expander("**RECOMMENDED VIDEOS**"):
        # st.write("Recommended videos on the topic:")
        search_term = user_input
        params = {"q": search_term}
        response = requests.get(apps_script_url, params=params)

        if response.status_code == 200:
            try:
                data = response.json()
                print("=====================")
                print(data)
                videos = data["videos"]

                top_videos = videos[
                    :5
                ]  # Get the top 5 videos with the highest view counts

                # Display the top videos in a row
                st.header(
                    "Top 5 Videos with the Highest View Counts", anchor="center"
                )  # Anchor the header to the center

                for i, video in enumerate(top_videos):
                    # Create a container for thumbnail, table, and title
                    st.markdown(
                        f"<div style='display: flex; align-items: center; margin: 10px;'>"
                        f"<a href='https://www.youtube.com/watch?v={video['videoId']}' target='_blank'>"
                        f"<img src='https://img.youtube.com/vi/{video['videoId']}/default.jpg' width='240' height='180'></a>"
                        f"<div style='margin-left: 10px;'>"
                        f"<h3 style='font-size: 18px;'>{video['videoTitle']}</h3>"  # Adjust the font size here
                        f"<table style='text-align: center; margin-top: 10px;'><tr><th>View Count</th><th>Like Count</th></tr><tr>"
                        f"<td>{int(video['statistics']['viewCount']):,}</td>"
                        f"<td>{int(video['statistics']['likeCount']):,}</td></tr></table></div></div>",
                        unsafe_allow_html=True,
                    )
            except ValueError as e:
                html_content = response.text
                soup = BeautifulSoup(html_content, "html.parser")
                pretty_html = soup.prettify()
                st.markdown(pretty_html, unsafe_allow_html=True)
                st.error(f"Error parsing JSON: {e}")
        else:
            st.error(
                "Error fetching data from the YouTube API, Status code: {response.status_code}"
            )
            # Send the input to the Google Apps Script using a GET request

    with st.expander("**RECOMMENDED COURSES**"):
        topic_to_search = get_topic(str(get_video_title(api_key, video_id)))

        if topic_to_search:
            course_info = get_course_info(topic_to_search)

            for info in course_info[:5]:
                # Display each container with thumbnail in the left column, title in the center column, and ratings in the right column
                st.markdown(
                    f'<div class="row" style="margin: 10px; display: flex; align-items: center; justify-content: space-around;">'
                    f'   <div style="flex: 1;"><a href="{info["link"]}" target="_blank"><img src="{info["image"]}" alt="{info["title"]}" style="width: 120px; height: 120px;"></a></div>'
                    f'   <div style="flex: 2; text-align: justify; font-weight: bold; margin: 10px;"><a href="{info["link"]}" target="_blank">{info["title"]}</a></div>'
                    f'   <div style="flex: 1; text-align: center; margin: 10px;">{info["ratings"]}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.write("---")
        else:
            st.warning("Please enter a subject to search for courses.")
else:
    st.warning("No YouTube video URL provided.")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
