import streamlit as st
from Quiz import final_questions

# final_questions = [
#         "What is a subset of machine learning methods?\n(a)Machine learning algorithms\n(b)Rnns\n(c)Fpgas\n(d)deep learning\n(d)\n\n",
#         "What is a subset of machine learning methods based on artificial neural networks?\n(a)deep\n(b)Burnt\n(c)Like\n(d)Mazzy\n(a)\n\n",
#         "What is Deep learning based on?\n(a)recurrent neural networks\n(b)deep\n(c)recurrent neural networks\n(d)uses multiple layers\n(a)\n\n",
#         "How many layers does deep learning use?\n(a)deep learning\n(b)deep\n(c)recurrent neural networks\n(d)uses multiple layers\n(d)\n\n",
#         "Deep learning is a subset of machine learning based on what artificial neural network?\n(a)deep learning\n(b)anns\n(c)recurrent neural networks\n(d)uses multiple layers\n(b)\n\n",
#         "What type of transformers are used in deep learning?\n(a)Snowpiercer\n(b)Incredibles\n(c)Terminator\n(d)transformers\n(d)\n\n",
#         "Deep learning is a subset of what type of learning methods?\n(a)machine translation\n(b)Native speakers\n(c)Sentence structure\n(d)Written form\n(a)\n\n",
#         "What is a subset of machine learning methods based on?\n(a)deep learning\n(b)deep\n(c)deep belief networks\n(d)uses multiple layers\n(c)\n\n",
#         "What is the purpose of deep learning?\n(a)deep learning\n(b)identify edges\n(c)recurrent neural networks\n(d)uses multiple layers\n(b)\n\n",
#         "What field of study is deep learning a subset of?\n(a)Subfield\n(b)Math background\n(c)bioinformatics\n(d)Software engineering\n(c)\n\n",
#         "Deep learning is a subset of what type of machine learning?\n(a)Biological sciences\n(b)drug design\n(c)Pathophysiology\n(d)Regenerative medicine\n(b)\n\n",
#         "What type of processing is deep learning a subset of?\n(a)Deep learning\n(b)Anns\n(c)Large data sets\n(d)natural language processing\n(d)\n\n",
#         "What is an example of a type of deep learning?\n(a)deep learning\n(b)deep\n(c)recurrent neural networks\n(d)medical image analysis\n(d)\n\n",
#         "Deep learning is a subset of what type of science?\n(a)Deniers\n(b)climate science\n(c)Creationists\n(d)Climatologists\n(b)\n\n",
#         "What type of inspection is a part of deep learning?\n(a)deep learning\n(b)deep\n(c)material inspection\n(d)uses multiple layers\n(c)\n\n",
#     ]

# Set title of the browser tab
# youtube_url = st.experimental_get_query_params().get("youtube_url", [""])[0]
youtube_url = st.session_state["youtube_url"]

# st.video(youtube_url)
# print(youtube_url)
# Set the title of the Streamlit app
st.title("Sensei Extension")

# Get query parameters
if youtube_url:
    # st.write(f"Embedding YouTube video from URL: {youtube_url}")

    # Use st.video to embed the YouTube video
    st.video(youtube_url)

        # Check if a YouTube URL is provided

    with st.expander("**QUIZ BASED ON THE TOPIC**"):
        # st.write("Quiz based on the topic")
        ques=[]
        for i in final_questions:
            parts = i.split("\n")
            options=parts[1:-3]
            # print(options)
            options_without_prefix = [option[3:] for option in options]
            print(options_without_prefix)
            is_not_redundant = len(options_without_prefix) == len(set(options_without_prefix))
            if is_not_redundant:
                ques.append(i)

        for i, question in enumerate(ques, 1):
            st.subheader(f"Question {i}:")

            parts = question.split("\n")
            que = parts[0]
            options = parts[1:-2]
            correct_answer = options[-1].strip()[1]
            options = parts[1:-3]
            # options = ['(a)deep learning', '(b)anns', '(c)recurrent neural networks', '(d)uses multiple layers']

            # Remove the first three characters from each element
            # options_without_prefix = [option[3:] for option in options]
            # is_redundant = len(options_without_prefix) != len(set(options_without_prefix))

            # # Display the result
            # if is_redundant:
            #     continue

            # print(options)
            st.markdown(que, unsafe_allow_html=True)

            # Radio buttons for answer choices
            user_choice = st.radio(
                f"Select your answer for Question {i}:", options, index=None)

            # Check if the selected choice is correct
            if user_choice:
                user_choice = user_choice.strip()[1]
                if user_choice == correct_answer:
                    st.write(f'Your answer "{user_choice}" is correct!')
                else:
                    st.write(f"Sorry, the correct answer is: {correct_answer}")
            else:
                st.write("Please select an answer before revealing the correct one.")   
else:
    st.warning("No YouTube video URL provided.")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

                    
