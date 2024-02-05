# import all the neccessary libraries
import warnings
import streamlit as st
warnings.filterwarnings("ignore") # type: ignore
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sense2vec import Sense2Vec
from sentence_transformers import SentenceTransformer
from textwrap3 import wrap
import random
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('brown')
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
import pke
import traceback
from flashtext import KeywordProcessor
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('omw-1.4')
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
import pickle
import time
import os
import torch
from youtube_transcript_api import YouTubeTranscriptApi



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# we need to download the 2015 trained on reddit sense2vec model as it is shown to give better results than the 2019 one.
s2v = Sense2Vec().from_disk("s2v_old")

# getitng the summary model and its tokenizer
if os.path.exists("t5_summary_model.pkl"):
    with open("t5_summary_model.pkl", "rb") as f:
        summary_model = pickle.load(f)
    print("Summary model found in the disc, model is loaded successfully.")

else:
    print(
        "Summary model does not exists in the path specified, downloading the model from web...."
    )
    start_time = time.time()
    summary_model = T5ForConditionalGeneration.from_pretrained("t5-base")
    end_time = time.time()

    print(
        "downloaded the summary model in ",
        (end_time - start_time) / 60,
        " min , now saving it to disc...",
    )

    with open("t5_summary_model.pkl", "wb") as f:
        pickle.dump(summary_model, f)

    print("Done. Saved the model to disc.")

if os.path.exists("t5_summary_tokenizer.pkl"):
    with open("t5_summary_tokenizer.pkl", "rb") as f:
        summary_tokenizer = pickle.load(f)
    print("Summary tokenizer found in the disc and is loaded successfully.")
else:
    print(
        "Summary tokenizer does not exists in the path specified, downloading the model from web...."
    )

    start_time = time.time()
    summary_tokenizer = T5Tokenizer.from_pretrained("t5-base")
    end_time = time.time()

    print(
        "downloaded the summary tokenizer in ",
        (end_time - start_time) / 60,
        " min , now saving it to disc...",
    )

    with open("t5_summary_tokenizer.pkl", "wb") as f:
        pickle.dump(summary_tokenizer, f)

    print("Done. Saved the tokenizer to disc.")


# Getting question model and tokenizer
if os.path.exists("t5_question_model.pkl"):
    with open("t5_question_model.pkl", "rb") as f:
        question_model = pickle.load(f)
    print("Question model found in the disc, model is loaded successfully.")
else:
    print(
        "Question model does not exists in the path specified, downloading the model from web...."
    )
    start_time = time.time()
    question_model = T5ForConditionalGeneration.from_pretrained(
        "ramsrigouthamg/t5_squad_v1"
    )
    end_time = time.time()

    print(
        "downloaded the question model in ",
        (end_time - start_time) / 60,
        " min , now saving it to disc...",
    )

    with open("t5_question_model.pkl", "wb") as f:
        pickle.dump(question_model, f)

    print("Done. Saved the model to disc.")

if os.path.exists("t5_question_tokenizer.pkl"):
    with open("t5_question_tokenizer.pkl", "rb") as f:
        question_tokenizer = pickle.load(f)
    print("Question tokenizer found in the disc, model is loaded successfully.")
else:
    print(
        "Question tokenizer does not exists in the path specified, downloading the model from web...."
    )

    start_time = time.time()
    question_tokenizer = T5Tokenizer.from_pretrained("ramsrigouthamg/t5_squad_v1")
    end_time = time.time()

    print(
        "downloaded the question tokenizer in ",
        (end_time - start_time) / 60,
        " min , now saving it to disc...",
    )

    with open("t5_question_tokenizer.pkl", "wb") as f:
        pickle.dump(question_tokenizer, f)

    print("Done. Saved the tokenizer to disc.")

# Loading the models in to GPU if available
summary_model = summary_model.to(device)  # type: ignore
question_model = question_model.to(device)  # type: ignore

# Getting the sentence transformer model and its tokenizer
# paraphrase-distilroberta-base-v1
if os.path.exists("sentence_transformer_model.pkl"):
    with open("sentence_transformer_model.pkl", "rb") as f:
        sentence_transformer_model = pickle.load(f)
    print("Sentence transformer model found in the disc, model is loaded successfully.")
else:
    print(
        "Sentence transformer model does not exists in the path specified, downloading the model from web...."
    )
    start_time = time.time()
    sentence_transformer_model = SentenceTransformer(
        "sentence-transformers/msmarco-distilbert-base-v2"
    )
    end_time = time.time()

    print(
        "downloaded the sentence transformer in ",
        (end_time - start_time) / 60,
        " min , now saving it to disc...",
    )

    with open("sentence_transformer_model.pkl", "wb") as f:
        pickle.dump(sentence_transformer_model, f)

    print("Done saving to disc.")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def postprocesstext(content):
    """
    this function takes a piece of text (content), tokenizes it into sentences, capitalizes the first letter of each sentence, and then concatenates the processed sentences into a single string, which is returned as the final result. The purpose of this function could be to format the input content by ensuring that each sentence starts with an uppercase letter.
    """
    final = ""
    for sent in sent_tokenize(content):
        sent = sent.capitalize()
        final = final + " " + sent
    return final


def summarizer(text, model, tokenizer):
    """
    This function takes the given text along with the model and tokenizer, which summarize the large text into useful information
    """
    text = text.strip().replace("\n", " ")
    text = "summarize: " + text
    # print (text)
    max_len = 512
    encoding = tokenizer.encode_plus(
        text,
        max_length=max_len,
        pad_to_max_length=False,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        early_stopping=True,
        num_beams=3,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        min_length=75,
        max_length=300,
    )

    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
    summary = dec[0]
    summary = postprocesstext(summary)
    summary = summary.strip()

    return summary


def get_nouns_multipartite(content):
    """
    This function takes the content text given and then outputs the phrases which are build around the nouns , so that we can use them for context based distractors
    """
    out = []
    try:
        extractor = pke.unsupervised.MultipartiteRank()  # type: ignore
        extractor.load_document(input=content, language="en")
        #    not contain punctuation marks or stopwords as candidates.
        # pos = {'PROPN','NOUN',}
        pos = {
            "PROPN",
            "NOUN",
            "ADJ",
            "VERB",
            "ADP",
            "ADV",
            "DET",
            "CONJ",
            "NUM",
            "PRON",
            "X",
        }

        # pos = {'PROPN','NOUN'}
        stoplist = list(string.punctuation)
        stoplist += ["-lrb-", "-rrb-", "-lcb-", "-rcb-", "-lsb-", "-rsb-"]
        stoplist += stopwords.words("english")
        # extractor.candidate_selection(pos=pos, stoplist=stoplist)
        extractor.candidate_selection(pos=pos)
        # 4. build the Multipartite graph and rank candidates using random walk,
        #    alpha controls the weight adjustment mechanism, see TopicRank for
        #    threshold/method parameters.
        extractor.candidate_weighting(alpha=1.1, threshold=0.75, method="average")
        keyphrases = extractor.get_n_best(n=15)

        for val in keyphrases:
            out.append(val[0])
    except:
        out = []
        # traceback.print_exc()

    return out


def get_keywords(originaltext):
    """
    This function takes the original text and the summary text and generates keywords from both which ever are more relevant
    This is done by checking the keywords generated from the original text to those generated from the summary, so that we get important ones
    """
    keywords = get_nouns_multipartite(originaltext)
    # print ("keywords unsummarized: ",keywords)
    # keyword_processor = KeywordProcessor()
    # for keyword in keywords:
    # keyword_processor.add_keyword(keyword)

    # keywords_found = keyword_processor.extract_keywords(summarytext)
    # keywords_found = list(set(keywords_found))
    # print ("keywords_found in summarized: ",keywords_found)

    # important_keywords =[]
    # for keyword in keywords:
    # if keyword in keywords_found:
    # important_keywords.append(keyword)

    # return important_keywords
    return keywords


def get_question(context, answer, model, tokenizer):
    """
    This function takes the input context text, pretrained model along with the tokenizer and the keyword and the answer and then generates the question from the large paragraph

    """
    text = "context: {} answer: {}".format(context, answer)
    encoding = tokenizer.encode_plus(
        text,
        max_length=384,
        pad_to_max_length=False,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        early_stopping=True,
        num_beams=5,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        max_length=72,
    )

    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
    Question = dec[0].replace("question:", "")
    Question = Question.strip()
    return Question


def filter_same_sense_words(original, wordlist):
    """
    This is used to filter the words which are of same sense, where it takes the wordlist which has the sense of the word attached as the string along with the word itself.
    """
    filtered_words = []
    base_sense = original.split("|")[1]
    # print (base_sense)
    for eachword in wordlist:
        if eachword[0].split("|")[1] == base_sense:
            filtered_words.append(
                eachword[0].split("|")[0].replace("_", " ").title().strip()
            )
    return filtered_words


def get_highest_similarity_score(wordlist, wrd):
    """
    This function takes the given word along with the wordlist and then gives out the max-score which is the levenshtein distance for the wrong answers
    because we need the options which are very different from one another but relating to the same context.
    """
    score = []
    normalized_levenshtein = NormalizedLevenshtein()
    for each in wordlist:
        score.append(normalized_levenshtein.similarity(each.lower(), wrd.lower()))
    return max(score)


def sense2vec_get_words(word, s2v, topn, question):
    """
    This function takes the input word, sentence to vector model and top similar words and also the question
    Then it computes the sense of the given word
    then it gets the words which are of same sense but are most similar to the given word
    after that we we return the list of words which satisfy the above mentioned criteria
    """
    output = []
    # print ("word ",word)
    try:
        sense = s2v.get_best_sense(
            word,
            senses=[
                "NOUN",
                "PERSON",
                "PRODUCT",
                "LOC",
                "ORG",
                "EVENT",
                "NORP",
                "WORK OF ART",
                "FAC",
                "GPE",
                "NUM",
                "FACILITY",
            ],
        )
        most_similar = s2v.most_similar(sense, n=topn)
        # print (most_similar)
        output = filter_same_sense_words(sense, most_similar)
        # print ("Similar ",output)
    except:
        output = []

    threshold = 0.6
    final = [word]
    checklist = question.split()
    for x in output:
        if (
            get_highest_similarity_score(final, x) < threshold
            and x not in final
            and x not in checklist
        ):
            final.append(x)

    return final[1:]


def mmr(doc_embedding, word_embeddings, words, top_n, lambda_param):
    """
    The mmr function takes document and word embeddings, along with other parameters, and uses the Maximal Marginal Relevance (MMR) algorithm to extract a specified number of keywords/keyphrases from the document. The MMR algorithm balances the relevance of keywords with their diversity, helping to select keywords that are both informative and distinct from each other.
    """

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphrase
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(
            word_similarity[candidates_idx][:, keywords_idx], axis=1
        )

        # Calculate MMR
        mmr = (lambda_param) * candidate_similarities - (
            1 - lambda_param
        ) * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)  # type: ignore
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]


def get_distractors_wordnet(word):
    """
    the get_distractors_wordnet function uses WordNet to find a relevant synset for the input word and then generates distractor words by looking at hyponyms of the hypernym associated with the input word. These distractors are alternative words related to the input word and can be used, for example, in educational or language-related applications to provide choices for a given word.
    """
    distractors = []
    try:
        syn = wn.synsets(word, "n")[0]

        word = word.lower()
        orig_word = word
        if len(word.split()) > 0:
            word = word.replace(" ", "_")
        hypernym = syn.hypernyms()  # type:ignore
        if len(hypernym) == 0:
            return distractors
        for item in hypernym[0].hyponyms():
            name = item.lemmas()[0].name()
            # print ("name ",name, " word",orig_word)
            if name == orig_word:
                continue
            name = name.replace("_", " ")
            name = " ".join(w.capitalize() for w in name.split())
            if name is not None and name not in distractors:
                distractors.append(name)
    except:
        print("Wordnet distractors not found")
    return distractors


def get_distractors(
    word, origsentence, sense2vecmodel, sentencemodel, top_n, lambdaval
):
    """
    this function generates distractor words (answer choices) for a given target word in the context of a provided sentence. It selects distractors based on their similarity to the target word's context and ensures that the target word itself is not included among the distractors. This function is useful for creating multiple-choice questions or answer options in natural language processing tasks.
    """
    distractors = sense2vec_get_words(word, sense2vecmodel, top_n, origsentence)
    # print ("distractors ",distractors)
    if len(distractors) == 0:
        return distractors
    distractors_new = [word.capitalize()]
    distractors_new.extend(distractors)
    # print ("distractors_new .. ",distractors_new)

    embedding_sentence = origsentence + " " + word.capitalize()
    # embedding_sentence = word
    keyword_embedding = sentencemodel.encode([embedding_sentence])
    distractor_embeddings = sentencemodel.encode(distractors_new)

    # filtered_keywords = mmr(keyword_embedding, distractor_embeddings,distractors,4,0.7)
    max_keywords = min(len(distractors_new), 5)
    filtered_keywords = mmr(
        keyword_embedding,
        distractor_embeddings,
        distractors_new,
        max_keywords,
        lambdaval,
    )
    # filtered_keywords = filtered_keywords[1:]
    final = [word.capitalize()]
    for wrd in filtered_keywords:
        if wrd.lower() != word.lower():
            final.append(wrd.capitalize())
    final = final[1:]
    return final


def get_mca_questions(context: str):
    """
    this function generates multiple-choice questions based on a given context. It summarizes the context, extracts important keywords, generates questions related to those keywords, and provides randomized answer choices, including the correct answer, for each question.
    """
    summarized_text = summarizer(context, summary_model, summary_tokenizer)

    # imp_keywords = get_keywords(context ,summarized_text)
    imp_keywords = get_keywords(context)
    output_list = []
    for answer in imp_keywords:
        output = ""
        ques = get_question(summarized_text, answer, question_model, question_tokenizer)

        distractors = get_distractors(
            answer.capitalize(), ques, s2v, sentence_transformer_model, 40, 0.2
        )

        output = output + ques + "\n"
        if len(distractors) == 0:
            distractors = imp_keywords

        if len(distractors) > 0:
            random_integer = random.randint(0, 3)
            alpha_list = ["(a)", "(b)", "(c)", "(d)"]
            for d, distractor in enumerate(distractors[:4]):
                if d == random_integer:
                    output = output + alpha_list[d] + answer + "\n"
                else:
                    output = output + alpha_list[d] + distractor + "\n"
            output = output + alpha_list[random_integer] + "\n\n"

        output_list.append(output)

    mca_questions = output_list
    return mca_questions

# transcript_text= "we all use smart phones but have you ever wondered how much data it generates in the form of texts phone calls emails photos videos searches and music approximately 40 exabytes of data gets generated every month by a single smartphone user now imagine this number multiplied by 5 billion smartphone users that's a lot for our mind even process isn't it in fact this amount of data is quite a lot for traditional computing systems to handle and this massive amount of data is what we term as big data let's have a look at the data generated per minute on the internet 2.1 million snaps are shared on snapchat 3.8 million search queries are made on google one million people log onto facebook 4.5 million videos are watched on youtube 188 million emails are sent that's a lot of data so how do you classify any data as big data this is possible with the concept of five v's volume velocity variety veracity and value let us understand this with an example from the health care industry hospitals and clinics across the world generate massive volumes of data 2314 exabytes of data are collected annually in the form of patient records and test results all this data is generated at a very high speed which attributes to the velocity of big data variety refers to the various data types such as structured semi-structured and unstructured data examples include excel records log files and x-ray images accuracy and trustworthiness of the generated data is termed as veracity analyzing all this data will benefit the medical sector by enabling faster disease detection better treatment and reduced cost this is known as the value of big data but how do we store and process this big data to do this job we have various frameworks such as cassandra hadoop and spark let us take hadoop as an example and see how hadoop stores and processes big data hadoop uses a distributed file system known as hadoop distributed file system to store big data if you have a huge file your file will be broken down into smaller chunks and stored in various machines not only that when you break the file you also make copies of it which goes into different nodes this way you store your big data in a distributed way and make sure that even if one machine fails your data is safe on another mapreduce technique is used to process big data a lengthy task a is broken into smaller tasks b c and d now instead of one machine three machines take up each task and complete it in a parallel fashion and assemble the results at the end thanks to this the processing becomes easy and fast this is known as parallel processing now that we have stored and processed our big data we can analyze this data for numerous applications in games like halo 3 and call of duty designers analyze user data to understand at which stage most of the users pause restart or quit playing this insight can help them rework on the story line of the game and improve the user experience which in turn reduces the customer churn rate similarly big data also helped with disaster management during hurricane sandy in 2012. it was used to gain a better understanding of the storm's effect on the east coast of the u.s and necessary measures were taken it could predict the hurricane's landfall five days in advance which wasn't possible earlier these are some of the clear indications of how valuable big data can be once it is accurately processed and analyzed so here's a question for you which of the following statements is not correct about hadoop distributed file system hdfs a hdfs is the storage layer of hadoop b data gets stored in a distributed manner in hdfs c hdfs performs parallel processing of data d smaller chunks of data are stored on multiple data nodes in hdfs give it a thought and leave your answers in the comment section below three lucky winners will receive amazon gift vouchers now that you have learned what big data is what do you think will be the most significant impact of big data in the future let us know in the comments below if you enjoyed this video it would only take a few seconds to like and share it also to subscribe to our channel if you haven't yet and hit the bell icon to get instant notifications about our new content stay tuned and keep learning [Music] you"
youtube_url = st.session_state["youtube_url"]
def get_transcript(video_url):
    try:
        video_id = video_url.split("v=")[1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = ' '.join([entry['text'] for entry in transcript])
        return text
    except Exception as e:
        print(f"Error: {e}")
        return None
transcript_text = get_transcript(youtube_url)
final_questions = get_mca_questions(str(transcript_text))
print(final_questions)


# text_1 = "Itachi Uchiha, a central figure in the renowned anime and manga series Naruto, is a complex and enigmatic character whose presence resonates throughout the narrative. Initially introduced as a prodigious and seemingly emotionless member of the Uchiha clan, Itachi's true nature unfolds gradually, revealing a tragic tale of sacrifice and duty. Gifted with exceptional intellect and unparalleled prowess in ninjutsu, genjutsu, and taijutsu, Itachi earned widespread admiration and fear within the shinobi world. However, his loyalty to the village led him down a path of internal conflict, forcing him to make agonizing choices for the greater good. The revelation of his painful past, marked by the slaughter of his own clan and the burden of shouldering the blame to protect his younger brother, Sasuke, adds layers of depth to his character. Itachi's philosophies, complex moral compass, and the profound impact he leaves on the series make him an iconic and unforgettable presence in the world of anime, transcending the boundaries of a traditional antagonist and earning him a place as one of the most compelling characters in the Naruto universe."
# context = "Madara Uchiha, a central and formidable character in the Naruto manga and anime series, is defined by a complex and rich narrative that spans the vast lore of the shinobi world. Born into the prestigious Uchiha clan during the tumultuous Warring States era, Madara exhibited prodigious talent and experienced the harsh realities of conflict. His pivotal bond with Hashirama Senju, a member of the rival Senju clan, led to the establishment of Konohagakure, the Hidden Leaf Village. However, their vision of lasting peace was shattered by ideological differences and clan animosity, propelling Madara on a gradual descent into darkness. Madara's prowess as a shinobi is exemplified by his mastery of the Sharingan, a unique ocular ability within the Uchiha lineage that grants heightened visual perception and predictive capabilities. Later in his life, he evolves further, awakening the Rinnegan, a legendary dojutsu associated with the divine, amplifying his abilities and granting control over gravity and space-time. Central to Madara's character is his unyielding pursuit of power and the realization of the Infinite Tsukuyomi a genjutsu that aims to cast the entire world into an eternal dream state. This ambitious endeavor stems from his belief in imposing order through dominance, fueled by a desire to end the perpetual cycle of hatred and conflict that has plagued the ninja world for generations. Madara's impact on the Naruto narrative extends far beyond his physical presence. Even in death, his legacy influences the destinies of key characters, notably Sasuke Uchiha and Naruto Uzumaki. The intricate interplay between personal history, ideology, and the pursuit of power is encapsulated in Madara's character, making him a compelling exploration of the consequences of unbridled ambition. In the latter part of the series, Madara undergoes a startling transformation, becoming the vessel for Kaguya Otsutsuki, a celestial being of immense power. This twist adds layers to Madara's character, blurring the lines between villainy and tragic destiny, as he becomes an unwitting pawn in the cosmic struggles that transcend the mortal realm. Madara Uchiha's legacy is indelible, leaving an enduring mark on the Naruto series and its thematic exploration of perseverance, redemption, and the cyclical nature of conflict. His character serves as a poignant reminder of the intricate complexities that shape the destinies of those caught in the turbulent currents of the ninja world."
# from courses import generate_transcription,get_video_id,youtube_url
# def questions():
#     video_id = get_video_id(youtube_url)
#     transcript=generate_transcription(video_id)
#     # text_2="Deep learning is the subset of machine learning methods based on artificial neural networks with representation learning. The adjective deep refers to the use of multiple layers in the network. Methods used can be either supervised, semi-supervised or unsupervised.Deep-learning architectures such as deep neural networks, deep belief networks, recurrent neural networks, convolutional neural networks and transformers have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics, drug design, medical image analysis, climate science, material inspection and board game programs, where they have produced results comparable to and in some cases surpassing human expert performance.Artificial neural networks (ANNs) were inspired by information processing and distributed communication nodes in biological systems. ANNs have various differences from biological brains. Specifically, artificial neural networks tend to be static and symbolic, while the biological brain of most living organisms is dynamic (plastic) and analog.[6][7] ANNs are generally seen as low quality models for brain function. Deep learning is a class of machine learning algorithms that 199–200  uses multiple layers to progressively extract higher-level features from the raw input. For example, in image processing, lower layers may identify edges, while higher layers may identify the concepts relevant to a human such as digits or letters or faces.From another angle to view deep learning, deep learning refers to computer-simulate or automate human learning processes from a source (e.g., an image of dogs) to a learned object (dogs). Therefore, a notion coined as deeper learning or deepest learning makes sense. The deepest learning refers to the fully automatic learning from a source to a final learned object. A deeper learning thus refers to a mixed learning process: a human learning process from a source to a learned semi-object, followed by a computer learning process from the human learned semi-object to a final learned object.Most modern deep learning models are based on multi-layered artificial neural networks such as convolutional neural networks and transformers, although they can also include propositional formulas or latent variables organized layer-wise in deep generative models such as the nodes in deep belief networks and deep Boltzmann machines.In deep learning, each level learns to transform its input data into a slightly more abstract and composite representation. In an image recognition application, the raw input may be a matrix of pixels; the first representational layer may abstract the pixels and encode edges; the second layer may compose and encode arrangements of edges; the third layer may encode a nose and eyes; and the fourth layer may recognize that the image contains a face. Importantly, a deep learning process can learn which features to optimally place in which level on its own. This does not eliminate the need for hand-tuning; for example, varying numbers of layers and layer sizes can provide different degrees of abstraction.The word deep in deep learning refers to the number of layers through which the data is transformed. More precisely, deep learning systems have a substantial credit assignment path (CAP) depth. The CAP is the chain of transformations from input to output. CAPs describe potentially causal connections between input and output. For a feedforward neural network, the depth of the CAPs is that of the network and is the number of hidden layers plus one (as the output layer is also parameterized). For recurrent neural networks, in which a signal may propagate through a layer more than once, the CAP depth is potentially unlimited. No universally agreed-upon threshold of depth divides shallow learning from deep learning, but most researchers agree that deep learning involves CAP depth higher than 2. CAP of depth 2 has been shown to be a universal approximator in the sense that it can emulate any function. Beyond that, more layers do not add to the function approximator ability of the network. Deep models (CAP > 2) are able to extract better features than shallow models and hence, extra layers help in learning the features effectively.Deep learning architectures can be constructed with a greedy layer-by-layer method. Deep learning helps to disentangle these abstractions and pick out which features improve performance.For supervised learning tasks, deep learning methods enable elimination of feature engineering, by translating the data into compact intermediate representations akin to principal components, and derive layered structures that remove redundancy in representation.Deep learning algorithms can be applied to unsupervised learning tasks. This is an important benefit because unlabeled data are more abundant than the labeled data. Examples of deep structures that can be trained in an unsupervised manner are deep belief networks.Machine learning models are now adept at identifying complex patterns in financial market data. Due to the benefits of artificial intelligence, investors are increasingly utilizing deep learning techniques to forecast and analyze trends in stock and foreign exchange markets"
#     final_questions = get_mca_questions(transcript)
#     return final_questions

































# for q in final_questions:
#     print(q)
# Import other necessary libraries and functions

# # Streamlit app
# def main():
#     st.title("MCQ Generator")

#     # Input text area for the user to enter the context
#     context = st.text_area("Enter the context for MCQ generation:")

#     # Button to generate MCQs
#     generate_button = st.button("Generate MCQs")

#     if generate_button:
#         if context:
#             # Generate MCQs
#             questions = get_mca_questions(context)

#             # Display questions and radio buttons
#             for i, question in enumerate(questions, 1):
#                 st.subheader(f"Question {i}:")
#                 st.markdown(question, unsafe_allow_html=True)

#                 # Radio buttons for answer choices
#                 options = question.split('\n')[1:-2]  # Extract answer choices
#                 user_choice = st.radio("Select your answer:", options, key=f"user_choice_{i}")

#                 # Display correct/wrong answer
#                 submit_button_key = f"submit_button_{i}"
#                 if st.button("Submit Answer", key=submit_button_key):
#                     correct_answer = options[0]
#                     if user_choice == correct_answer:
#                         st.success("Correct! 🟢")
#                     else:
#                         st.error(f"Wrong! Correct answer is {correct_answer}. 🔴")

# if __name__ == "__main__":
#     main()




# import streamlit as st
# from Quiz import get_mca_questions  # Assuming get_mca_questions is implemented correctly

# st.set_page_config(page_title="Quiz")

# # Streamlit app
# def main():
    
#     st.title("MCQ Generator")

#     # Input text area for the user to enter the context
#     context = "Madara Uchiha, a central and formidable character in the Naruto manga and anime series, is defined by a complex and rich narrative that spans the vast lore of the shinobi world. Born into the prestigious Uchiha clan during the tumultuous Warring States era, Madara exhibited prodigious talent and experienced the harsh realities of conflict. His pivotal bond with Hashirama Senju, a member of the rival Senju clan, led to the establishment of Konohagakure, the Hidden Leaf Village. However, their vision of lasting peace was shattered by ideological differences and clan animosity, propelling Madara on a gradual descent into darkness. Madara's prowess as a shinobi is exemplified by his mastery of the Sharingan, a unique ocular ability within the Uchiha lineage that grants heightened visual perception and predictive capabilities. Later in his life, he evolves further, awakening the Rinnegan, a legendary dojutsu associated with the divine, amplifying his abilities and granting control over gravity and space-time. Central to Madara's character is his unyielding pursuit of power and the realization of the Infinite Tsukuyomi a genjutsu that aims to cast the entire world into an eternal dream state. This ambitious endeavor stems from his belief in imposing order through dominance, fueled by a desire to end the perpetual cycle of hatred and conflict that has plagued the ninja world for generations. Madara's impact on the Naruto narrative extends far beyond his physical presence. Even in death, his legacy influences the destinies of key characters, notably Sasuke Uchiha and Naruto Uzumaki. The intricate interplay between personal history, ideology, and the pursuit of power is encapsulated in Madara's character, making him a compelling exploration of the consequences of unbridled ambition. In the latter part of the series, Madara undergoes a startling transformation, becoming the vessel for Kaguya Otsutsuki, a celestial being of immense power. This twist adds layers to Madara's character, blurring the lines between villainy and tragic destiny, as he becomes an unwitting pawn in the cosmic struggles that transcend the mortal realm. Madara Uchiha's legacy is indelible, leaving an enduring mark on the Naruto series and its thematic exploration of perseverance, redemption, and the cyclical nature of conflict. His character serves as a poignant reminder of the intricate complexities that shape the destinies of those caught in the turbulent currents of the ninja world."
#     questions= get_mca_questions(context)

#     # questions = [
#     #     "What is a subset of machine learning methods?\n(a)Machine learning algorithms\n(b)Rnns\n(c)Fpgas\n(d)deep learning\n(d)\n\n",
#     #     "What is a subset of machine learning methods based on artificial neural networks?\n(a)deep\n(b)Burnt\n(c)Like\n(d)Mazzy\n(a)\n\n",
#     #     "What is Deep learning based on?\n(a)recurrent neural networks\n(b)deep\n(c)recurrent neural networks\n(d)uses multiple layers\n(a)\n\n",
#     #     "How many layers does deep learning use?\n(a)deep learning\n(b)deep\n(c)recurrent neural networks\n(d)uses multiple layers\n(d)\n\n",
#     #     "Deep learning is a subset of machine learning based on what artificial neural network?\n(a)deep learning\n(b)anns\n(c)recurrent neural networks\n(d)uses multiple layers\n(b)\n\n",
#     #     "What type of transformers are used in deep learning?\n(a)Snowpiercer\n(b)Incredibles\n(c)Terminator\n(d)transformers\n(d)\n\n",
#     #     "Deep learning is a subset of what type of learning methods?\n(a)machine translation\n(b)Native speakers\n(c)Sentence structure\n(d)Written form\n(a)\n\n",
#     #     "What is a subset of machine learning methods based on?\n(a)deep learning\n(b)deep\n(c)deep belief networks\n(d)uses multiple layers\n(c)\n\n",
#     #     "What is the purpose of deep learning?\n(a)deep learning\n(b)identify edges\n(c)recurrent neural networks\n(d)uses multiple layers\n(b)\n\n",
#     #     "What field of study is deep learning a subset of?\n(a)Subfield\n(b)Math background\n(c)bioinformatics\n(d)Software engineering\n(c)\n\n",
#     #     "Deep learning is a subset of what type of machine learning?\n(a)Biological sciences\n(b)drug design\n(c)Pathophysiology\n(d)Regenerative medicine\n(b)\n\n",
#     #     "What type of processing is deep learning a subset of?\n(a)Deep learning\n(b)Anns\n(c)Large data sets\n(d)natural language processing\n(d)\n\n",
#     #     "What is an example of a type of deep learning?\n(a)deep learning\n(b)deep\n(c)recurrent neural networks\n(d)medical image analysis\n(d)\n\n",
#     #     "Deep learning is a subset of what type of science?\n(a)Deniers\n(b)climate science\n(c)Creationists\n(d)Climatologists\n(b)\n\n",
#     #     "What type of inspection is a part of deep learning?\n(a)deep learning\n(b)deep\n(c)material inspection\n(d)uses multiple layers\n(c)\n\n",
#     # ]

#     for i, question in enumerate(questions, 1):
#         st.subheader(f"Question {i}:")

#         parts = question.split("\n")
#         que = parts[0]
#         options = parts[1:-3]
#         correct_answer = options[-1].strip()[1]
#         st.markdown(que, unsafe_allow_html=True)

#         # Radio buttons for answer choices
#         user_choice = st.radio(
#             f"Select your answer for Question {i}:", options,index=None)

#         # Check if the selected choice is correct
#         if user_choice:
#             user_choice = user_choice.strip()[1]
#             if user_choice == correct_answer:
#                 st.write(f'Your answer "{user_choice}" is correct!')
#             else:
#                 st.write(f"Sorry, the correct answer is: {correct_answer}")
#         else:
#             st.write("Please select an answer before revealing the correct one.")

# if __name__ == "__main__":
#     main()
