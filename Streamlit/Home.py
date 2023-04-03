import base64
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Home", page_icon=":earth_africa:")

def load_bootstrap():
    return st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

st.sidebar.markdown(
    '''
        <div style="border: thin solid black; border-radius: 5px;">
            <div style="background-image: url(data:image/png;base64,{}); background-repeat: no-repeat; height: 125px;">
                <a href='https://github.com/samya-ravenXI' style="position: absolute; z-index: 2; top: 45px; left: 30px;">
                    <img src="https://skillicons.dev/icons?i=github" alt="GitHub"/>
                </a>
                <h1 style="color: rgb(224, 224, 224); position: absolute; z-index: 2; top: 20px; left: 100px; font-family: sans-serif;">Sign Language Recognition</h1>
            </div>
            <div style="margin-top: 40px">
                <a href="https://github.com/samya-ravenXI/Sign-Language-Recognition" style="position: absolute; z-index: 2; top: 131px; left: 35px">
                    <img src="https://img.shields.io/badge/github-repo-white" alt="repo"/>
                </a>
                <a href="https://colab.research.google.com/drive/1ZgvceZNj3cT_-OrwEomMa8od80DG2UrX?usp=sharing" style="position: absolute; z-index: 2; top: 131px; right: 35px">
                    <img src="https://img.shields.io/badge/colab-notebook-orange" alt="repo"/>
                </a>
            </div>
        </div>
    '''.format(img_to_bytes('./icons/cover.jpg')),
    unsafe_allow_html=True)

with st.container():
    st.title("Sign Language Recognition")
    for i in range(2):
        st.markdown('#')
    st.caption('This project utilizes MediaPipe Holisitcs and OpenCV to classify the key values of the alphanumeric characters of ASL.')
    st.caption('It also contains a learning preview of a huge variety of hand signs from the ASLLVD Dataset.')
    st.caption('Tip: The model is simplified for easy hosting and only contains data records of the right hand. Please recollect and retrain the model to accomodate further variations')

    for i in range(2):
        st.markdown('#')
    st.markdown('---')

    col1, col2, col3 = st.columns(3, gap='large')
    with col1:
        st.empty()
        st.empty()
        st.markdown("<a href='https://docs.streamlit.io/library/get-started'><img src='data:image/png;base64,{}' class='img-fluid' width=80%/></a>".format(img_to_bytes('./icons/streamlit.png')), unsafe_allow_html=True)
    with col2:
        st.markdown("<a href='https://opencv.org/'><img src='data:image/png;base64,{}' class='img-fluid' width=70%/></a>".format(img_to_bytes('./icons/opencv.png')), unsafe_allow_html=True)
    with col3:
        st.markdown("<a href='https://google.github.io/mediapipe/solutions/holistic.html'><img src='data:image/png;base64,{}' class='img-fluid' width=70%/></a>".format(img_to_bytes('./icons/mediapipe.png')), unsafe_allow_html=True)
