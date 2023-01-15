import streamlit as st
from keras.models import load_model
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import tensorflow as tf
import keras

st.set_page_config(layout="wide", page_title="BRAINLENS")

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file) 
    page_bg_img = '''
   <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    </style>

    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return 
set_png_as_page_bg("C:/Users/User-PC/Downloads/22af95b7ca4bcfa6f52f9370414f146b.jpg")



@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("model.h5")
    model.load_weights("weights.h5")
    return model

with st.spinner('Model is being loaded..'):
    model = load_model()

# Get the class indices
class_indices = {"MildDemented": 0, "ModerateDemented": 1, "NonDemented": 2, "VeryMildDemented": 3}

def predict(image):
    # Pre-process the image data
    image = image.resize((150, 150))
    image = image.convert("RGB")
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Run the image data through the model
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions[0])
    predicted_score = predictions[0][predicted_class]

    return predicted_class, predicted_score

def get_recommendation(predicted_class):
    if predicted_class == 0:
        return "You may have some memory problems, but you can still handle your own care and financial affairs. You may benefit from support groups and counseling."
    elif predicted_class == 1:
        return "You have more memory problems and will have trouble with daily activities. you will need help with many aspects of your care, such as bathing and dressing. Support groups and counseling may be beneficial."
    elif predicted_class == 2:
        return "You do not show any signs of dementia at this time. It is important to continue maintaining a healthy lifestyle including a balanced diet, regular exercise, and engaging in mentally stimulating activities to promote cognitive health."
    elif predicted_class == 3:
        return "You may have some problems with memory, but you can still carry out most of your daily activities. you may benefit from support groups and counseling."
    else:
        return "Recommendation: Please consult with a doctor or healthcare professional for a personalized diagnosis and recommendations."



st.markdown(
    "<h1><span style='color: blue;'text-align: center';>BRAINLENS</span></h1>",
    unsafe_allow_html=True
)
st.markdown("**🚫 This App is relevant to doctors and anyone else that has already been diagnosed with Alzheimer's disease 🚫**")
st.markdown("**To get your results, select the Prediction section on the Navigation section on your left**")

st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose a section", ["Project Overview", "Dementia symptoms", "Prediction section"])



# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def fix_image(upload):
    image = Image.open(upload)
    fixed = image
    st.markdown("<h3><span style='color: blue;'>Image Results</span></h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')
    with col2:
        st.image(fixed)
    with col3:
        st.write(' ')
   

    predicted_class, predicted_score = predict(fixed)
    class_name = [class_name for class_name, class_index in class_indices.items() if class_index == predicted_class]
    st.write("Predicted class: " + class_name[0])
    st.write("Predicted score: " + str(predicted_score))
    st.write("Recommendation: " + get_recommendation(predicted_class))
    st.empty()
    st.write("Please consult with a doctor or healthcare professional for a personalized diagnosis and recommendations.")
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download fixed image", convert_image(fixed), "fixed.png", "image/png")

    st.sidebar.empty()
    st.sidebar.markdown("**Made by Team 4**")


if app_mode == "Prediction section":
    my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if my_upload is not None:
        fix_image(upload=my_upload)
    else:
        fix_image("C:/Users/User-PC/sample_deploy/WhatsApp Image 2023-01-12 at 14.38.58.jpeg")
elif app_mode == "Project Overview":
    st.markdown("<h3><span style='color: blue;'>Alzheimer's Disease(AD)</span></h1>", unsafe_allow_html=True)
    st.write("""
    Alzheimer's Disease is a progressive disorder that destroys memory and other important mental functions. It is the most common cause of dementia among older adults. According to Alzheimer's association, AD is incurable and the sixth leading cause of death in the united states and is expected to become four times as of now by the end of 2050. Although there is no cure for Alzheimers disease, getting a diagnosis quickly at an early stage helps patients. It allows them to access help and support, get treatment to manage their symptoms, and plan for the future. However, it can be challenging to diagnose Alzheimers disease, which can lead to suboptimal patient care. 
    """)
    st.write("To solve this problem we created Brainlens which is a machine learning application that correctly classifies the stage of one's alzheimer's disease and advises on the next course of action.")
    st.video("https://www.youtube.com/watch?v=RT907zjpZUM")
else:
    st.write("**As mentioned in the Project Overview section, Alzheimer's is the main cause of dementia. Symptoms of dementia include:** ")
    st.markdown(
    "<h3><span style='color: blue;'>Cognitive changes</span></h1>",
    unsafe_allow_html=True
    )
    st.write("- Memory loss, which is usually noticed by someone else")
    st.write("- Difficulty communicating or finding words")
    st.write("- Difficulty with visual and spatial abilities, such as getting lost while driving")
    st.write("- Difficulty reasoning or problem-solving")
    st.write("- Difficulty handling complex tasks")
    st.write("- Difficulty with planning and organizing")
    st.write("- Difficulty with coordination and motor functions")
    st.write("- Confusion and disorientation")
    st.markdown(
    "<h3><span style='color: blue;'>Psychological changes</span></h1>",
    unsafe_allow_html=True
    )
    st.write("- Personality changes")
    st.write("- Depression")
    st.write("- Anxiety")
    st.write("- Inappropriate behavior")
    st.write("- Paranoia")
    st.write("- Agitation")
    st.write("- Hallucinations")

