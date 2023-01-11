import streamlit as st
from keras.models import load_model
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import tensorflow as tf

st.set_page_config(layout="wide", page_title="Alzheimer's Disease Stage Classification")

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

set_png_as_page_bg("C:\\Users\\Musoo\\OneDrive\\Desktop\\Alzheimer's second deployment\\ai-robotic-technology_53876-91628.webp")


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
        return "You do not show any signs of Alzheimer's disease at this time. It is important to continue maintaining a healthy lifestyle including a balanced diet, regular exercise, and engaging in mentally stimulating activities to promote cognitive health."
    elif predicted_class == 3:
        return "You may have some problems with memory, but you can still carry out most of your daily activities. you may benefit from support groups and counseling."
    else:
        return "Recommendation: Please consult with a doctor or healthcare professional for a personalized diagnosis and recommendations."



st.markdown(
    "<h1><span style='color: white;'>Alzheimer's Disease Stage Classification</span></h1>",
    unsafe_allow_html=True
)
st.markdown("**🚫 This App is relevant to doctors and anyone else that has already been diagnosed with Alzheimer's disease 🚫**")


st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose a section", ["Homepage", "Project Overview"])

col1, col2 = st.columns(2)

# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def fix_image(upload):
    image = Image.open(upload)
    col1.write("Brain scan image :camera:")
    col1.image(image)

    fixed = image
    col2.write("Results :wrench:")
    col2.image(fixed)

    predicted_class, predicted_score = predict(fixed)
    class_name = [class_name for class_name, class_index in class_indices.items() if class_index == predicted_class]
    col2.write("Predicted class: " + class_name[0])
    col2.write("Predicted score: " + str(predicted_score))
    col2.write("Recommendation: " + get_recommendation(predicted_class))
    col2.empty()
    col2.write("Please consult with a doctor or healthcare professional for a personalized diagnosis and recommendations.")
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download fixed image", convert_image(fixed), "fixed.png", "image/png")

    st.sidebar.empty()
    st.sidebar.markdown("**Made by Team 4**")





if app_mode == "Homepage":
    my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if my_upload is not None:
        fix_image(upload=my_upload)
    else:
        fix_image("C:\\Users\\Musoo\\OneDrive\\Desktop\\Alzheimer's second deployment\\brain image.jpeg")
else:
    st.header("Project Overview")
    st.write("""
    This project is designed to classify the stage of Alzheimer's disease in patients by analyzing brain scan images. 
    The model was trained on a dataset of brain scans and uses a convolutional neural network (CNN) to make predictions.
    """)
    st.write("Alzheimer's Disease is a progressive disorder that destroys memory and other important mental functions. It is the most common cause of dementia among older adults. The main aim of creating this app is to classify the stage of dementia for patients already diagnosed with the Alzheimer's disease.")
    st.write("Symptoms of Alzheimer's disease include: ")
    st.write("- Memory loss that disrupts daily life.")
    st.write("- Challenges in planning or solving problems.")
    st.write("- Difficulty completing familiar tasks.")
    st.write("- Confusion with time or place.")
    st.write("- Trouble understanding visual images and spatial relationships.")
    st.write("- New problems with words in speaking or writing.")
    st.write("- Misplacing things and losing the ability to retrace steps.")
    st.write("- Decreased or poor judgment.")
    st.write("- Withdrawal from work or social activities.")
    st.write("- Changes in mood and personality.")
