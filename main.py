import streamlit as st
import AwesomeObjectDetector as mydetector
from PIL import Image,ImageDraw
import torch

from transformers import DetrFeatureExtractor

feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")


st.title("Benedikt's first Streamlit App")

color='Red'
st.markdown("<p style='color:" + color +"'>This is my first time using it!</p>", unsafe_allow_html=True)


file=st.file_uploader("Pick a image!",type=['png','jpg','jfif'])

if st.button("GO GO GO!"):
    if file is None:
        st.write("no file selected!!")
    else:
        image = Image.open(file)
        output=mydetector.object_detection(image)

        probas = output.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.9

        # rescale bounding boxes
        target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
        postprocessed_outputs = feature_extractor.post_process(output, target_sizes)
        bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]

        img_draw = ImageDraw.Draw(image)

        for bb in bboxes_scaled:
            shape = [(bb[0].item(), bb[1].item()), (bb[2].item(), bb[3].item())]
            img_draw.rectangle(shape, outline="red")




        st.image(image)



# Press the green button in the gutter to run the script.
#if __name__ == '__main__':
#    pass

