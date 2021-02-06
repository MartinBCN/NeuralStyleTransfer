from PIL import Image

from NeuralStyleTransfer.loader import ImageLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import streamlit as st

from NeuralStyleTransfer.style_transfer import NeuralStyleTransfer

unloader = transforms.ToPILImage()  # reconvert into PIL image


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated


def main():

    style = st.selectbox(label='Select Image Style', options=['Mona Lisa', 'Starry Night', 'Picasso'])

    periods = st.slider(label='Periods', min_value=2, max_value=30, value=10, step=1)
    resolution = st.slider(label='Resolution', min_value=128, max_value=1024, value=128, step=128)

    loader = ImageLoader(resolution)
    styles = loader.load_styles()
    style_tensor = styles[style]

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # st.image(image, caption='Uploaded Image.', use_column_width=True)

        content_tensor = loader.pil_to_tensor(image)

        input_img = content_tensor.clone()

        print('Building the style transfer model..')
        model = NeuralStyleTransfer(num_steps=periods)

        model.get_style_model_and_losses(style_img=style_tensor, content_img=content_tensor)

        result = model.fit_transform(input_img)

        st.write('Optimization finished')

        fig = plt.figure()
        imshow(result)

        st.write(fig)


if __name__ == '__main__':
    main()
