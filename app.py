from PIL import Image
from matplotlib.axes import Axes

from NeuralStyleTransfer.loader import ImageLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import streamlit as st

from NeuralStyleTransfer.style_transfer import NeuralStyleTransfer

unloader = transforms.ToPILImage()  # reconvert into PIL image


def imshow(tensor, title=None, ax: Axes = None):
    if ax is None:
        ax = plt
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    ax.imshow(image)
    ax.axis('off')
    if title is not None:
        ax.title(title)


def main():
    st.set_page_config(
        layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
        initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
        page_title=None,  # String or None. Strings get appended with "• Streamlit".
        page_icon=None,  # String, anything supported by st.image, or None.
    )

    col1, col2, col3 = st.beta_columns(3)

    style = col1.selectbox(label='Select Image Style', options=['Mona Lisa', 'Starry Night', 'Picasso'])
    periods = col2.slider(label='Periods', min_value=2, max_value=30, value=2, step=1)
    resolution = col3.slider(label='Resolution', min_value=128, max_value=1024, value=128, step=128)

    loader = ImageLoader(resolution)
    styles = loader.load_styles()
    style_tensor = styles[style]

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    cols = st.beta_columns(3)
    okay = cols[1].button(label='Calculate')

    if (uploaded_file is not None) and okay:
        image = Image.open(uploaded_file)

        content_tensor = loader.pil_to_tensor(image)

        input_img = content_tensor.clone()

        print('Building the style transfer model..')
        model = NeuralStyleTransfer(num_steps=periods)

        model.get_style_model_and_losses(style_img=style_tensor, content_img=content_tensor)

        result = model.fit_transform(input_img)

        st.write('Optimization finished')

        fig, axes = plt.subplots(1, 3)
        imshow(content_tensor, ax=axes[0])
        imshow(result, ax=axes[1])
        imshow(style_tensor, ax=axes[2])

        st.write(fig)


if __name__ == '__main__':
    main()
