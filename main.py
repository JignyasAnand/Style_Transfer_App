from typing import List
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO
import numpy as np
from nst.hub_api import HUB_NST
import matplotlib.pyplot as plt

app = FastAPI()

def image_to_numpy(data):
    return np.array(Image.open(BytesIO(data)))


@app.post("/")
async def test_process(files: List[UploadFile]):
    style_img = image_to_numpy(await files[0].read())
    content_img = image_to_numpy(await files[1].read())
    nst = HUB_NST()
    result = nst.get_stylized_image(style_img, content_img)
    plt.imshow(result[0])
    plt.show()
    return {"Hello": len(files)}