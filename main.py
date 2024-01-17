from typing import List
from fastapi import FastAPI, UploadFile, Response
from PIL import Image
from io import BytesIO
import numpy as np
from nst.hub_api import HUB_NST
import matplotlib.pyplot as plt

app = FastAPI()

def image_to_numpy(data):
    return np.array(Image.open(BytesIO(data)))


@app.post("/")
async def process_and_return(files: List[UploadFile]):
    style_img = image_to_numpy(await files[0].read())
    content_img = image_to_numpy(await files[1].read())
    nst = HUB_NST()
    result = nst.get_stylized_image(style_img, content_img)
    buf = BytesIO()
    image = Image.fromarray(np.uint8(result*255))
    image.save(buf, format="PNG")
    bytes = buf.getvalue()
    return Response(content=bytes, media_type="image/png")