import replicate
from dotenv import load_dotenv
import os
load_dotenv()

output = replicate.run(
    "viktorfa/oot_diffusion:9f8fa4956970dde99689af7488157a30aa152e23953526a605df1d77598343d7",
    input={
        "seed": 0,
        "steps": 20,
        "model_image": "https://res.cloudinary.com/dgsrxvev1/image/upload/v1713973800/phu_dsls3s.jpg",
        "garment_image": "https://i.etsystatic.com/21090857/r/il/18dbe0/2967196916/il_794xN.2967196916_t4g1.jpg",
        "guidance_scale": 2
    }
)
print(output)