import replicate
from langchain.prompts import ChatPromptTemplate
from openai import OpenAI
from dotenv import load_dotenv
from app.prompts.prompt import review_synthesis_prompt_tmpl_str, generate_product_description_prompt_tmpl_str, data_visualization_tool_prompt_tmpl_str
import json
import os
load_dotenv()

OPENAI_KEY = os.getenv('OPENAI_API_KEY')

# REVIEW_SYSTHESIS_SYSTEM_MSG = """Bạn là một chuyên gia phân tích đánh giá, bạn sẽ được cung cấp một danh sách các đánh giá của một sản phẩm, hãy phân tích các đánh giá đó. Hãy đảm bảo kết quả trả về luôn luôn là một json với cấu trúc {"positiveCount" : đây là số lượng đánh giá tích cực , "negativeCount": đây là số lượng đánh giá tiêu cực, "trashCount" : đây là số lượng đánh giá không thể xác định được là tiêu cực hay tích cực, "positiveSumary" : "Đây là một đoạn tóm tắt mô tả ngắn về các đánh giá tích cực, độ dài đoạn tóm tắt khoảng 50 từ. Ví dụ: Hầu hết người mua đánh giá tích cực về chất lượng sản phẩm, bao gồm vải đẹp, chất jean dày dặn, co giãn tốt và form chuẩn. Một số khách hàng nhận xét sản phẩm đáng mua, đẹp, sang trọng và bền chắc. Đa số khách hàng hài lòng với dịch vụ giao hàng nhanh, đúng hẹn và đóng gói cẩn thận. Một số khách hàng đánh giá tích cực về sự nhiệt tình và trách nhiệm của shop.", "negativeSumary" : "Đây là một đoạn tóm tắt mô tả ngắn về các đánh giá tiêu cực,độ dài đoạn tóm tắt khoảng 30 từ. Ví dụ: Tuy nhiên, có một số nhận xét tiêu cực về khuy nút bị lỏng và màu không thích, có một nhận xét tiêu cực về việc nhầm hàng."}.
# """

# GENERATE_DESCRIPTION_SYSTEM_MSG = """Bạn là một trợ lý ảo trên nền tảng thương mại điện tử đang hỗ trợ người dùng viết mô tả cho sản phẩm của họ. Bạn sẽ được cung cấp một đoạn mô tả sản phẩm mẫu và một đoạn mô tả sơ lược về sản phẩm của khách hàng cung cấp ở dạng html, bao gồm các thẻ chứa thông tin sơ bộ có thể có thẻ hình ảnh hoặc các thẻ khác. Hãy viết lại mô tả sản phẩm dựa trên thông tin mà khách hàng cung cấp và tuân theo format của mô tả mẫu với từ ngữ phù hợp. Bạn có thể viết thêm mô tả theo các thông tin mà bạn biết về chất liệu, thiết kế, kích cỡ với từ ngữ càng sinh động càng tốt. Chú ý mô tả mẫu chỉ là thiết kế mẫu, không nhất thiết phải đầy đủ giống như mẫu, bạn phải dựa trên mô tả sơ lược do người dùng cung cấp và viết lại một cách nổi bật
# Đoạn mô tả sản phẩm mẫu phía dưới:
# ---------------------------------------
# <p class="QN2lPu"><strong>&Aacute;o sơ mi nam ngắn tay cổ thường tho&aacute;ng m&aacute;t kh&aacute;ng khuẩn, form đẹp dễ phối đồ</strong></p> <!-- This is name of product-->
# <p class="QN2lPu">⏩ Th&ocirc;ng tin sản phẩm:</p> <!-- This is section title-->
# <p class="QN2lPu">👉 Chất liệu: chất đũi thấm h&uacute;t tốt, tho&aacute;ng m&aacute;t</p> <!-- This is content of this section-->
# <p class="QN2lPu">&nbsp;</p>
# <p class="QN2lPu"><img style="display: block; margin-left: auto; margin-right: auto;" src="https://down-vn.img.susercontent.com/file/vn-11134207-7qukw-ley33b4kzpmyac" alt="" width="573" height="573"></p> <!-- This is image tag if user have provided image link-->
# <p class="QN2lPu"><video style="width: 612px; height: 306px; display: table; margin-left: auto; margin-right: auto;" controls="controls" width="612" height="306"> <source src="https://cvf.shopee.vn/file/api/v4/11110105/mms/vn-11110105-6ke15-lu7a25d0b1n547.16000081713323497.mp4" type="video/mp4"></video></p> <!-- This is video tag if user have provided video link-->
# <p class="QN2lPu"><strong>TH&Ocirc;NG TIN THƯƠNG HIỆU</strong></p>
# <p class="QN2lPu"><strong>LADOS </strong>l&agrave; Nh&agrave; ph&acirc;n phối chuy&ecirc;n sỉ &amp; lẻ c&aacute;c mặt h&agrave;ng thời trang chất lượng v&agrave; gi&aacute; cả phải chăng với thương hiệu LADOS. Ch&uacute;ng t&ocirc;i h&acirc;n hạnh v&agrave; lu&ocirc;n cố gắng để mang đến cho qu&yacute; kh&aacute;ch những sản phẩm chất lượng với gi&aacute; cả tốt nhất v&agrave; dịch vụ uy t&iacute;n. Tất cả c&aacute;c sản phẩm của shop đều được ch&uacute;ng t&ocirc;i tuyển chọn một c&aacute;ch kỹ lưỡng sao cho ph&ugrave; hợp với phong c&aacute;ch Ch&acirc;u &Aacute; v&agrave; bắt nhịp c&ugrave;ng xu hướng trẻ. Đến với ch&uacute;ng t&ocirc;i kh&aacute;ch h&agrave;ng c&oacute; thể y&ecirc;n t&acirc;m mua h&agrave;ng với nhiều mẫu m&atilde; được cập nhật thường xuy&ecirc;n v&agrave; nhiều khuyến mại hấp dẫn.</p>
# <p class="QN2lPu">📣 CH&Iacute;NH S&Aacute;CH MUA H&Agrave;NG</p> <!-- This is additional section title (if any)-->
# <p class="QN2lPu">👉 Cam kết chất lượng v&agrave; mẫu m&atilde; sản phẩm giống với h&igrave;nh ảnh.</p>  <!-- This is content of this section-->
# <p class="QN2lPu">👉 Ho&agrave;n tiền nếu sản phẩm kh&ocirc;ng giống với m&ocirc; tả.</p>
# <p class="QN2lPu">👉 ĐỔI TRẢ TRONG 7 NG&Agrave;Y NẾU KH&Ocirc;NG Đ&Uacute;NG MI&Ecirc;U TẢ</p>
# <p class="QN2lPu">&nbsp;</p>
# ----------------------------------------
# Đoạn mô tả sơ lược do người dùng cung cấp phía dưới:
# ----------------------------------------
# {prompt}
# ----------------------------------------
# Hãy đảm bảo rằng kết quả trả về luôn luôn chỉ là đoạn mã html và ngôn ngữ của phần mô tả dựa theo phần mô tả tôi cung cấp (ưu tiên tiếng việt) và phần mô tả không vượt quá 500 từ. 
# """

def getReviewSynthesis(query):
    client = OpenAI(
        api_key = OPENAI_KEY,
    )
    print("Query", query)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": review_synthesis_prompt_tmpl_str},
            {"role": "user", "content": query}
        ],
    )
    message = response.choices[0].message.content
    return message

def index(query):
    client = OpenAI(
        api_key = OPENAI_KEY,
    )
    print("Query", query)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "user", "content": query}
        ],
    )
    message = response.choices[0].message.content
    return message

def generateProductDesciption(shortDescription):
    client = OpenAI(
        api_key = OPENAI_KEY,
    )
    print("Query", shortDescription)

    SYSTEM_MSG = generate_product_description_prompt_tmpl_str.format(prompt=shortDescription)

    print(SYSTEM_MSG)

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": shortDescription}
        ],
    )
    message = response.choices[0].message.content
    return message

def generateChart(data: str):
    client = OpenAI(
        api_key = OPENAI_KEY,
    )
    print("Data", data)

    SYSTEM_MSG = data_visualization_tool_prompt_tmpl_str.format(user_prompt=data)

    print(SYSTEM_MSG)

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": data}
        ],
    )
    message = response.choices[0].message.content
    return message



def generateProductImage(prompt: str, context: str, garmentImage: str):
    comfyuiApiWorkflow = {
      "3": {
        "inputs": {
          "seed": 780862346192377,
          "steps": 35,
          "cfg": 8,
          "sampler_name": "dpmpp_2m",
          "scheduler": "karras",
          "denoise": 1,
          "model": [
            "4",
            0
          ],
          "positive": [
            "6",
            0
          ],
          "negative": [
            "7",
            0
          ],
          "latent_image": [
            "5",
            0
          ]
        },
        "class_type": "KSampler",
        "_meta": {
          "title": "KSampler"
        }
      },
      "4": {
        "inputs": {
          "ckpt_name": "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
        },
        "class_type": "CheckpointLoaderSimple",
        "_meta": {
          "title": "Load Checkpoint"
        }
      },
      "5": {
        "inputs": {
          "width": 1024,
          "height": 1024,
          "batch_size": 1
        },
        "class_type": "EmptyLatentImage",
        "_meta": {
          "title": "Empty Latent Image"
        }
      },
      "6": {
        "inputs": {
          "text": prompt,
          "clip": [
            "4",
            1
          ]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {
          "title": "CLIP Text Encode (Prompt)"
        }
      },
      "7": {
        "inputs": {
          "text": "(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgl, render, blender, digital art, manga, amateur:1.3), (3D, 3D Game, 3D Game Scene, 3D Character:1.1), acne",
          "clip": [
            "4",
            1
          ]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {
          "title": "CLIP Text Encode (Prompt)"
        }
      },
      "8": {
        "inputs": {
          "samples": [
            "3",
            0
          ],
          "vae": [
            "4",
            2
          ]
        },
        "class_type": "VAEDecode",
        "_meta": {
          "title": "VAE Decode"
        }
      },
      "10": {
        "inputs": {
          "image": "$10-0",
          "images": [
            "8",
            0
          ]
        },
        "class_type": "PreviewBridge",
        "_meta": {
          "title": "Preview Bridge (Image)"
        }
      },
      "11": {
        "inputs": {
          "channel": "red",
          "image": [
            "25",
            0
          ]
        },
        "class_type": "ImageToMask",
        "_meta": {
          "title": "Convert Image to Mask"
        }
      },
      "25": {
        "inputs": {
          "resolution": 1024,
          "image": [
            "10",
            0
          ]
        },
        "class_type": "OneFormer-COCO-SemSegPreprocessor",
        "_meta": {
          "title": "OneFormer COCO Segmentor"
        }
      },
      "26": {
        "inputs": {
          "image": "$26-0",
          "images": [
            "25",
            0
          ]
        },
        "class_type": "PreviewBridge",
        "_meta": {
          "title": "Preview Bridge (Image)"
        }
      },
      "27": {
        "inputs": {
          "image": "$27-0",
          "images": [
            "10",
            0
          ]
        },
        "class_type": "PreviewBridge",
        "_meta": {
          "title": "Preview Bridge (Image)"
        }
      },
      "28": {
        "inputs": {
          "weight": 0.5,
          "weight_faceidv2": 1.5,
          "weight_type": "linear",
          "combine_embeds": "concat",
          "start_at": 0,
          "end_at": 1,
          "embeds_scaling": "V only",
          "model": [
            "4",
            0
          ],
          "ipadapter": [
            "29",
            0
          ],
          "image": [
            "27",
            0
          ],
          "attn_mask": [
            "11",
            0
          ],
          "clip_vision": [
            "30",
            0
          ],
          "insightface": [
            "31",
            0
          ]
        },
        "class_type": "IPAdapterFaceID",
        "_meta": {
          "title": "IPAdapter FaceID"
        }
      },
      "29": {
        "inputs": {
          "ipadapter_file": "ip-adapter-faceid-plusv2_sdxl.bin"
        },
        "class_type": "IPAdapterModelLoader",
        "_meta": {
          "title": "IPAdapter Model Loader"
        }
      },
      "30": {
        "inputs": {
          "clip_name": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
        },
        "class_type": "CLIPVisionLoader",
        "_meta": {
          "title": "Load CLIP Vision"
        }
      },
      "31": {
        "inputs": {
          "provider": "CPU"
        },
        "class_type": "IPAdapterInsightFaceLoader",
        "_meta": {
          "title": "IPAdapter InsightFace Loader"
        }
      },
      "33": {
        "inputs": {
          "ipadapter_file": "ip-adapter-plus-face_sdxl_vit-h.safetensors"
        },
        "class_type": "IPAdapterModelLoader",
        "_meta": {
          "title": "IPAdapter Model Loader"
        }
      },
      "34": {
        "inputs": {
          "weight": 0.3,
          "weight_type": "linear",
          "combine_embeds": "concat",
          "start_at": 0,
          "end_at": 1,
          "embeds_scaling": "V only",
          "model": [
            "28",
            0
          ],
          "ipadapter": [
            "33",
            0
          ],
          "image": [
            "27",
            0
          ],
          "attn_mask": [
            "11",
            0
          ],
          "clip_vision": [
            "30",
            0
          ]
        },
        "class_type": "IPAdapterAdvanced",
        "_meta": {
          "title": "IPAdapter Advanced"
        }
      },
      "35": {
        "inputs": {
          "text": context,
          "clip": [
            "4",
            1
          ]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {
          "title": "CLIP Text Encode (Prompt)"
        }
      },
      "36": {
        "inputs": {
          "text": "(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgl, render, blender, digital art, manga, amateur:1.3), (3D, 3D Game, 3D Game Scene, 3D Character:1.1), acne",
          "clip": [
            "4",
            1
          ]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {
          "title": "CLIP Text Encode (Prompt)"
        }
      },
      "37": {
        "inputs": {
          "seed": 290334798307561,
          "steps": 35,
          "cfg": 8,
          "sampler_name": "dpmpp_2m",
          "scheduler": "karras",
          "denoise": 1,
          "model": [
            "54",
            0
          ],
          "positive": [
            "35",
            0
          ],
          "negative": [
            "36",
            0
          ],
          "latent_image": [
            "38",
            0
          ]
        },
        "class_type": "KSampler",
        "_meta": {
          "title": "KSampler"
        }
      },
      "38": {
        "inputs": {
          "width": 1024,
          "height": 1024,
          "batch_size": 1
        },
        "class_type": "EmptyLatentImage",
        "_meta": {
          "title": "Empty Latent Image"
        }
      },
      "39": {
        "inputs": {
          "samples": [
            "37",
            0
          ],
          "vae": [
            "4",
            2
          ]
        },
        "class_type": "VAEDecode",
        "_meta": {
          "title": "VAE Decode"
        }
      },
      "41": {
        "inputs": {
          "image": garmentImage, 
          "upload": "image"
        },
        "class_type": "LoadImage",
        "_meta": {
          "title": "Load Image"
        }
      },
      "42": {
        "inputs": {
          "resolution": 1024,
          "image": [
            "41",
            0
          ]
        },
        "class_type": "UniFormer-SemSegPreprocessor",
        "_meta": {
          "title": "UniFormer Segmentor"
        }
      },
      "43": {
        "inputs": {
          "channel": "red",
          "image": [
            "42",
            0
          ]
        },
        "class_type": "ImageToMask",
        "_meta": {
          "title": "Convert Image to Mask"
        }
      },
      "44": {
        "inputs": {
          "ipadapter_file": "ip-adapter-plus_sdxl_vit-h.safetensors"
        },
        "class_type": "IPAdapterModelLoader",
        "_meta": {
          "title": "IPAdapter Model Loader"
        }
      },
      "48": {
        "inputs": {
          "images": [
            "42",
            0
          ]
        },
        "class_type": "PreviewImage",
        "_meta": {
          "title": "Preview Image"
        }
      },
      "54": {
        "inputs": {
          "weight": 1,
          "weight_type": "linear",
          "combine_embeds": "concat",
          "start_at": 0,
          "end_at": 1,
          "embeds_scaling": "V only",
          "model": [
            "34",
            0
          ],
          "ipadapter": [
            "44",
            0
          ],
          "image": [
            "41",
            0
          ],
          "attn_mask": [
            "43",
            0
          ],
          "clip_vision": [
            "30",
            0
          ]
        },
        "class_type": "IPAdapterAdvanced",
        "_meta": {
          "title": "IPAdapter Advanced"
        }
      },
      "55": {
        "inputs": {
          "filename_prefix": "ComfyUI",
          "images": [
            "39",
            0
          ]
        },
        "class_type": "SaveImage",
        "_meta": {
          "title": "Save Image"
        }
      }
    }

    workflowJson =  json.dumps(comfyuiApiWorkflow, indent=2)
    output = replicate.run(
        "fofr/any-comfyui-workflow:6c42f2d9c7c0fcc873656b0156b554cde147118960a02c10c14e44312f9c8d7f",
        input={
            "output_format": "webp",
            "workflow_json": workflowJson,
            "output_quality": 80,
            "randomise_seeds": True,
            "return_temp_files": False
        }
    )
    return {
        "prompt": prompt,
        "context": context,
        "garmentImage": garmentImage,
        "genaiProductImage": output,
    }

def generateTryOnImage(modelImage:str, garmentImage:str):
    output = replicate.run(
        "viktorfa/oot_diffusion:9f8fa4956970dde99689af7488157a30aa152e23953526a605df1d77598343d7",
        input={
            "seed": 0,
            "steps": 20,
            "model_image": modelImage,
            "garment_image": garmentImage,
            "guidance_scale": 2
        }
    )

    return {
        "modelImage": modelImage,
        "garmentImage": garmentImage,
        "tryOnImage": output
    }