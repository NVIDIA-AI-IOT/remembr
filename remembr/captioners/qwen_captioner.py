import torch
from PIL import Image
from captioners.captioner import Captioner
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


class Qwen25VLCaptioner(Captioner):
    def __init__(self, args):
        self.args = args

        print(f"Loading Qwen2.5-VL from {args.model_path}...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

        self.processor = AutoProcessor.from_pretrained(args.model_path, use_fast=True)

        self.default_query = "You are a wandering around a university campus.\
        Please describe in detail what you see in the few seconds of the video. \
        Specifically focus on the people, objects, environmental features, events/activities, and other interesting details. Think step by step about these details and be very specific."

    def caption(self, images: list[Image.Image]):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": images,  # Pass the list of PIL Images directly
                        # Optional: You can control frame resolution/fps here if using qwen_vl_utils,
                        # but passing the PIL list directly to the processor is the standard transformer way.
                    },
                    {"type": "text", "text": self.default_query},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            videos=[images],  # Wrap in list because it expects a batch of videos
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.args.max_new_tokens,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text
