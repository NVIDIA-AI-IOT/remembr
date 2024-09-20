import rclpy
import numpy as np
import PIL.Image
import time
from threading import Thread
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from nano_llm import NanoLLM, ChatHistory
from dataclasses import dataclass




class CaptionerNode(Node):

    def __init__(self):
        super().__init__("CaptionerNode")

        self.declare_parameter("model", "Efficient-Large-Model/VILA1.5-3B")
        self.declare_parameter("image_topic", "/front_stereo_camera/left/image_raw")
        self.declare_parameter("caption_topic", "/caption")
        self.declare_parameter(
            "prompt",
            "<video> Please describe in detail what you see in the few seconds of " + \
            "the video. Specifically focus on the people, objects, environmental " + \
            "features, events/ectivities, and other interesting details. Think step " + \
            "by step about these details and be very specific."
        )
        self.declare_parameter("use_every_nth_image", 15)
        self.declare_parameter("caption_image_count", 6)
        self.declare_parameter("caption_interval", 3.0)

        self.image_subscriber = self.create_subscription(
            Image,
            self.get_parameter("image_topic").value,
            self.image_callback,
            10
        )

        self.caption_publisher = self.create_publisher(
            String, 
            self.get_parameter("caption_topic").value,
            10
        )

        self.debug = False
        self.cv_bridge = CvBridge()
        if not self.debug:
            self.model = NanoLLM.from_pretrained(self.get_parameter("model").value)
            self.chat_history = ChatHistory(self.model)

        self.prompt  = self.get_parameter("prompt").value.strip("][()")
        self.use_every_nth_image = self.get_parameter("use_every_nth_image").value
        self.caption_interval = self.get_parameter("caption_interval").value
        self.caption_topic = self.get_parameter("caption_topic").value
        self.caption_image_count = self.get_parameter("caption_image_count").value
        self.caption_interval = self.get_parameter("caption_interval").value

        # state
        self.image_buffer = []
        self.image_counter = 0

        self.caption_loop_thread = None
        self.caption_loop_running = False
        self.logger = self.get_logger()

    def start_caption_loop(self):
        self.caption_loop_running = True
        thread = Thread(target=self.caption_loop)
        thread.start()
        self.caption_loop_thread = thread

    def stop_caption_loop(self):
        self.caption_loop_running = False
        self.caption_loop_thread.join()
        self.caption_loop_thread = None

    def caption_loop(self):

        last_publish = time.perf_counter()

        while self.caption_loop_running:
            
            dt = time.perf_counter() - last_publish

            if dt < self.caption_interval:
                time.sleep(self.caption_interval - dt)

            # get last N images
            images = [b for b in self.image_buffer]
            
            if len(images) < self.caption_image_count:
                self.logger.info("Skipped image captioning for current time window.  No images available in buffer.")
            else:
                caption = self.caption_images(images)
                self.logger.info(f"Generated caption using {len(images)} images.")
                self.publish_caption(caption)
                self.logger.info(f"Published caption: " + caption)

            last_publish = time.perf_counter()
            
    def image_callback(self, image_msg: Image):
        
        if self.image_counter % self.use_every_nth_image == 0:
            self.logger.info("Received image.")
            np_image = self.cv_bridge.imgmsg_to_cv2(image_msg, 'rgb8')
            pil_image = PIL.Image.fromarray(np_image)
            timestamp = time.perf_counter()
            self.image_buffer.append(pil_image)
            if len(self.image_buffer) > self.caption_image_count:
                self.image_buffer = self.image_buffer[1:]
            
        self.image_counter += 1
    
    def caption_images(self, images):

        if self.debug:
            return "Dummy caption"

        # Add images to chat history
        for image in images:
            self.chat_history.append('user', image=image)

        # Add prompt to chat history
        self.chat_history.append('user', self.prompt, use_cache=True)

        # Embed chat history
        embedding, _ = self.chat_history.embed_chat()

        # Generate output
        caption = self.model.generate(
            inputs=embedding,
            kv_cache=self.chat_history.kv_cache,
            min_new_tokens = 50,
            streaming = False, 
            do_sample = True,
        )

        # Clean up
        self.chat_history.reset()

        return caption
    
    def publish_caption(self, caption: str):
        caption_msg = String()
        caption_msg.data = caption
        self.caption_publisher.publish(caption_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CaptionerNode()
    node.start_caption_loop()
    rclpy.spin(node)
    node.stop_caption_loop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()