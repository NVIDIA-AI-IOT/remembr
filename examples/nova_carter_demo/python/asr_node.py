import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rcl_interfaces.msg import ParameterDescriptor
from asr_pipeline import ASRPipeline


class AsrNode(Node):
    def __init__(self):
        super().__init__('AsrNode')

        self.declare_parameter("model", "small.en")
        self.declare_parameter("backend", "whisper_trt") 

        # TODO: remove placeholder default
        self.declare_parameter("cache_dir", "data")#rclpy.Parameter.Type.STRING)
        self.declare_parameter("vad_window", 5)

        self.declare_parameter("mic_device_index", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("mic_sample_rate", 16000)
        self.declare_parameter("mic_channels", 6)
        self.declare_parameter("mic_bitwidth", 2)
        self.declare_parameter("mic_channel_for_asr", 0)

        self.declare_parameter("speech_topic", "/speech")

        self.speech_publisher = self.create_publisher(
            String, 
            self.get_parameter("speech_topic").value, 
            10
        )

        logger = self.get_logger()

        def handle_vad_start():
            logger.info("vad start")

        def handle_vad_end():
            logger.info("vad end")

        def handle_asr(text):
            msg = String()
            msg.data = text
            self.speech_publisher.publish(msg)
            logger.info("published " + text)

        self.pipeline = ASRPipeline(
            model=self.get_parameter("model").value,
            vad_window=self.get_parameter("vad_window").value,
            backend=self.get_parameter("backend").value,
            cache_dir=self.get_parameter_or("cache_dir", None).value,
            vad_start_callback=handle_vad_start,
            vad_end_callback=handle_vad_end,
            asr_callback=handle_asr,
            mic_device_index=self.get_parameter_or("mic_device_index", None).value,
            mic_sample_rate=self.get_parameter("mic_sample_rate").value,
            mic_channel_for_asr=self.get_parameter("mic_channel_for_asr").value,
            mic_num_channels=self.get_parameter("mic_channels").value,
            mic_bitwidth=self.get_parameter("mic_bitwidth").value
        )

    def start_asr_pipeline(self):
        self.pipeline.start()


def main(args=None):
    rclpy.init(args=args)
    node = AsrNode()

    node.start_asr_pipeline()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()