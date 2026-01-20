import gradio as gr
import logging

logging.basicConfig(filename="gradio_debug.log", level=logging.INFO, force=True)

def process(x):
    logging.info(f"Processing {x}")
    return f"Echo: {x}"

demo =gr.Interface(fn=process, inputs="text", outputs="text")
demo.launch(server_name="172.30.82.104", server_port=7860)

if __name__ == '__main__':
    process("manual test")