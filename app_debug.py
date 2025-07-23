# app_debug.py
import gradio as gr
import shutil

def debug_video(video_path):
    """
    This function does no processing. It just copies the input video
    to a new location to test Gradio's file handling.
    """
    print(f"Received video at: {video_path}")
    output_path = "debug_output.mp4"
    shutil.copy(video_path, output_path)
    print(f"Copied video to: {output_path}")
    return output_path

iface = gr.Interface(
    fn=debug_video,
    inputs=gr.Video(label="Upload Any Video"),
    outputs=gr.Video(label="Output Video"),
    title="Gradio Video Handling Test"
)

if __name__ == "__main__":
    iface.launch()