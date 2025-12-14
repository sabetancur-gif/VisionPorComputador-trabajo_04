from moviepy.editor import VideoFileClip

input_video = "results/cis_val_run/tracking_result.mp4"
output_video = "results/cis_val_run/tracking_web.mp4"

clip = VideoFileClip(input_video)
clip.write_videofile(
    output_video,
    codec="libx264",
    audio=False
)