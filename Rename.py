import os

video_path = './mediapipe_videos/videos/'
fullpath = os.path.join(os.getcwd(), video_path)
print('Current file path:', fullpath)

for root, dirs, files in os.walk(fullpath):
    for idx, f in enumerate(files):
        os.rename(fullpath + f, fullpath + 'shot_' + str(idx+1) + '.mp4')
        print('File:', f, 'completed.')