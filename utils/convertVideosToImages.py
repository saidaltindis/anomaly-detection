import os

with os.scandir('Fighting/') as videos:
    for video in videos:
        video_name = video.name
        folder_name = video_name.split(".")[0]
        os.mkdir("Images/"+folder_name)
        commandToExecute = 'ffmpeg -i Fighting/' + (video_name) + ' -r 24 Images/' + (folder_name) +'/$image%06d.jpg'
        #print(commandToExecute)
        os.system(commandToExecute)
	
