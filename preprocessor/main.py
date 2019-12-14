import ResizeVideos as RV
import CropVideos as CV

CATEGORIES = ['Fighting']

# Crop videos
CV.crop_videos(CATEGORIES[0],CV.read_annotation_file(CATEGORIES[0]))

# Resize videos
RV.resize_videos(CATEGORIES[0])