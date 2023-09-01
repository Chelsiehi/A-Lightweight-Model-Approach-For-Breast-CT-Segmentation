from PIL import Image
import os


input_dir = 'Dataset_BUSI_with_GT/normal'
output_dir = 'Normal_Dataset/normal/'


for filename in os.listdir(input_dir):
   
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
       
        with Image.open(os.path.join(input_dir, filename)) as img:
            
            img = img.resize((256, 256))
            img.save(os.path.join(output_dir, filename))