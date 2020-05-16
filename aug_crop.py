import os
from PIL import Image
### Training images
# LE7
# im_filepath = r"D:\\GitHub\BMES725\Q2\Train\LE7\Comparison-of-different-samples-for-2019-novel-cor_2020_International-Journa-p2-21%10.png"

# G7
# im_filepath = r"D:\GitHub\BMES725\Q2\Train\G7\2020.03.13.20035212-p23-157.png"
# im_filepath = r"D:\GitHub\BMES725\Q2\Train\G7\2020.02.10.20021584-p6-52%12.png"
# im_filepath = r"D:\GitHub\BMES725\Q2\Train\G7\2020.03.10.20032136-p34-118_2%0.png"
# im_filepath = r"D:\GitHub\BMES725\Q2\Train\G7\2020.03.16.20036145-p19-128-3.png"

### Validation images
# im_filepath = r"D:\GitHub\BMES725\Q2\Valid\LE7\396A81A5-982C-44E9-A57E-9B1DC34E2C08.jpeg"
# im_filepath = r"D:\GitHub\BMES725\Q2\Valid\LE7\3ED3C0E1-4FE0-4238-8112-DDFF9E20B471.jpeg"


# Determine output location
im_folder = os.path.dirname(im_filepath)
im_basename = os.path.basename(im_filepath)
file_parts = os.path.splitext(im_basename)
im_filename = "%s_CropAug.png" % (file_parts[0])
out_filename = os.path.join(im_folder, im_filename)

# Open
im = Image.open(im_filepath)
width, height = im.size

# Rescale
newsize = (512, 512)
im = im.resize(newsize)
width, height = im.size

# Setting the points for cropped image (Crop by 2% on all sides)
pct = 0.02
left = width * pct
right = width * (1-pct)
top = height * pct
bottom = height * (1-pct)
im = im.crop((left, top, right, bottom))
# im.show()
im.save(out_filename)
