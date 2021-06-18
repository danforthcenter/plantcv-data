import os
import numpy as np
from plantcv import plantcv as pcv
import sys
sys.path.append("/Users/hudanyunsheng/Documents/github/plantcv/plantcv/plantcv/visualize/ecdf")
from ecdf import obj_size, pix_intensity

os.chdir(os.path.dirname(os.path.abspath(__file__)))

name_rgb = "input_color_img.jpg"
name_gray = "input_gray_img.jpg"

img_rgb, _,_ = pcv.readimage(name_rgb)
img_gray, _,_ = pcv.readimage(name_gray)

# create binary mask by threshoding and save result
mask = np.where(img_gray > 130,255,0).astype(np.uint8)
# pcv.print_image(mask, "mask.png")

# # ecdf for obj_size
# pcv.params.debug = None
# ecdf_obj_size =  obj_size(mask)
# pcv.print_image(ecdf_obj_size, "ecdf_obj_size.png")

# ecdf for pixel intensity: vis, no mask
ecdf_pix_int_rgb =  pix_intensity(img_rgb)
pcv.print_image(ecdf_pix_int_rgb, "ecdf_pix_int_rgb.png")

# ecdf for pixel intensity: vis, with mask
ecdf_pix_int_rgb_mask =  pix_intensity(img_rgb, mask=mask)
pcv.print_image(ecdf_pix_int_rgb_mask, "ecdf_pix_int_rgb_mask.png")

# ecdf for pixel intensity: grayscale, no mask
ecdf_pix_int_gray = pix_intensity(img_gray)
pcv.print_image(ecdf_pix_int_gray, "ecdf_pix_int_gray.png")

# ecdf for pixel intensity: grayscale, with mask
ecdf_pix_int_gray_mask =  pix_intensity(img_gray, mask=mask)
pcv.print_image(ecdf_pix_int_gray_mask, "ecdf_pix_int_gray_mask.png")