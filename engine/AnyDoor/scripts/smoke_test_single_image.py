import os
import sys
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_inference import inference_single_image

reference_image_path = './examples/TestDreamBooth/FG/01.png'
bg_image_path = './examples/TestDreamBooth/BG/000000309203_GT.png'
bg_mask_path = './examples/TestDreamBooth/BG/000000309203_mask.png'
save_path = './examples/TestDreamBooth/GEN/smoke_test_res.png'

image = cv2.imread(reference_image_path, cv2.IMREAD_UNCHANGED)
mask = (image[:, :, -1] > 128).astype('uint8')
image = image[:, :, :-1]
image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
ref_image = image
ref_mask = mask

back_image = cv2.imread(bg_image_path).astype('uint8')
back_image = cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB)

tar_mask = cv2.imread(bg_mask_path)[:, :, 0] > 128
tar_mask = tar_mask.astype('uint8')

gen_image = inference_single_image(ref_image, ref_mask, back_image.copy(), tar_mask)
h, w = back_image.shape[0], back_image.shape[1]
ref_image_resized = cv2.resize(ref_image, (w, h))
vis_image = cv2.hconcat([ref_image_resized, back_image, gen_image])

cv2.imwrite(save_path, vis_image[:, :, ::-1])
print('SAVED', save_path)
