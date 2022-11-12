from main import *
import matplotlib.image as mpimg

# image = Image('data/ischemic/1_li.jpg')
# plt.imshow(np.vstack(get_segments(img=image.img,degInterval=10,widthPixels=20,center=image.center)))
# plt.savefig('test.jpg')
#
# mpimg.imsave('mp_test.jpg', np.vstack(get_segments(img=image.img,degInterval=10,widthPixels=20,center=image.center)))
"""
for i in range(1,22):
    try:
        image = Image(f"data/01_raw/{i:0>5}_li.jpg")  # i:0>5 is used to pre-pad with 0s
        segments = get_segments(img=image.img,
                                            interval_deg=10,
                                            wd_px=20,
                                            center=image.center)
        unwrapped_img = np.vstack([segment for ind, segment in segments.items()])
        # Can save alpha by saving to png. However, that will greatly increase file size (e.g. 16kb -> 160kb)
        mpimg.imsave(f"data/02_concatenated/{i:0>5}_li_concatenated.jpg", unwrapped_img )
        print(f"data/01_raw/{i:0>5}.jpg")
    except:
        print(f"EXCEPTION: " + f"data/01_raw/{i:0>5}.jpg")
"""
for i in range(11000,11011+1):
    try:
        image = Image(f"data/01_raw/{i:0>5}.jpg")  # i:0>5 is used to pre-pad with 0s
        segments = get_segments(img=image.img,
                                            interval_deg=10,
                                            wd_px=20,
                                            center=image.center, borderValue=(0,255,0,0))
        unwrapped_img = np.vstack([segment for ind, segment in segments.items()])
        # Can save alpha by saving to png. However, that will greatly increase file size (e.g. 16kb -> 160kb)
        mpimg.imsave(f"data/02_concatenated/{i:0>5}_concatenated.png", unwrapped_img )
        print(f"data/01_raw/{i:0>5}.jpg")
    except:
        print(f"EXCEPTION: " + f"data/01_raw/{i:0>5}.jpg")