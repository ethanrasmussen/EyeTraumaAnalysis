from main import *
import matplotlib.image as mpimg

# image = Image('data/ischemic/1_li.jpg')
# plt.imshow(np.vstack(get_segments(img=image.img,degInterval=10,widthPixels=20,center=image.center)))
# plt.savefig('test.jpg')
#
# mpimg.imsave('mp_test.jpg', np.vstack(get_segments(img=image.img,degInterval=10,widthPixels=20,center=image.center)))

for i in range(1,22):
    try:
        image = Image(f"data/01_raw/{i:0>5}_li.jpg")  # i:0>5 is used to pre-pad with 0s
        mpimg.imsave(f"data/02_concatenated/{i:0>5}_li_concatenated.jpg",
                     np.vstack(get_segments(img=image.img,
                                            interval_deg=10,
                                            wd_px=20,
                                            center=image.center))
                     )
        print(i)
    except:
        print(f"EXCEPTION: {i:0>5}")
