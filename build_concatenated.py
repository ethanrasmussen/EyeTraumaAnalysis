from main import *
import matplotlib.image as mpimg

# image = Image('data/ischemic/1_li.jpg')
# plt.imshow(np.vstack(get_segments(img=image.img,degInterval=10,widthPixels=20,center=image.center)))
# plt.savefig('test.jpg')
#
# mpimg.imsave('mp_test.jpg', np.vstack(get_segments(img=image.img,degInterval=10,widthPixels=20,center=image.center)))

for i in range(1,22):
    try:
        image = Image(f'data/ischemic/{i}_li.jpg')
        mpimg.imsave(f'data/concatenated/{i}_li_concatenated.jpg', np.vstack(get_segments(img=image.img, degInterval=10, widthPixels=20, center=image.center)))
        print(i)
    except:
        print("EXCEPTION: "+str(i))