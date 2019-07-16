import cv2


def dodge(image, mask):
    return cv2.divide(image, 255 - mask, scale=256)


def rgb_to_sketch(raw_img, img):
    img_rgb = cv2.imread(raw_img)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    img_gray_inv = 255 - img_gray
    img_blur = cv2.GaussianBlur(img_gray_inv, ksize=(21, 21),
                                sigmaX=0, sigmaY=0)
    img_blend = dodge(img_gray, img_blur)

    cv2.imwrite(img, img_blend)


if __name__ == '__main__':
    src_image_name = '../example/raw_img/demo1.png'
    dst_image_name = '../example/out_img/demo1.png'
    rgb_to_sketch(src_image_name, dst_image_name)
