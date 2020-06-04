def read_img_list(path):
    with open(path, 'r') as f:
        result = list(f.readlines())
        img_str = result[0]
        img_str = img_str.split(' ')
        img_list = [int(item) for item in img_str]
        print("--load {} samples".format(len(img_list)))
        return img_list