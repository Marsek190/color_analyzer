from PIL import Image
from io import BytesIO
from operator import itemgetter
from collections import Counter
import time

MAX_SIZE = 500
PATH = '/home/jonik/8565fbe3607f23edd2f46703651a5d2d.jpg'

def resize_image(path=PATH):    
    image = Image.open(path)
    w, h = image.size  # Определяем ширину и высоту
    MAX_SIDE = max(w, h)
    if MAX_SIDE > MAX_SIZE:
        scale = MAX_SIZE / MAX_SIDE
        new_image = image.resize((int(w * scale), int(h * scale)))
        with BytesIO() as fake_file:
            try:
                new_image.save(fake_file, 'JPEG')        
            except:
                new_image.save(fake_file, path.split('.')[-1])
            return process(Image.open(fake_file))
    return process(image)

def process(image):
    begin = time.time()
    width, height = image.size  # Определяем ширину и высоту
    img_resolution = width * height
    pix = image.load() # Выгружаем значения пикселей
    rgb_by_rate = Counter()
    INC = 1 / img_resolution
    for x in range(width):
        for y in range(height):
            rgb = '.'.join(list(map(str, pix[x, y])))           
            rgb_by_rate[rgb] += INC
    end = time.time()
    rgb_by_rate_sorted = sorted(
            rgb_by_rate.items(), 
            reverse=True, 
            key=itemgetter(1))
    print(rgb_by_rate_sorted[0:3])    
    print(sum(rgb_by_rate.values()))
    print(f'seconds: {end - begin}')

def main():
    resize_image()
    
if __name__ == '__main__':
    main()