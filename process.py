from PIL import Image, ImageOps    
import pillow_avif
import csv

def resizeImage(fileName):
    with Image.open("./data/"+fileName) as img:
        ImageOps.pad(img, size, color="#000000").save("./data/modified_"+fileName)
        
if __name__ == "__main__":
    size=(256,256)
    with open("./data/carbonCredits.csv", "r") as file:
        reader = csv.reader(file, delimiter=',')
        for i,row in enumerate(reader):
            if i != 0:
                resizeImage(row[0])
                resizeImage(row[1])
                