import cv2
import sys

if __name__=='__main__':
    if len(sys.argv)<4:
        print("Usage:python reshape.py {input_file_path} {width} {height} {output_name}")
        exit()
    input_image=cv2.imread(sys.argv[1])
    cv2.imwrite(sys.argv[4],cv2.resize(input_image,(int(sys.argv[2]),int(sys.argv[3]))))
    print('done')