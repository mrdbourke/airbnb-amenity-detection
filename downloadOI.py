#Author : Sunita Nayak, Big Vision LLC

#### Usage example: python3 downloadOI.py --classes 'Ice_cream,Cookie' --dataset train

import argparse
import csv
import subprocess
import os
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool

cpu_count = multiprocessing.cpu_count()
print(f"CPU count: {cpu_count}")

parser = argparse.ArgumentParser(description="Download Class specific images from OpenImagesV5")
parser.add_argument("--dataset", help="Dataset category - train, validation or test", required=True)
parser.add_argument("--classes", help="Names of object classes to be downloaded", required=True)
parser.add_argument("--nthreads", help="Number of threads to use", required=False, type=int, default=cpu_count)
parser.add_argument("--occluded", help="Include occluded images", required=False, type=int, default=1)
parser.add_argument("--truncated", help="Include truncated images", required=False, type=int, default=1)
parser.add_argument("--groupOf", help="Include groupOf images", required=False, type=int, default=1)
parser.add_argument("--depiction", help="Include depiction images", required=False, type=int, default=1)
parser.add_argument("--inside", help="Include inside images", required=False, type=int, default=1)

args = parser.parse_args()

dataset = args.dataset

threads = args.nthreads 

classes = []
for class_name in args.classes.split(','):
    classes.append(class_name)

# Open class descriptions csv and save make ClassID:ClassName dictionary
with open("./class-descriptions-boxable.csv", mode="r") as infile:
    reader = csv.reader(infile)
    dict_list = {rows[1]:rows[0] for rows in reader}

# Remove directories for target dataset (e.g. "validation" and "train" if they exist)
subprocess.run(["rm", "-rf", dataset])

# Make directory for target images (directory will be named after target dataset, e.g. "validation")
subprocess.run(["mkdir", dataset])

pool = Pool(threads)
commands = []
cnt = 0

for ind in range(0, len(classes)):
    
    class_name = classes[ind]
    print("Downloading class " + str(ind) + ": " + class_name)
    
    # Match ClassID to annotations information
    command = "grep "+ dict_list[class_name.replace("_", " ")] + " ./" + dataset + "-annotations-bbox.csv"
    print(command)
    # Image annotations from annotations CSV
    image_annotations = subprocess.run(command.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')
    #print(f"Class annotations 1: {image_annotations}")
    image_annotations = image_annotations.splitlines()
    #print(f"Class annotations 2: {image_annotations}")
    
    for line in image_annotations:
        line_parts = line.split(',')
         
        # IsOccluded, IsTruncated, IsGroupOf, IsDepiction, IsInside (see Open Images download page)
        if (args.occluded==0 and int(line_parts[8])>0):
            print("Skipped %s", line_parts[0])
            continue
        if (args.truncated==0 and int(line_parts[9])>0):
            print("Skipped %s", line_parts[0])
            continue
        if (args.groupOf==0 and int(line_parts[10])>0):
            print("Skipped %s", line_parts[0])
            continue
        if (args.depiction==0 and int(line_parts[11])>0):
            print("Skipped %s", line_parts[0])
            continue
        if (args.inside==0 and int(line_parts[12])>0):
            print("Skipped %s", line_parts[0])
            continue

        cnt = cnt + 1
        
        # Download image from S3 and save it to dataset folder such as "validation/0b6227bb06345402.jpg"
        command = 'aws s3 --no-sign-request --only-show-errors cp s3://open-images-dataset/'+dataset+'/'+line_parts[0]+'.jpg '+ dataset+'/'+line_parts[0]+'.jpg'
        #print(f"Command 2: {command}")
        commands.append(command)
        
#         with open('%s/%s/%s.txt'%(dataset,class_name,line_parts[0]),'a') as f:
#             f.write(','.join([class_name, line_parts[4], line_parts[5], line_parts[6], line_parts[7]])+'\n')

#print("Annotation Count : " + str(cnt))
commands = list(set(commands))
print(f"Downloading: {str(len(commands))} images | Num classes: {len(classes)} | Dataset: {dataset}")

if __name__ == "__main__":
    list(tqdm(pool.imap(os.system, commands), total = len(commands)))
    pool.close()
    pool.join()
