import numpy as np
import argparse
import time
import cv2

#argparse 
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help= "path to input image")
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to CAffe pre-trained model")
ap.add_argument("-l", "--labels", required=True, help="path to ImageNet labels")
args = vars(ap.parse_args())

#load input image from disk
image = cv2.imread(args["image"])

#load the class labels from disk
rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

#normalize input image
blob = cv2.dnn.blobFromImage(image, 1, (224,224), (104,117,123))

#load our sirialized model from disk
print ("[INFO] loading model....")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

#set the blob as input to the network and perform a forward-pass to obtain our output classifiation
net.setInput(blob)
start = time.time()
preds = net.forward()
end = time.time()
print("[INFO] classification took {:.5} seconds". format(end - start))

#sort the indexes of the probabilities in descending order (higher probability first) and grab the top5 prediction
idxs = np.argsort(preds[0])[::-1][:5]

#loop over top 5 prediction and display them
for(i,idx) in enumerate(idxs):
	#draw the top prediction on the input image
	if i == 0:
		text = "Label: {}, {:.2f}%".format(classes[idx],preds[0][idx] * 100)
		cv2.putText(image, text, (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	#display the predicted label + associated probability to the console
	print("[INFO] {}.label: {}, probability: {:.5}".format(i +1, classes[idx], preds[0][idx]))

#display the output image
cv2.imshow("Image", image)
cv2.waitKey(0)




