from transformers import DetrFeatureExtractor, DetrForObjectDetection


def object_detection(image):
    #url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    #image = Image.open(requests.get(url, stream=True).raw)

    feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')
    model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')

    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # model predicts bounding boxes and corresponding COCO classes
    logits = outputs.logits
    bboxes = outputs.pred_boxes
    return outputs

#out=object_detection( Image.open(requests.get('http://images.cocodataset.org/val2017/000000039769.jpg', stream=True).raw))
#print(out)