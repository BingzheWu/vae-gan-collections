import torch
import torchvision

from torchvision.datasets import CocoDetection


def make_dataset(opt, dataset_name, dataroot, annFile = None, imageSize = 224):
    if dataset_name == 'coco_obj_detect':
        trans = torchvision.transforms
        transform = trans.Compose([trans.Resize((imageSize, imageSize)), trans.ToTensor()])
        dataset = CocoDetection(dataroot, annFile, transform)
        data_iter = torch.utils.data.DataLoader(dataset, batch_size = opt.batchSize, shuffle = True, num_workers = 1)
    return data_iter
def test_coco_detect():
    dataset_name = 'coco_obj_detect'
    dataroot = '/datasets/coco/train2014/'
    annFile = '/datasets/coco/annotations/instances_train2014.json'
    dataset = make_dataset(dataset_name, dataroot, annFile )
    print(len(dataset))

if __name__ == '__main__':
    test_coco_detect()