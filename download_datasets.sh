cd datasets

# VOC2007
mkdir VOC2007
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtrainval_06-Nov-2007.tar
mv VOCdevkit VOC2007/
rm VOCtrainval_06-Nov-2007.tar

# VOC2012
mkdir VOC2012
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar 
tar -xvf VOCtrainval_11-May-2012.tar
mv VOCdevkit VOC2012/
rm VOCtrainval_11-May-2012.tar

# COCO2014
wget https://raw.githubusercontent.com/YangtaoWANG95/TokenCut/master/datasets/coco_20k_filenames.txt

mkdir COCO
mkdir COCO/images

wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip COCO/images
rm train2014.zip

wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip COCO/images
rm val2014.zip

wget http://images.cocodataset.org/zips/test2014.zip
unzip test2014.zip COCO/images
rm test.2014.zip

wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip -d COCO
rm annotations_trainval2014.zip

cd ..
