cuda=0
out_dir='out'
dataset='coco2014'
img_dir='coco_trainval2014'
lr=1e-5

python main.py \
    --cuda $cuda \
    --out_dir $out_dir \
    --dataset $dataset \
    --img_dir $img_dir \
    --do_pretrain \
    --pooling_scale '77' \
    --batch_size 64 \
    --lr_resnet $lr \
    --lr_visual_mlp $lr \
    --pretrain_epoch 10 \
    --seed 34
