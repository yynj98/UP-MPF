cuda=0
out_dir='out'
dataset='mvsa-s'             # Options: 't2015', 't2017', 'masad', 'mvsa-s', 'mvsa-m', 'tumemo'
train_file='train_few1.tsv'  # Options: 'train_few1.tsv', 'train_few2.tsv'
dev_file='dev_few1.tsv'      # Options: 'dev_few1.tsv', 'dev_few2.tsv'

case $dataset in
    'mvsa-s')
        img_dir='MVSA-S_data'
        early_stop=40
        ;;
    'mvsa-m')
        img_dir='MVSA-M_data'
        early_stop=40
        ;;
    'tumemo')
        img_dir='TumEmo_data'
        early_stop=20
        ;;
    't2015')
        img_dir='IJCAI2019_data/twitter2015_images'
        early_stop=40
        ;;
    't2017')
        img_dir='IJCAI2019_data/twitter2017_images'
        early_stop=40
        ;;
    'masad')
        img_dir='MASAD_imgs'
        early_stop=30
        ;;
esac

# template 1 settings
case $dataset in
    'mvsa-s' | 'mvsa-s' | 'tumemo')
        prompt_shape_pf='11'
        prompt_shape_mpf='111'
        ;;
    't2015' | 't2017' | 'masad')
        prompt_shape_pf='111'
        prompt_shape_mpf='1111'
        ;;
esac

# MPF
for pooling_scale in '11' '22' '33' '44' '55' '66' '77'
do
    for template in 1 2
    do
        for lr in 1e-5 2e-5 3e-5 4e-5 5e-5
        do
            for seed in 5 13 22 34 42
            do
                python main.py \
                    --cuda $cuda \
                    --out_dir $out_dir \
                    --img_dir $img_dir \
                    --dataset $dataset \
                    --train_file $train_file \
                    --dev_file $dev_file \
                    --template $template \
                    --prompt_shape $prompt_shape_mpf \
                    --pooling_scale $pooling_scale \
                    --batch_size 32 \
                    --lr_lm_model $lr \
                    --lr_resnet 0 \
                    --lr_visual_mlp $lr \
                    --early_stop $early_stop \
                    --seed $seed
            done
        done
    done
done
