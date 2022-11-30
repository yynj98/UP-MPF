import copy
import os
import torch
import time
import sklearn.metrics as metrics
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from param import args
from utils import processors
from plot import plot4pretrain
from model import MSAModel
from dataset import MSADataset
from dataset_pretrain import RotationImgDataset
from asyn_dataloader import CudaDataLoader, MultiEpochsDataLoader
from transformers.optimization import AdamW

if args.do_pretrain:
    print('[#] Pre-train')
else:
    if args.dataset in ['t2015', 't2017', 'masad']:
        print('[#] Fine grain')
    else:
        print('[#] Coarse grain')


class PreTrainer(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device

        # load datasets and create dataloaders
        self.train_set = RotationImgDataset(args, mode='train')
        self.train_loader = MultiEpochsDataLoader(self.train_set, collate_fn=self.train_set.collate_fn, 
                                                  batch_size=args.batch_size, pin_memory=True, num_workers=4, 
                                                  shuffle=True, drop_last=True)
        self.train_loader = CudaDataLoader(self.train_loader, device=self.device)
        
        self.test_set = RotationImgDataset(args, mode='test')
        self.test_loader = MultiEpochsDataLoader(self.test_set, collate_fn=self.test_set.collate_fn, 
                                                 batch_size=args.batch_size, pin_memory=True, num_workers=4)
        self.test_loader = CudaDataLoader(self.test_loader, device=self.device)

        # create model
        self.model = MSAModel(args, self.test_set.label_id_list)
        self.model.to(self.device)

        # save some performance metrics and plot later
        self.d = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': [], 'train_f1_mac': [], 'test_f1_mac': []}
        self.best_ckpt = None

        # get save dir
        pretrain_info = 'pretrain_rotation[{}][lp{}]'.format(self.args.visual_model_name, self.args.pooling_scale)
        self.save_dir = os.path.join(self.args.out_dir, pretrain_info, str(self.args.lr_visual_mlp))
        os.makedirs(self.save_dir, exist_ok=True)

    def update_best_ckpt(self, optimizer, epoch_idx, test_acc, test_f1_mac, test_f1_wtd):
        if self.args.lr_lm_model == 0:  # only train the visual encoder in UP stage
            model_params = copy.deepcopy(self.model.visual_encoder.state_dict())
        else:
            model_params = copy.deepcopy(self.model.state_dict())

        optimizer_params = copy.deepcopy(optimizer.state_dict())

        ckpt_name = time.strftime("%y%m%d_%H:%M:%S", time.localtime())
        ckpt_name += "[Ep{:02}][Test{:.2f}-{:.2f}-{:.2f}].ckpt".format(epoch_idx, test_acc * 100, test_f1_mac * 100, test_f1_wtd * 100)

        self.best_ckpt = {
            'ckpt_name': ckpt_name,
            'embedding': model_params,
            'epoch': epoch_idx,
            'optimizer': optimizer_params
        }
    
    def save(self):
        ckpt_name = self.best_ckpt['ckpt_name']
        torch.save(self.best_ckpt, os.path.join(self.save_dir, ckpt_name))
        fig = plot4pretrain(self.best_ckpt['epoch'], self.d)
        fig.savefig(os.path.join(self.save_dir, ckpt_name) + ".png")
        print("[#] Checkpoint {} saved.".format(ckpt_name))
    
    def _evaluate(self):
        self.model.eval()
        with torch.no_grad():
            self.model.eval()
            loss = []
            y_ = []
            y = []
            
            pbar = tqdm(self.test_loader, unit="batch", desc='*Test pbar')
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                imgs = batch['imgs'].to(self.device) if not self.args.no_img else None
                
                _loss, _y_, _y = self.model(input_ids, attention_mask, labels, imgs)
                loss.append(_loss.item())
                y_.extend(_y_)
                y.extend(_y)

            loss = sum(loss) / len(self.test_loader)
            acc = metrics.accuracy_score(y, y_)
            f1_mac = metrics.f1_score(y, y_, average='macro')
            f1_wtd = metrics.f1_score(y, y_, average='weighted')
            print(f"[ Test] Loss: {loss:.4f} Acc: {acc:.4f}, Macro F1: {f1_mac:.4f}, Weighted F1: {f1_wtd:.4f}")
        return loss, acc, f1_mac, f1_wtd, y, y_
    
    def train(self):
        if self.args.lr_lm_model == 0:  # only train the visual encoder in UP stage
            for p in self.model.lm_model.parameters():
                p.requires_grad = False
        
        def get_params_for_decay(named_params: list, lr: float):
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            returns = [
                {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay, 'lr': lr},
                {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': lr}
            ]
            return returns
        
        params = []
        params.extend(get_params_for_decay(self.model.lm_model.named_parameters(), lr=self.args.lr_lm_model))
        params.extend(get_params_for_decay(self.model.visual_encoder.backbone.named_parameters(), self.args.lr_resnet))
        params.extend(get_params_for_decay(self.model.visual_encoder.visual_mlp.named_parameters(), self.args.lr_visual_mlp))

        optimizer = AdamW(params=params, weight_decay=self.args.weight_decay)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.args.decay_rate)

        best_test_acc, best_test_epoch = 0, 0
        for epoch_idx in range(self.args.pretrain_epoch):
            print(f'\n[#] Epoch {epoch_idx}')
            train_loss = []
            y_ = []
            y = []

            pbar = tqdm(self.train_loader, unit="batch")
            for batch in pbar:
                self.model.train()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                imgs = batch['imgs'].to(self.device) if not self.args.no_img else None
            
                _loss, _y_, _y = self.model(input_ids, attention_mask, labels, imgs)
                train_loss.append(_loss.item())
                y_.extend(_y_)
                y.extend(_y)
                pbar.set_description(f'*Train batch loss: {_loss.item():.4f}')

                _loss.backward()
                # torch.cuda.empty_cache()
                optimizer.step()
                # torch.cuda.empty_cache()
                optimizer.zero_grad()

                self.train_set.shuffle()
            my_lr_scheduler.step()
            train_loss = sum(train_loss) / len(self.train_loader)
            train_acc = metrics.accuracy_score(y, y_)
            train_f1_mac = metrics.f1_score(y, y_, average='macro')
            print(f"[Train] Loss: {train_loss:.4f} Acc: {train_acc:.4f}")

            test_loss, test_acc, test_f1_mac, test_f1_wtd, _, _ = self._evaluate()

            self.d['train_loss'].append(train_loss)
            self.d['test_loss'].append(test_loss)
            self.d['train_acc'].append(train_acc)
            self.d['test_acc'].append(test_acc)
            self.d['train_f1_mac'].append(train_f1_mac)
            self.d['test_f1_mac'].append(test_f1_mac)

            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                best_test_epoch = epoch_idx
                print(f'[#] Best test acc: {test_acc:.4f}')
            self.update_best_ckpt(optimizer, epoch_idx, test_acc, test_f1_mac, test_f1_wtd)
            self.save()
            
        print('[*] Ending Training')
        print(f'[#] Best test Acc: {best_test_acc:.4f} at epoch {best_test_epoch}')


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.img_template_mode = args.img_template_mode
        
        # load datasets and create dataloaders
        self.train_set = MSADataset(args, processors[args.dataset], tsv_file=args.train_file, max_seq_length=64 if args.dataset=='masad' else 128)
        self.train_loader = DataLoader(self.train_set, collate_fn=self.train_set.collate_fn, batch_size=args.batch_size, shuffle=True)
        self.dev_set = MSADataset(args, processors[args.dataset], tsv_file=args.dev_file, max_seq_length=64 if args.dataset=='masad' else 128)
        self.dev_loader = DataLoader(self.dev_set, collate_fn=self.dev_set.collate_fn, batch_size=args.batch_size * 4)
        self.test_set = MSADataset(args, processors[args.dataset], tsv_file='test.tsv', max_seq_length=64 if args.dataset=='masad' else 128)
        self.test_loader = DataLoader(self.test_set, collate_fn=self.test_set.collate_fn, batch_size=args.batch_size * 4)
        
        # create model
        self.model = MSAModel(args, self.test_set.label_id_list)
        self.model.to(self.device)

        # create ckpt
        self.best_ckpt = {
            'test_size': len(self.test_set),
            'args': self.args
        }

        self.save_dir = self.get_save_dir()
        os.makedirs(self.save_dir, exist_ok=True)

    def get_save_dir(self):
        template_info = '{}{}[t{}]'.format(self.args.train_abbr, self.args.dev_abbr, self.args.template)  # [s1][d1][t1]
        if self.img_template_mode != 'default':
            template_info += '[{}]'.format(self.img_template_mode)
        if self.args.template == 1:
            template_info += '[ps{}]'.format(self.args.prompt_shape)  # [ps111]
        if not self.args.no_img:
            template_info += "[{}][lp{}]".format(self.args.visual_model_name, self.args.pooling_scale)  # [resnet50-1][lp11]

        dataset_info = self.args.dataset
        
        # add another dir when load UP model
        if self.args.up_model_path:
            dataset_info = '{}+'.format(self.args.dataset)
            up_model_path = os.path.splitext(self.args.up_model_path)[0]
            up_model_path = os.path.split(up_model_path)[1]
            return os.path.join(self.args.out_dir, dataset_info, template_info, up_model_path, str(self.args.lr_lm_model))

        return os.path.join(self.args.out_dir, dataset_info, template_info, str(self.args.lr_lm_model))

    def save(self):
        '''
        save predictions but do not save model parameters
        '''
        ckpt_name = self.best_ckpt['ckpt_name']
        test_y = self.best_ckpt['test_y']
        test_y_ = self.best_ckpt['test_y_']
        with open(os.path.join(self.save_dir, ckpt_name) + '.txt', 'w', encoding='utf-8') as f:
            f.write('#True\t#Pred\n')
            for y, y_ in zip(test_y, test_y_):
                token_y = self.test_set.tokenizer.convert_ids_to_tokens(y)
                token_y_ = self.test_set.tokenizer.convert_ids_to_tokens(y_)
                f.write(f'{token_y:10}\t{token_y_:10}\n')
        print("[#] Checkpoint {} saved.".format(ckpt_name))
        return

    def load_up_model(self):
        '''
        load UP model
        '''
        print('[#] Loading {}'.format(self.args.up_model_path))
        ckpt_dict = torch.load(self.args.up_model_path, map_location='cpu')
        if self.args.load_visual_encoder:
            self.model.visual_encoder.load_state_dict(ckpt_dict["embedding"])
        else:            
            self.model.load_state_dict(ckpt_dict["embedding"], strict=False)

    def _evaluate(self, evaluate_type):
        self.model.eval()
        if evaluate_type == 'Test':
            loader = self.test_loader
        else:
            loader = self.dev_loader
        with torch.no_grad():
            loss = []
            y_ = []
            y = []
            
            pbar = tqdm(loader, unit="batch", desc=f'*{evaluate_type} pbar')
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                imgs = batch['imgs'].to(self.device) if not self.args.no_img else None
                
                _loss, _y_, _y = self.model(input_ids, attention_mask, labels, imgs)
                loss.append(_loss.item())
                y_.extend(_y_)
                y.extend(_y)

            loss = sum(loss) / len(loader)
            acc = metrics.accuracy_score(y, y_)
            f1_mac = metrics.f1_score(y, y_, average='macro')
            f1_wtd = metrics.f1_score(y, y_, average='weighted')
            print(f"[{evaluate_type:5}] Loss: {loss:.4f} Acc: {acc:.4f}, Macro F1: {f1_mac:.4f}, Weighted F1: {f1_wtd:.4f}")
        return loss, acc, f1_mac, f1_wtd, y, y_

    def update_best_ckpt(self, epoch_idx=None, dev_acc=None, test_acc=None, test_f1_mac=None, test_f1_wtd=None, test_y=None, test_y_=None):
        if test_acc == None:  # during training
            model_params = copy.deepcopy(self.model.state_dict())
            self.best_ckpt['time'] = datetime.now()
            self.best_ckpt['embedding'] = model_params
            self.best_ckpt['epoch'] = epoch_idx
            self.best_ckpt['dev_acc'] = dev_acc
        else:  # after testing
            ckpt_name = time.strftime("%y%m%d_%H:%M:%S", time.localtime())
            epoch = epoch_idx if epoch_idx else self.best_ckpt['epoch']
            ckpt_name += "[Ep{:02}][Test{:.2f}-{:.2f}-{:.2f}].ckpt".format(epoch, test_acc * 100, test_f1_mac * 100, test_f1_wtd * 100)
            self.best_ckpt['ckpt_name'] = ckpt_name
            self.best_ckpt['test_acc'] = test_acc
            self.best_ckpt['test_f1_mac'] = test_f1_mac
            self.best_ckpt['test_f1_wtd'] = test_f1_wtd
            self.best_ckpt['test_y'] = test_y
            self.best_ckpt['test_y_'] = test_y_
        

    def train(self):
        if not self.args.no_img and self.args.lr_resnet == 0:  # do not train ResNet in Downstream MPF stage
            for p in self.model.visual_encoder.backbone.parameters():
                p.requires_grad = False
        
        def get_params_for_decay(named_params: list, lr: float):
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            returns = [
                {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay, 'lr': lr},
                {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': lr}
            ]
            return returns

        params = get_params_for_decay(self.model.lm_model.named_parameters(), lr=self.args.lr_lm_model)
        if self.args.template == 1:
            params.extend(get_params_for_decay(self.model.prompt_encoder.named_parameters(), self.args.lr_lm_model))
        if not self.args.no_img:
            params.extend(get_params_for_decay(self.model.visual_encoder.backbone.named_parameters(), self.args.lr_resnet))
            params.extend(get_params_for_decay(self.model.visual_encoder.visual_mlp.named_parameters(), self.args.lr_visual_mlp))
        
        optimizer = AdamW(params=params, weight_decay=self.args.weight_decay)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.args.decay_rate)

        best_dev_acc, early_stop = 0, 0
        for epoch_idx in range(100):
            print(f'\n[#] Epoch {epoch_idx}')
            loss = []
            y_ = []
            y = []

            self.model.train()
            pbar = tqdm(self.train_loader, unit="batch")
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                imgs = batch['imgs'].to(self.device) if not self.args.no_img else None


                _loss, _y_, _y = self.model(input_ids, attention_mask, labels, imgs)
                loss.append(_loss.item())
                y_.extend(_y_)
                y.extend(_y)
                pbar.set_description(f'*Train batch loss: {_loss.item():.4f}')

                optimizer.zero_grad()
                _loss.backward()
                # torch.cuda.empty_cache()
                optimizer.step()
                # torch.cuda.empty_cache()
            my_lr_scheduler.step()
            loss = sum(loss) / len(self.train_loader)
            acc = metrics.accuracy_score(y, y_)
            print(f"[Train] Loss: {loss:.4f} Acc: {acc:.4f}")

            dev_loss, dev_acc, dev_f1, _, _, _ = self._evaluate('Dev')
            if dev_acc >= best_dev_acc:
                print(f'[#] Best dev acc: {dev_acc:.4f}')
                self.update_best_ckpt(epoch_idx, dev_acc)
                early_stop = 0
                best_dev_acc = dev_acc
            else:
                early_stop += 1
                if early_stop >= self.args.early_stop:
                    print("[*] Early stopping at epoch {}.".format(epoch_idx))
                    return
        print('[*] Ending Training')

    def evaluate_on_test(self):
        print('[#] Begin to evaluate on test set')
        self.model.load_state_dict(self.best_ckpt["embedding"])
        _, test_acc, test_f1_mac, test_f1_wtd, test_y, test_y_ = self._evaluate('Test')
        self.update_best_ckpt(None, None, test_acc, test_f1_mac, test_f1_wtd, test_y, test_y_)
        self.save()


def main():
    if args.do_pretrain:
        pre_trainer = PreTrainer(args)
        pre_trainer.train()
    else:
        trainer = Trainer(args)
        if args.up_model_path:
            trainer.load_up_model()
        trainer.train()
        trainer.evaluate_on_test()
        

if __name__ == '__main__':
    main()
