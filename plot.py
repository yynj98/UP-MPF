import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from param import args

def plot4pretrain(epoch: int, d: dict):
    '''
    d = {
        'train_loss': [], 
        'test_loss': [], 
        'train_acc': [], 
        'test_acc': [], 
        'train_f1_mac': [], 
        'test_f1_mac': []
    }
    '''
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
    fig.suptitle(
        f'[bs: {args.batch_size}]-[lr_visual_mlp: {args.lr_visual_mlp}]', 
        fontdict={"weight": "bold"}, 
        fontsize=24
    )
    ax1.set_xlabel("epoch", fontdict={'size': 20})
    ax0.set_ylabel("Train", fontdict={'size': 20})
    ax0.plot(d['train_acc'], marker='o')
    ax0.plot(d['train_f1_mac'], marker='o')
    ax0.legend(['Acc', 'Mac-F1'], loc='lower right')
    ax0.grid(True)

    ax1.set_ylabel("Test", fontdict={'size': 20})
    ax1.plot(d['test_acc'], marker='o')

    x, y = epoch, d['test_acc'][epoch]
    ax1.text(x, y, f"({x}, {y:.4f})", ha='center', va='bottom', fontdict={"color": "red"}, fontsize=10)  # 默认就是10号字
    ax1.plot(d['test_f1_mac'], marker='o')
    
    x, y = epoch, d['test_f1_mac'][epoch]  # Mac-F1 corresponding to given Acc
    ax1.text(x, y, f"({x}, {y:.4f})", ha='center', va='top', fontdict={"color": "red"}, fontsize=10)
    
    x, y = d['test_f1_mac'].index(max(d['test_f1_mac'])), max(d['test_f1_mac'])  # max Mac-F1
    if x != epoch:
        ax1.text(x, y, f"({x}, {y:.4f})", ha='center', va='top', fontsize=10)

    ax1.legend(['Acc', 'Mac-F1'], loc='lower right')
    ax1.grid(True)

    return fig
