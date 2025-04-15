import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import GRIDDataset
import torch.optim as optim
from model import CTCLipModel, DRLModel
from jiwer import cer, wer
import numpy as np
from ctc_decode import ctc_beam_decode
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from constants import *

DEVICE = torch.device('cuda:1')


def train(train_set, val_set=None, lr=1e-4, epochs=100, batch_size=50, model_path=None):
    model = CTCLipModel(len(train_set.vocab)).to(DEVICE)
    print(sum(param.numel() for param in model.parameters()))
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location='cpu')
        states = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(states)
        print('loading weights ...')
    model.train()
    print(model)
    print('train VSR model ...')
    data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=5)
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)
    # lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_iters//10, num_training_steps=num_iters)
    best_wer, best_cer = 1., 1.
    for ep in range(1, 1 + epochs):
        ep_loss = 0.
        for i, batch_data in enumerate(data_loader):  # (B, T, C, H, W)
            inputs = batch_data['vid'].to(DEVICE)
            targets = batch_data['txt'].to(DEVICE)
            input_lens = batch_data['vid_lens'].to(DEVICE)
            target_lens = batch_data['txt_lens'].to(DEVICE)
            optimizer.zero_grad()
            logits = model(inputs, input_lens)[0]
            logits = logits.transpose(0, 1).log_softmax(dim=-1)  # (T, B, V)
            loss = F.ctc_loss(logits, targets, input_lens.reshape(-1), target_lens.reshape(-1), zero_infinity=True)
            loss.backward()
            optimizer.step()
            ep_loss += loss.data.item()
            # lr_scheduler.step()
            # if (i + 1) % 5 == 0:
            print("Epoch {}, Iteration {}, loss: {:.4f}".format(ep, i + 1, loss.data.item()), flush=True)

        if ep % 1 == 0:
            print("Epoch {}, loss: {:.4f}".format(ep, ep_loss), flush=True)
            savename = 'iter_{}.pt'.format(ep)
            savedir = os.path.join('checkpoints', 'vsr_grid2')
            if not os.path.exists(savedir): os.makedirs(savedir)
            save_path = os.path.join(savedir, savename)
            torch.save({'model': model.state_dict()}, save_path)
            print(f'Saved to {save_path}!!!', flush=True)
            if val_set is not None:
                wer, cer = evaluate(save_path, val_set, batch_size=50)
                print(f'Val WER: {wer}, CER: {cer}', flush=True)
                if wer < best_wer:
                    best_wer, best_cer = wer, cer
                print(f'Best WER: {best_wer}, CER: {best_cer}', flush=True)



# DRL training for VSR and SV
def drl_train(vsr_set, spk_set, drl_set, val_set=None, lr=1e-4, epochs=100, batch_size=32, model_path=None):
    model = DRLModel(len(vsr_set.vocab), len(vsr_set.spks)).to(DEVICE)
    print(sum(param.numel() for param in model.parameters()))
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location='cpu')
        states = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(states, strict=False)
        print('loading weights ...')
    model.train()
    print(model)
    spk_data_loader = DataLoader(spk_set, batch_size=2, shuffle=True, num_workers=2)
    vsr_data_loader = DataLoader(vsr_set, batch_size=batch_size, shuffle=True, num_workers=5)
    drl_data_loader = DataLoader(drl_set, batch_size=2, shuffle=True, num_workers=2)
    #drl_data_loader = DataLoader(drl_set, batch_size=batch_size // 2, shuffle=True, num_workers=6)
    #optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)
    #spk_optimizer = optim.AdamW(model.spk.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)
    mi_optimizer = optim.AdamW(model.mi_net.parameters(), lr=3*lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)
    #vsr_optimizer = optim.AdamW(model.parameters(), lr=3*lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)
    vsr_optimizer = optim.AdamW([*model.vsr.parameters(), *model.spk.parameters()], lr=3*lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)
    num_iters = len(vsr_data_loader) * epochs
    lr_scheduler = get_cosine_schedule_with_warmup(vsr_optimizer, num_warmup_steps=num_iters//20, num_training_steps=num_iters)
    best_wer, best_cer = 1., 1.

    '''
    for ep in range(300):
        for i, batch_data in enumerate(spk_data_loader):  # (2, N, T, C, H, W)
            inputs = batch_data['vid'].to(DEVICE)
            model.zero_grad()
            loss = model.calc_triplet_loss(inputs)
            loss.backward()
            spk_optimizer.step()
            # lr_scheduler.step()
            print("Epoch {}, Iteration {}, sv loss: {:.4f}".format(ep, i + 1, loss.data.item()), flush=True)
    savedir = os.path.join('checkpoints', 'drl_grid2')
    if not os.path.exists(savedir): os.makedirs(savedir)
    save_path = os.path.join(savedir, 'spk.pt')
    torch.save({'model': model.state_dict()}, save_path)
    print(f'Saved to {save_path}!!!', flush=True)
    '''

    '''
    for ep in range(10):
        ep_loss = 0.
        for i, batch_data in enumerate(vsr_data_loader):  # (B, T, C, H, W)
            inputs = batch_data['vid'].to(DEVICE)
            targets = batch_data['txt'].to(DEVICE)
            input_lens = batch_data['vid_lens'].to(DEVICE)
            target_lens = batch_data['txt_lens'].to(DEVICE)
            model.zero_grad()
            loss = model(inputs, targets, input_lens, target_lens)
            loss.backward()
            optimizer.step()
            ep_loss += loss.data.item()
            # lr_scheduler.step()
            print("Epoch {}, Iteration {}, vsr loss: {:.4f}".format(ep, i + 1, loss.data.item()), flush=True)
        if ep % 1 == 0:
            print("Epoch {}, loss: {:.4f}".format(ep, ep_loss), flush=True)
            savename = 'iter_{}.pt'.format(ep)
            savedir = os.path.join('checkpoints', 'drl_grid')
            if not os.path.exists(savedir): os.makedirs(savedir)
            save_path = os.path.join(savedir, savename)
            torch.save({'model': model.state_dict()}, save_path)
            print(f'Saved to {save_path}!!!', flush=True)
            if val_set is not None:
                wer, cer = evaluate(save_path, val_set, batch_size=50)
                print(f'Val WER: {wer}, CER: {cer}', flush=True)
                if wer < best_wer:
                    best_wer, best_cer = wer, cer
                print(f'Best WER: {best_wer}, CER: {best_cer}', flush=True)
    '''

    for ep in range(1, 1 + epochs):
        ep_loss = 0.
        #for i, batch_data in enumerate(drl_data_loader):  # (S, 2, T, C, H, W)
        for i, batch_data in enumerate(vsr_data_loader):  # (S, 2, T, C, H, W)
            inputs = batch_data['vid'].to(DEVICE)
            targets = batch_data['txt'].to(DEVICE)
            spk_ids = batch_data['spk_id'].to(DEVICE)
            input_lens = batch_data['vid_lens'].to(DEVICE)
            target_lens = batch_data['txt_lens'].to(DEVICE)
            model.zero_grad()
            #loss = model(inputs, targets, input_lens, target_lens)
            #loss = model.calc_orth_loss(inputs, targets, spk_ids, input_lens, target_lens)
            loss = model.calc_orth_loss2(inputs, targets, spk_ids, input_lens, target_lens, mi_optimizer)
            #loss = model.calc_drl_loss(inputs, targets, input_lens, target_lens)
            loss.backward()
            vsr_optimizer.step()
            lr_scheduler.step()
            ep_loss += loss.data.item()
            print("Epoch {}, Iteration {}, loss: {:.4f}".format(ep, i + 1, loss.data.item()), flush=True)
        if ep % 1 == 0:
            print("Epoch {}, loss: {:.4f}".format(ep, ep_loss), flush=True)
            savename = 'iter_{}.pt'.format(ep)
            savedir = os.path.join('checkpoints', 'drl_grid2')
            if not os.path.exists(savedir): os.makedirs(savedir)
            save_path = os.path.join(savedir, savename)
            torch.save({'model': model.state_dict()}, save_path)
            print(f'Saved to {save_path}!!!', flush=True)
            if val_set is not None:
                wer, cer = evaluate(save_path, val_set, batch_size=50)
                print(f'Val WER: {wer}, CER: {cer}', flush=True)
                if wer < best_wer:
                    best_wer, best_cer = wer, cer
                print(f'Best WER: {best_wer}, CER: {best_cer}', flush=True)



def adapt(model_path, data_path, lr=1e-4, epochs=100, batch_size=50):
    model = DRLModel(28).to(DEVICE)
    print(sum(param.numel() for param in model.parameters()) / 1e6, 'M')
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location='cpu')
        states = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(states, strict=False)
        print('loading weights ...')
    # model.model.reset_params()
    model.train()
    print(model)
    spk_data = [os.path.join(data_path, fn) for fn in os.listdir(data_path)]
    adapt_data = spk_data[:500]  # half
    # dataset = GRIDDataset(adapt_data[:20])  # 1min
    # dataset = GRIDDataset(adapt_data[:60])  # 3min
    dataset = GRIDDataset(adapt_data[:100])  # 5min
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)
    optimizer = optim.AdamW(model.model.adapter.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    # optimizer = optim.AdamW([*model.model.adanet.parameters(), model.model.sc], lr=lr, betas=(0.9, 0.98), eps=1e-9)
    # optimizer = optim.AdamW([*model.model.adanet.parameters(), *model.model.adanet2.parameters(), model.model.sc], lr=lr, betas=(0.9, 0.98), eps=1e-9)
    # optimizer = optim.AdamW([*model.model.fc.parameters(), *model.model.gru2.parameters()], lr=lr, betas=(0.9, 0.98), eps=1e-9)
    # lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_iters//10, num_training_steps=num_iters)
    for ep in range(epochs):
        for i, batch_data in enumerate(data_loader):
            inputs = batch_data['vid'].to(DEVICE)
            targets = batch_data['txt'].to(DEVICE)
            input_lens = batch_data['vid_lens'].to(DEVICE)
            target_lens = batch_data['txt_lens'].to(DEVICE)
            model.zero_grad()
            loss = model(inputs, targets, input_lens, target_lens)
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()
            if (i + 1) % 10 == 0:
                print("Epoch {}, Iteration {}, loss: {:.4f}".format(ep + 1, i + 1, loss.data.item()), flush=True)
        savename = 'vanilla_iter_{}.pt'.format(ep + 1)
        savedir = os.path.join('checkpoints', 'adapt_grid')
        if not os.path.exists(savedir): os.makedirs(savedir)
        torch.save({'model': model.state_dict()}, os.path.join(savedir, savename))
        print(f'Saved to {savename}.')


@torch.no_grad()
def evaluate(model_path, dataset, batch_size=50):
    model = DRLModel(len(dataset.vocab), 29).to(DEVICE)
    # checkpoint = torch.load(opt.load, map_location=lambda storage, loc: storage)
    checkpoint = torch.load(model_path, map_location='cpu')
    states = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(states)
    model.eval()
    #print(model)
    print(len(dataset))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=5)
    preds = []
    refs = []
    PAD_ID, BOS_ID, EOS_ID = (dataset.vocab.index(x) for x in [PAD, BOS, EOS])
    for batch_data in data_loader:
        vid_inp = batch_data['vid'].to(DEVICE)
        tgt_txt = batch_data['txt'].to(DEVICE)
        input_lens = batch_data['vid_lens'].to(DEVICE)
        #output = model.greedy_decode(vid_inp, input_lens)
        output = model.beam_decode(vid_inp, input_lens, bos_id=BOS_ID, eos_id=EOS_ID, max_dec_len=30)
        pred = []
        gold = []
        for out, tgt in zip(output, tgt_txt):
            #pred.append(''.join([dataset.vocab[i] for i in torch.unique_consecutive(out).tolist() if i != 0]))
            #gold.append(''.join([dataset.vocab[i] for i in tgt.tolist() if i != 0]))
            pred.append(' '.join([dataset.vocab[i] for i in torch.unique_consecutive(out).tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]))
            #pred.append(' '.join([dataset.vocab[i] for i in out if i != 0]))
            gold.append(' '.join([dataset.vocab[i] for i in tgt.tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]))
        #print(pred, gold)
        preds.extend(pred)
        refs.extend(gold)
    test_wer, test_cer = wer(refs, preds), cer(refs, preds)
    print('JIWER wer: {:.4f}, cer: {:.4f}'.format(test_wer, test_cer))
    return test_wer, test_cer
    


if __name__ == '__main__':
    seed = 1347
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    data_root = r'E:\GRID\LIP_160_80\lip'
    #train_set = GRIDDataset(data_root, r'data\unseen_train.json', phase='train')
    val_set = GRIDDataset(data_root, r'data\unseen_val.json', phase='test')
    #train(train_set, val_set, lr=1e-4, epochs=50, batch_size=32)
    #test_set = GRIDDataset(data_root, r'data\unseen_val.json', phase='test')
    #evaluate('checkpoints/vsr_grid2/iter_15.pt', test_set, batch_size=50)

    vsr_set = GRIDDataset(data_root, r'data\unseen_train.json', phase='train')
    spk_set = GRIDDataset(data_root, r'data\unseen_train.json', phase='drl_train', sample_size=32)
    drl_set = GRIDDataset(data_root, r'data\unseen_train.json', phase='drl_train', sample_size=32)
    #drl_train(vsr_set, spk_set, drl_set, val_set, lr=1e-4, epochs=50, batch_size=32, model_path=None)
    drl_train(vsr_set, spk_set, drl_set, val_set, lr=1e-4, epochs=50, batch_size=32, model_path='checkpoints/drl_grid2/spk.pt')
    #test_set = GRIDDataset(r'E:\GRID\LIP_160_80\lip', r'data\unseen_val.json', phase='test')
    #evaluate('checkpoints/drl_grid/iter_20.pt', val_set, batch_size=50)

    # adapt('checkpoints/grid3/vanilla_iter_32.pt', r'E:\GRID\LIP_160_80\lip\s22', lr=1e-4, epochs=50, batch_size=10)
    # evaluate('checkpoints/adapt_grid/vanilla_iter_50.pt', r'E:\GRID\LIP_160_80\lip\s1', batch_size=50)

