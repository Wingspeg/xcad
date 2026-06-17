MAX_TRIPLES_PER_STEP = 2048
# @Time    : 2019-08-10 11:20
# @Author  : Lee_zix
# @Email   : Lee_zix@163.com
# @File    : main.py
# @Software: PyCharm
"""
The entry of the KGEvolve
"""

import argparse
import itertools
import os
import sys
import time
import pickle
import copy

import dgl
import numpy as np
import torch
# from tqdm import tqdm
import random
sys.path.append("..")
from rgcn import utils
from rgcn.utils import build_sub_graph, sort_and_rank
from src.rrgcn import RecurrentRGCN
from src.hyperparameter_range import hp_range
import torch.nn.modules.rnn
from collections import defaultdict
from rgcn.knowledge_graph import _read_triplets_as_list
# os.environ['KMP_DUPLICATE_LIB_OK']='True'


def print_all_model_parameters(model):
    print('\nModel Parameters')
    print('--------------------------')
    for name, param in model.named_parameters():
        print(name, param.numel(), 'requires_grad={}'.format(param.requires_grad))
    param_sizes = [param.numel() for param in model.parameters()]
    print('Total # parameters = {}'.format(sum(param_sizes)))
    print('--------------------------')
    print()


def temporal_regularization(params1, params2):
    regular = 0
    for (param1, param2) in zip(params1, params2):
        regular += torch.norm(param1 - param2, p=2)
    # print(regular)
    return regular
    # param_sizes = [param.numel() for param in model.parameters()]    


def test(model, history_len, history_list, test_list, num_rels, num_nodes, use_cuda, all_ans_list, all_ans_r_list, model_name, mode):
    """
    :param model: model used to test
    :param history_list:    all input history snap shot list, not include output label train list or valid list
    :param test_list:   test triple snap shot list
    :param num_rels:    number of relations
    :param num_nodes:   number of nodes
    :param use_cuda:
    :param all_ans_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param all_ans_r_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param model_name:
    :param mode
    :return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r
    """
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    ranks_raw_r, ranks_filter_r, mrr_raw_list_r, mrr_filter_list_r = [], [], [], []

    # 按关系类型分组收集 filter rank
    r3_ranks_filter = []       # r3_drives: id=3(正) 和 id=7(逆=3+4)
    relX_ranks = {0: [], 1: [], 2: []}  # placement, r1_suits, r2_requires 正向

    idx = 0
    start_time = len(history_list)
    if mode == "test":
        # test mode: load parameter form file
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        print("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))  # use best stat checkpoint
        print("\n"+"-"*10+"start testing"+"-"*10+"\n")
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    # do not have inverse relation in test input
    input_list = [snap for snap in history_list[-history_len:]]

    for time_idx, test_snap in enumerate(test_list):
        tc = start_time + time_idx
        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in input_list]
    
        test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        rank_raw, rank_filter = model.predict(history_glist, test_triples_input, use_cuda,
                                             all_ans_ent=all_ans_list[tc], eval_bs=1000)

        # ----- 按 relation 分组收集 rank -----
        rel_ids = test_triples_input[:, 1].cpu()         # [2*N_snap], relation id 含逆 (run_experiment 里 add_inverse 已加)
        # r3_drives: 正向=3, 逆向=7(正向+4)
        r3_mask = (rel_ids == 3) | (rel_ids == 7)
        r3_ranks_filter.append(rank_filter[r3_mask].cpu())
        for rel in [0, 1, 2]:
            relX_ranks[rel].append(rank_filter[rel_ids == rel].cpu())

        # entity rank 直接 append（无需再调 get_total_rank）
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        mrr_raw_list.append(torch.mean(1.0 / rank_raw.float()).item())
        mrr_filter_list.append(torch.mean(1.0 / rank_filter.float()).item())

        # relation 路：score_rel 全零无意义，快速跳过
        ranks_raw_r.append(torch.zeros(1))
        ranks_filter_r.append(torch.zeros(1))
        mrr_raw_list_r.append(0.0)
        mrr_filter_list_r.append(0.0)

        # reconstruct history graph list
        input_list.pop(0)
        input_list.append(test_snap)
        idx += 1

    mrr_raw = utils.stat_ranks(ranks_raw, "raw_ent")
    mrr_filter = utils.stat_ranks(ranks_filter, "filter_ent")
    mrr_raw_r = utils.stat_ranks(ranks_raw_r, "raw_rel")
    mrr_filter_r = utils.stat_ranks(ranks_filter_r, "filter_rel")

    # ----- 按关系类型分组打印 MRR -----
    def stat_ranks_by_rel(ranks_list, name):
        if len(ranks_list) == 0:
            return None
        all_ranks = torch.cat(ranks_list).float()
        mrr  = torch.mean(1.0 / all_ranks).item()
        h1   = torch.mean((all_ranks <= 1).float()).item()
        h3   = torch.mean((all_ranks <= 3).float()).item()
        h10  = torch.mean((all_ranks <= 10).float()).item()
        return mrr, h1, h3, h10

    print()
    print("=" * 30 + " Per-Relation filter MRR " + "=" * 30)
    r3_stat = stat_ranks_by_rel(r3_ranks_filter, "r3_drives")
    if r3_stat:
        mrr, h1, h3, h10 = r3_stat
        print(f"  r3_drives (attr, id=3+7):  MRR={mrr:.6f}  H@1={h1:.6f}  H@3={h3:.6f}  H@10={h10:.6f}")
    else:
        print("  r3_drives: no data")
    rel_names = {0: "placement", 1: "r1_suits", 2: "r2_requires"}
    for rel in [0, 1, 2]:
        st = stat_ranks_by_rel(relX_ranks[rel], rel_names[rel])
        if st:
            mrr, h1, h3, h10 = st
            print(f"  {rel_names[rel]:25s} MRR={mrr:.6f}  H@1={h1:.6f}  H@3={h3:.6f}  H@10={h10:.6f}")
        else:
            print(f"  {rel_names[rel]}: no data")
    print("=" * 70)

    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r



def continual_test(model, test_history_len, history_list, data_list, num_rels, num_nodes, use_cuda, all_ans_list, all_ans_r_list, model_name, mode):
    """
    :param model: model used to test
    :param history_list:    all input history snap shot list, not include output label train list or valid list
    :param test_list:   test triple snap shot list
    :param num_rels:    number of relations
    :param num_nodes:   number of nodes
    :param use_cuda:
    :param all_ans_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param all_ans_r_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param model_name:
    :param mode
    :return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r
    """
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    ranks_raw_r, ranks_filter_r, mrr_raw_list_r, mrr_filter_list_r = [], [], [], []

    # load pretrained model which valid in the whole valid data
    start_idx = len(history_list)
    # if test, load the model fine tuned at the last timestamp at the valid dataset
    # else load the pretrained model.
    # if mode=="test":
    #     model_name = "{}-{}".format(model_name, start_idx-1)
    if not os.path.exists(model_name):
        print("Pretrain the model first before continual learning...")
        sys.exit()
    else:
        if mode=="test":
            checkpoint = torch.load("{}-flr{}-norm{}-{}".format(model_name, args.ft_lr, args.norm_weight, start_idx-1), map_location=torch.device(args.gpu))
            init_checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            # print(model_name)
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
            init_checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        print("Load pretrain model: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))  # use best stat checkpoint
        print("\n"+"-"*10+"start continual learning"+"-"*10+"\n")
        model.load_state_dict(checkpoint['state_dict'])
        # save an init model for analysis
        model_initial = copy.deepcopy(model)
        model_initial.load_state_dict(init_checkpoint['state_dict']) 
        model_initial.eval()
        # parameter for the temporal normalize at the first timestamp
        previous_param = [param.detach().clone() for param in model.parameters()]

        model.eval()
        epoch = checkpoint['epoch']
        
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.ft_lr, weight_decay=0)
    
    # Note: do not have inverse relation in test input
    valid_input_list = [snap for snap in history_list[-args.test_history_len-2:-2]] # history for ft training (, tc-2)
    valid_snap = history_list[-2]    # snapshot for ft training at snapshot at tc-2
    ft_input_list = [snap for snap in history_list[-args.test_history_len-1:-1]] # history for ft validation (,tc-1)
    ft_snap = history_list[-1]   # snapshot for ft validation snapshot at tc-1
    test_input_list = [snap for snap in history_list[-args.test_history_len:]]  # history for testing (, tc)
    # ft_output_list = history_list[-args.test_history_len:]
     
    # starting continual learning
    for time_idx, test_snap in enumerate(data_list):
        tc = start_idx + time_idx 
        print("-----------------------{}-----------------------".format(tc))
        # step 1: get the history graphs for ft training : ft_input_list -> ft_snapshot
        ft_history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in ft_input_list]
        ft_tensor = torch.LongTensor(ft_snap).cuda() if use_cuda else torch.LongTensor(ft_snap)
        ft_tensor = ft_tensor.to(args.gpu)
        ft_output_tensor = [torch.from_numpy(_).long().cuda() for _ in test_input_list] if use_cuda else [torch.from_numpy(_).long() for _ in test_input_list]
   
        # step 2: get the history graphs for ft validation : valid_input_list -> valid_snap
        valid_history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in valid_input_list]
        valid_tensor = torch.LongTensor(valid_snap).cuda() if use_cuda else torch.LongTensor(valid_snap)
        valid_tensor = valid_tensor.to(args.gpu)

        # step 2: prepare inputs for test
        test_history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in test_input_list] 
        test_tensor = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        test_tensor = test_tensor.to(args.gpu)
       
        # result of the pre-trained model on validation set (tc-1)
        rank_raw, rank_filter = model.predict(valid_history_glist, valid_tensor, use_cuda,
                                              all_ans_ent=all_ans_list[tc-2], eval_bs=1000)
        mrr_filter_valid_snap = torch.mean(1.0 / rank_filter.float()).item()
        mrr_valid_snap = torch.mean(1.0 / rank_raw.float()).item()
        rank_raw_r = rank_filter_r = torch.zeros(1)
        mrr_filter_valid_snap_r = mrr_valid_snap_r = 0.0

        # result of the pre-trained model on test set (tc)
        rank_raw, rank_filter = model_initial.predict(test_history_glist, test_tensor, use_cuda,
                                                      all_ans_ent=all_ans_list[tc], eval_bs=1000)
        mrr_filter_test_snap = torch.mean(1.0 / rank_filter.float()).item()
        mrr_test_snap = torch.mean(1.0 / rank_raw.float()).item()
        print("Pretrained Model : test mrr ", mrr_filter_test_snap)

        # result of the last step fine-tuned model on test set (tc)
        rank_raw, rank_filter = model.predict(test_history_glist, test_tensor, use_cuda,
                                              all_ans_ent=all_ans_list[tc], eval_bs=1000)
        mrr_filter_test_snap = torch.mean(1.0 / rank_filter.float()).item()
        mrr_test_snap = torch.mean(1.0 / rank_raw.float()).item()
        print("Continual Model : test mrr before ft ", mrr_filter_test_snap) 

        # init mrr for validation
        best_mrr = mrr_filter_valid_snap
        
        ft_epoch, losses = 0, []

        while ft_epoch < args.ft_epochs:
            # print(best_mrr)
            model.train()
            # print(ft_tensor.size()) 
            loss  = model.get_ft_loss(ft_history_glist, ft_output_tensor, use_cuda)
            loss_norm = temporal_regularization(model.parameters(), previous_param)
            
            loss += args.norm_weight*loss_norm

            losses.append(loss.item())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
            optimizer.step()
            optimizer.zero_grad()            
            
            model.eval()
            
            # validation on tc-1 snapshot
            rank_raw, rank_filter = model.predict(valid_history_glist, valid_tensor, use_cuda,
                                                  all_ans_ent=all_ans_list[tc-2], eval_bs=1000)
            mrr_filter_valid_snap = torch.mean(1.0 / rank_filter.float()).item()
            mrr_valid_snap = torch.mean(1.0 / rank_raw.float()).item()
            rank_raw_r = rank_filter_r = torch.zeros(1)
            mrr_filter_valid_snap_r = mrr_valid_snap_r = 0.0

            # update best_mrr 
            ft_epoch += 1
            
            if mrr_filter_valid_snap >= best_mrr:
                print(mrr_filter_valid_snap, best_mrr,"model updated")
                best_mrr = mrr_filter_valid_snap
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, "{}-flr{}-norm{}-{}".format(model_name, args.ft_lr, args.norm_weight, tc))
                # update_flag = True
                # model param at timestamp tc
                
            else:  
                if not os.path.exists("{}-flr{}-norm{}-{}".format(model_name, args.ft_lr, args.norm_weight, tc)):
                    print(mrr_filter_valid_snap, best_mrr, "copy model at {}".format(tc-1))
                    if mode == "valid" and time_idx == 0:
                        checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
                    else:
                        checkpoint = torch.load("{}-flr{}-norm{}-{}".format(model_name, args.ft_lr, args.norm_weight, tc-1), map_location=torch.device(args.gpu))
                    model.load_state_dict(checkpoint['state_dict'])
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, "{}-flr{}-norm{}-{}".format(model_name, args.ft_lr, args.norm_weight, tc))
                    if ft_epoch > 3:
                        break
                else:
                    print("exist best model, skip save")
                    break
        
        # save the best parameter in model-tc
        previous_param = [param.detach().clone() for param in model.parameters()]  
        # ---------------start evaluate test snaoshot---------------
        
        # step 1: load current model
        checkpoint = torch.load("{}-flr{}-norm{}-{}".format(model_name, args.ft_lr, args.norm_weight, tc), map_location=torch.device(args.gpu))
        model.load_state_dict(checkpoint['state_dict']) 
        model.eval()
        # step 3: start test
        rank_raw, rank_filter = model.predict(test_history_glist, test_tensor, use_cuda,
                                              all_ans_ent=all_ans_list[tc], eval_bs=1000)
        # step 4: evaluation
        mrr_filter_snap = torch.mean(1.0 / rank_filter.float()).item()
        mrr_snap = torch.mean(1.0 / rank_raw.float()).item()
        print("Continual Model : ***test mrr*** ", mrr_filter_snap)
        
        # step 5: update history glist and prepare inputs
        ft_input_list.pop(0)
        ft_input_list.append(ft_snap)
        valid_input_list.pop(0)
        valid_input_list.append(valid_snap)
        test_input_list.pop(0)
        test_input_list.append(test_snap)
        # ft_output_list.pop(0)
        # ft_output_list.append(test_snap) 
        
        valid_snap = ft_snap.copy()
        ft_snap = test_snap.copy()

        
        # step 6: save results
        # used to global statistic
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        
        # used to show slide results
        mrr_raw_list.append(mrr_snap)
        mrr_filter_list.append(mrr_filter_snap)
        ranks_raw_r.append(rank_raw_r)
        ranks_filter_r.append(rank_filter_r)
        mrr_raw_list_r.append(mrr_snap_r)
        mrr_filter_list_r.append(mrr_filter_snap_r)

    
    mrr_raw = utils.stat_ranks(ranks_raw, "raw_ent")
    mrr_filter = utils.stat_ranks(ranks_filter, "filter_ent")
    mrr_raw_r = utils.stat_ranks(ranks_raw_r, "raw_rel")
    mrr_filter_r = utils.stat_ranks(ranks_filter_r, "filter_rel")
    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r


def add_inverse(snap_list, num_rel):
    # add inverse triples to train_list
    all_list = []
    for triples in snap_list:
        inverse_triples = triples[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + num_rel
        all_triples = np.concatenate((triples, inverse_triples))
        all_list.append(all_triples)
    return all_list


def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
    # load configuration for grid search the best configuration
    if n_hidden:
        args.n_hidden = n_hidden
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases

    # load graph data
    print("loading graph data")
    data = utils.load_data(args.dataset)
    train_list = utils.split_by_time(data.train)
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)
    total_data = np.concatenate((data.train, data.valid, data.test), axis=0)
    print("total data length ", len(total_data))
    num_nodes = data.num_nodes
    num_rels = data.num_rels
    all_ans_list = utils.load_all_answers_for_time_filter(total_data, num_rels, num_nodes, False)
    all_ans_list_r = utils.load_all_answers_for_time_filter(total_data, num_rels, num_nodes, True)

    train_list = add_inverse(train_list, num_rels)
    valid_list = add_inverse(valid_list, num_rels)
    test_list = add_inverse(test_list, num_rels)


    test_model_name = "{}-{}-ly{}-dilate{}-his{}-dp{}-gpu{}"\
        .format(args.encoder, args.decoder, args.n_layers, args.dilate_len, args.test_history_len, args.dropout, args.gpu)
    if not os.path.exists('../models/{}/'.format(args.dataset)):
        os.makedirs('../models/{}/'.format(args.dataset))
    test_state_file = '../models/{}/{}'.format(args.dataset, test_model_name) 
    print("Sanity Check: stat name : {}".format(test_state_file))
    print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    # create stat
    model = RecurrentRGCN(args.decoder,
                          args.encoder,
                        num_nodes,
                        num_rels,
                        args.n_hidden,
                        args.opn,
                        sequence_len=args.train_history_len,
                        num_bases=args.n_bases,
                        num_basis=args.n_basis,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        self_loop=args.self_loop,
                        skip_connect=args.skip_connect,
                        layer_norm=args.layer_norm,
                        input_dropout=args.input_dropout,
                        hidden_dropout=args.hidden_dropout,
                        feat_dropout=args.feat_dropout,
                        entity_prediction=args.entity_prediction,
                        relation_prediction=args.relation_prediction,
                        use_cuda=use_cuda,
                        gpu = args.gpu)

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    if args.test == 1:  # normal test on validation set  Note that mode=test
        if os.path.exists(test_state_file):
            mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model, 
                                                args.test_history_len, 
                                                train_list, 
                                                valid_list, 
                                                num_rels, 
                                                num_nodes, 
                                                use_cuda, 
                                                all_ans_list, 
                                                all_ans_list_r, 
                                                test_state_file, 
                                                "test")                                 
    elif args.test == 2:    # normal test on test set
        if os.path.exists(test_state_file):
            mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model, 
                                                args.test_history_len, 
                                                train_list+valid_list, 
                                                test_list, 
                                                num_rels, 
                                                num_nodes, 
                                                use_cuda, 
                                                all_ans_list, 
                                                all_ans_list_r, 
                                                test_state_file, 
                                                "test")
    elif args.test == 3:    # continual test the validataion set 
        if os.path.exists(test_state_file):
                mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = continual_test(model,
                                                args.test_history_len,  
                                                train_list, 
                                                valid_list, 
                                                num_rels, 
                                                num_nodes, 
                                                use_cuda, 
                                                all_ans_list, 
                                                all_ans_list_r, 
                                                test_state_file, 
                                                "valid")
    elif args.test == 4:    # continual test the testing set
        if os.path.exists(test_state_file):
                mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = continual_test(model, 
                                                args.test_history_len, 
                                                train_list+valid_list, 
                                                test_list, 
                                                num_rels, 
                                                num_nodes, 
                                                use_cuda, 
                                                all_ans_list, 
                                                all_ans_list_r, 
                                                test_state_file, 
                                                "test")
    elif args.test == -1:
        print("----------------------------------------start pre training model with history length {}----------------------------------------\n".format(args.start_history_len))
        model_name = "{}-{}-ly{}-dilate{}-his{}-dp{}-gpu{}"\
            .format(args.encoder, args.decoder, args.n_layers, args.dilate_len, args.start_history_len, args.dropout, args.gpu)
        if not os.path.exists('../models/{}/'.format(args.dataset)):
            os.makedirs('../models/{}/'.format(args.dataset))
        model_state_file = '../models/{}/{}'.format(args.dataset, model_name) 
        print("Sanity Check: stat name : {}".format(model_state_file))
        print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))
            
        best_mrr = 0
        best_epoch = 0
        for epoch in range(args.n_epochs):
            model.train()
            losses = []

            idx = [_ for _ in range(len(train_list))]
            random.shuffle(idx)
            for train_sample_num in idx:
                if train_sample_num == 0 or train_sample_num == 1: continue
                if train_sample_num - args.start_history_len<0:
                    input_list = train_list[0: train_sample_num]
                    output = train_list[1:train_sample_num+1]
                else:
                    input_list = train_list[train_sample_num-args.start_history_len: train_sample_num]
                    output = train_list[train_sample_num-args.start_history_len+1:train_sample_num+1]

                # generate history graph
                history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
                output = [torch.from_numpy(_).long().cuda() for _ in output] if use_cuda else [torch.from_numpy(_).long() for _ in output]

                # 边采样：限制传给 get_loss 的 triples 规模
                if output[-1].shape[0] > MAX_TRIPLES_PER_STEP:
                    indices = torch.randperm(output[-1].shape[0])[:MAX_TRIPLES_PER_STEP]
                    output[-1] = output[-1][indices]

                loss = model.get_loss(history_glist, output[-1], None, use_cuda)
                losses.append(loss.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                optimizer.step()
                optimizer.zero_grad()

            print("His {:04d}, Epoch {:04d} | Ave Loss: {:.4f} | Best MRR {:.4f} | Model {} "
                .format(args.start_history_len, epoch, np.mean(losses), best_mrr, model_name))

            # validation        
            if epoch % args.evaluate_every == 0:
                mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model,
                                                                    args.start_history_len, 
                                                                    train_list, 
                                                                    valid_list, 
                                                                    num_rels, 
                                                                    num_nodes, 
                                                                    use_cuda, 
                                                                    all_ans_list, 
                                                                    all_ans_list_r, 
                                                                    model_state_file, 
                                                                    mode="train")
                
                if not args.relation_evaluation:  # entity prediction evalution
                    if mrr_filter< best_mrr:
                        if epoch >= args.n_epochs or epoch - best_epoch > 5:
                            break
                    else:
                        best_mrr = mrr_filter
                        best_epoch = epoch
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
                else:
                    if mrr_filter_r < best_mrr:
                        if epoch >= args.n_epochs:
                            break
                    else:
                        best_mrr = mrr_filter_r
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model, 
                                                            args.start_history_len, 
                                                            train_list+valid_list,
                                                            test_list, 
                                                            num_rels, 
                                                            num_nodes, 
                                                            use_cuda, 
                                                            all_ans_list, 
                                                            all_ans_list_r, 
                                                            model_state_file, 
                                                            mode="test")
        return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r
    elif args.test == 0:
        # load best model with start history length
        init_state_file = '../models/{}/'.format(args.dataset) + "{}-{}-ly{}-dilate{}-his{}-dp{}-gpu{}".format(args.encoder, args.decoder, args.n_layers, args.dilate_len, args.start_history_len, args.dropout, args.gpu)
        init_checkpoint = torch.load(init_state_file, map_location=torch.device(args.gpu))
        print("Load Previous Model name: {}. Using best epoch : {}".format(init_state_file, init_checkpoint['epoch']))  # use best stat checkpoint
        print("\n"+"-"*10+"Load model with history length {}".format(args.start_history_len)+"-"*10+"\n")
        model.load_state_dict(init_checkpoint['state_dict'])
        test_history_len = args.start_history_len
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model, 
                                                            args.start_history_len,
                                                            train_list+valid_list,
                                                            test_list, 
                                                            num_rels, 
                                                            num_nodes, 
                                                            use_cuda, 
                                                            all_ans_list, 
                                                            all_ans_list_r, 
                                                            init_state_file, 
                                                            mode="test")
        best_mrr_list = [mrr_filter.item()]                                                   
        # start knowledge distillation
        ks_idx = 0
        for history_len in range(args.start_history_len+1, args.train_history_len+1, 1):
            # current model
            print("best mrr list :", best_mrr_list)
            # lr = 0.1*args.lr - 0.002*args.lr*ks_idx
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1*args.lr, weight_decay=0.00001)
            model_name = "{}-{}-ly{}-dilate{}-his{}-dp{}-gpu{}".format(args.encoder, args.decoder, args.n_layers, args.dilate_len, history_len, args.dropout, args.gpu)
            model_state_file = '../models/{}/'.format(args.dataset) + model_name
            print("Sanity Check: stat name : {}".format(model_state_file))

            # load model with the least history length
            prev_model_name = "{}-{}-ly{}-dilate{}-his{}-dp{}-gpu{}".format(args.encoder, args.decoder, args.n_layers, args.dilate_len, history_len-1, args.dropout, args.gpu)
            prev_state_file = '../models/{}/'.format(args.dataset) + prev_model_name
            checkpoint = torch.load(prev_state_file, map_location=torch.device(args.gpu))
            # print("Load Previous Model name: {}. Using best epoch : {}".format(prev_model_name, checkpoint['epoch']))  # use best stat checkpoint
            model.load_state_dict(checkpoint['state_dict'])
            # prev_model = copy.deepcopy(model)
            # prev_model.eval()
            print("\n"+"-"*10+"start knowledge distillation for history length at "+ str(history_len)+"-"*10+"\n")
 
            best_mrr = 0
            best_epoch = 0
            for epoch in range(args.n_epochs):
                model.train()
                losses = []

                idx = [_ for _ in range(len(train_list))]
                random.shuffle(idx)
                for train_sample_num in idx:
                    if train_sample_num == 0 or train_sample_num == 1: continue
                    if train_sample_num - history_len<0:
                        input_list = train_list[0: train_sample_num]
                        output = train_list[1:train_sample_num+1]
                    else:
                        input_list = train_list[train_sample_num - history_len: train_sample_num]
                        output = train_list[train_sample_num-history_len+1:train_sample_num+1]

                    # generate history graph
                    history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
                    output = [torch.from_numpy(_).long().cuda() for _ in output] if use_cuda else [torch.from_numpy(_).long() for _ in output]

                    # 边采样：限制传给 get_loss 的 triples 规模
                    if output[-1].shape[0] > MAX_TRIPLES_PER_STEP:
                        indices = torch.randperm(output[-1].shape[0])[:MAX_TRIPLES_PER_STEP]
                        output[-1] = output[-1][indices]

                    loss= model.get_loss(history_glist, output[-1], None, use_cuda)
                    # print(loss)
                    losses.append(loss.item())

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                    optimizer.step()
                    optimizer.zero_grad()

                print("His {:04d}, Epoch {:04d} | Ave Loss: {:.4f} |Best MRR {:.4f} | Model {} "
                    .format(history_len, epoch, np.mean(losses), best_mrr, model_name))

                # validation
                if epoch % args.evaluate_every == 0:
                    mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model,
                                                                        history_len,
                                                                        train_list, 
                                                                        valid_list, 
                                                                        num_rels, 
                                                                        num_nodes, 
                                                                        use_cuda, 
                                                                        all_ans_list, 
                                                                        all_ans_list_r, 
                                                                        model_state_file, 
                                                                        mode="train")
                    
                
                    if mrr_filter< best_mrr:
                        if epoch >= args.n_epochs or epoch-best_epoch>2:
                            break
                    else:
                        best_mrr = mrr_filter
                        best_epoch = epoch
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)

  
            mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model, 
                                                history_len,
                                                train_list,
                                                valid_list, 
                                                num_rels, 
                                                num_nodes, 
                                                use_cuda, 
                                                all_ans_list, 
                                                all_ans_list_r, 
                                                model_state_file, 
                                                mode="test")
            ks_idx += 1
            if mrr_filter.item() < max(best_mrr_list):
                test_history_len = history_len-1
                break
            else:
                best_mrr_list.append(mrr_filter.item())

        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model, 
                                                            test_history_len,
                                                            train_list+valid_list,
                                                            test_list, 
                                                            num_rels, 
                                                            num_nodes, 
                                                            use_cuda, 
                                                            all_ans_list, 
                                                            all_ans_list_r, 
                                                            prev_state_file, 
                                                            mode="test")
        return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='REGCN')

    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use")
    parser.add_argument("--test", type=int, default=0,
                        help="1: formal test 2: continual test")
  
    parser.add_argument("--run-statistic", action='store_true', default=False,
                        help="statistic the result")

    parser.add_argument("--relation-evaluation", action='store_true', default=False,
                        help="save model accordding to the relation evalution")

    
    # configuration for encoder RGCN stat
    parser.add_argument("--weight", type=float, default=1,
                        help="weight of static constraint")
    parser.add_argument("--task-weight", type=float, default=1,
                        help="weight of entity prediction task")
    parser.add_argument("--kl-weight", type=float, default=0.7,
                        help="weight of entity prediction task")
   
    parser.add_argument("--encoder", type=str, default="uvrgcn",
                        help="method of encoder")

    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")

    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=False,
                        help="add relation prediction loss")
    parser.add_argument("--entity-prediction", action='store_true', default=False,
                        help="add entity prediction loss")


    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=500,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--ft_epochs", type=int, default=30,
                        help="number of minimum fine-tuning epoch")
    parser.add_argument("--ft_lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--norm_weight", type=float, default=0.1,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=20,
                        help="perform evaluation every n epochs")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse",
                        help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int, default=10,
                        help="history length")
    parser.add_argument("--test-history-len", type=int, default=20,
                        help="history length for test")
    parser.add_argument("--start-history-len", type=int, default=1,
                    help="start history length")
    parser.add_argument("--dilate-len", type=int, default=1,
                        help="dilate history graph")

    # configuration for optimal parameters
    parser.add_argument("--grid-search", action='store_true', default=False,
                        help="perform grid search for best configuration")
    parser.add_argument("-tune", "--tune", type=str, default="n_hidden,n_layers,dropout,n_bases",
                        help="stat to use")
    parser.add_argument("--num-k", type=int, default=500,
                        help="number of triples generated")


    args = parser.parse_args()
    print(args)
    if args.grid_search:
        out_log = '{}.{}.gs'.format(args.dataset, args.encoder+"-"+args.decoder)
        o_f = open(out_log, 'w')
        print("** Grid Search **")
        o_f.write("** Grid Search **\n")
        hyperparameters = args.tune.split(',')

        if args.tune == '' or len(hyperparameters) < 1:
            print("No hyperparameter specified.")
            sys.exit(0)
        grid = hp_range[hyperparameters[0]]
        for hp in hyperparameters[1:]:
            grid = itertools.product(grid, hp_range[hp])
        hits_at_1s = {}
        hits_at_10s = {}
        mrrs = {}
        grid = list(grid)
        print('* {} hyperparameter combinations to try'.format(len(grid)))
        o_f.write('* {} hyperparameter combinations to try\n'.format(len(grid)))
        o_f.close()

        for i, grid_entry in enumerate(list(grid)):

            o_f = open(out_log, 'a')

            if not (type(grid_entry) is list or type(grid_entry) is list):
                grid_entry = [grid_entry]
            grid_entry = utils.flatten(grid_entry)
            print('* Hyperparameter Set {}:'.format(i))
            o_f.write('* Hyperparameter Set {}:\n'.format(i))
            signature = ''
            print(grid_entry)
            o_f.write("\t".join([str(_) for _ in grid_entry]) + "\n")
            # def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
            mrr, hits, ranks = run_experiment(args, grid_entry[0], grid_entry[1], grid_entry[2], grid_entry[3])
            print("MRR (raw): {:.6f}".format(mrr))
            o_f.write("MRR (raw): {:.6f}\n".format(mrr))
            for hit in hits:
                avg_count = torch.mean((ranks <= hit).float())
                print("Hits (raw) @ {}: {:.6f}".format(hit, avg_count.item()))
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, avg_count.item()))
    # single run
    else:
        run_experiment(args)
    sys.exit()



