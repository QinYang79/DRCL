import torch
import numpy as np
from torch import optim
import cal_utils as utils
import torch.nn.functional as F
import data_loader
import scipy.io as sio
from loss import gce_loss, LossModule, feature_augmentation
import time
class Solver(object):
    def __init__(self, config, logger):
        self.logger = logger
        self.output_shape = config.output_shape
        data = data_loader.load_deep_features(config.datasets)
        self.datasets = config.datasets
        (self.train_data, self.train_labels, self.val_data, self.val_labels, self.test_data, self.test_labels) = data

        self.n_view = len(self.train_data)
        for v in range(self.n_view):
            if min(self.train_labels[v].shape) == 1:
                self.train_labels[v] = self.train_labels[v].reshape([-1])
            if min(self.val_labels[v].shape) == 1:
                self.val_labels[v] = self.val_labels[v].reshape([-1])
            if min(self.test_labels[v].shape) == 1:
                self.test_labels[v] = self.test_labels[v].reshape([-1])
        if len(self.train_labels[0].shape) == 1:
            self.classes = np.unique(np.concatenate(self.train_labels).reshape([-1]))
            self.classes = self.classes[self.classes >= 0]
            self.num_classes = len(self.classes)
        else:
            self.num_classes = self.train_labels[0].shape[1]

        if self.output_shape == -1:
            self.output_shape = self.num_classes

        self.dropout_prob = 0.5
        self.input_shape = [self.train_data[v].shape[1] for v in range(self.n_view)]
        self.lr = config.lr
        self.lr_SPL = config.lr_SPL
        self.wselect = config.wselect
        self.batch_size = config.batch_size
        self.alpha = config.alpha
        self.beta = config.beta
        self.eta = config.eta
        self.gamma = config.gamma
        self.view_id = config.view_id
        self.gpu_id = config.gpu_id
        self.epochs = config.epochs
        self.sample_interval = config.sample_interval
        self.just_val = config.just_val
        self.seed = config.seed
        
        print("datasets: %s, batch_size: %d, output_shape: %d, hyper-alpha1: %f, hyper-beta1：%f "% (
                (config.datasets, self.batch_size, self.output_shape, self.alpha, self.beta)))

        W = torch.Tensor(self.output_shape, self.output_shape)
        self.W = torch.nn.init.orthogonal_(W, gain=1)[:, 0: self.num_classes]
            
        self.runing_time = config.running_time


    def to_one_hot(self, x):
        if len(x.shape) == 1 or x.shape[1] == 1:
            one_hot = (self.classes.reshape([1, -1]) == x.reshape([-1, 1])).astype('float32')
            labels = one_hot
            y = torch.tensor(labels).cuda()
        else:
            y = torch.tensor(x.astype('float32')).cuda()
        return y

    def view_result(self, _acc):
        res = ''
        res += ((' - mean: %.5f' % (np.sum(_acc) / (self.n_view * (self.n_view - 1)))) + ' - detail:')
        for _i in range(self.n_view):
            for _j in range(self.n_view):
                if _i != _j:
                    res += ('%.5f' % _acc[_i, _j]) + ','
        return res


    def train(self):
        if self.view_id >= 0:
            W = sio.loadmat('PriorW/W_' + str(self.output_shape) + 'X' + str(self.num_classes) + self.datasets  + '.mat')['W']
            W = torch.tensor(W, requires_grad=False).cuda()
            start = time.time()
            self.train_view(self.view_id, W)
        else:
            if self.wselect:
                W = self.learning_best_prior(self.n_view)
                # W = self.load_prior(4) # different priors
            else:
                W = sio.loadmat('PriorW/W_' + str(self.output_shape) + 'X' + str(self.num_classes) + self.datasets  + '.mat')['W']
                W = torch.tensor(W, requires_grad=False).cuda()
            import torch.multiprocessing as mp
            mp = mp.get_context('spawn')
            process = []
            start = time.time()
            for v in range(self.n_view):
                self.train_view(v, W)
            #     process.append(mp.Process(target=self.train_view, args=(v,W)))
            #     process[v].daemon = True
            #     process[v].start()
            # for p in process:
            #     p.join()

        end = time.time()
        runing_time = end - start
        if self.runing_time:
            print('runing_time: ' + str(runing_time))
            return runing_time
        test_fea, test_lab, = [], []
        for v in range(self.n_view):
            tmp = sio.loadmat('features/' + self.datasets + '_' + str(v) + '.mat')
            test_fea.append(tmp['test_fea'])
            test_lab.append(tmp['test_lab'].reshape([-1,]) if min(tmp['test_lab'].shape) == 1 else tmp['test_lab'])
        test_results = utils.multi_test(test_fea, test_lab)
        print("test resutls@all:" + self.view_result(test_results))
        self.logger.info("test resutls@all:" + self.view_result(test_results))
        test_results = utils.multi_test(test_fea, test_lab, 50)
        print("test resutls@50:" + self.view_result(test_results))
        self.logger.info("test resutls@50:" + self.view_result(test_results))
        return test_results

    def train_view(self, view_id, W):

        from to_seed import to_seed
        to_seed(seed=self.seed)
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        from model import Data_Net
        W.requires_grad=False

        datanet = Data_Net(input_dim=self.input_shape[view_id], out_dim=self.output_shape).cuda()
        lossmodule = LossModule(W, self.alpha, self.beta, self.eta, self.gamma).cuda()
        get_grad_params = lambda model: [x for x in model.parameters() if x.requires_grad]
        params_dnet = get_grad_params(datanet)

        optimizer_dnet = optim.Adam(params_dnet, self.lr[view_id], [0.5, 0.999])

        best_loss = 1e9
        best_epoch = 0
        for epoch in range(self.epochs):
            batch_nums = int(self.train_labels[view_id].shape[0] / float(self.batch_size))

            rand_didx = np.arange(self.train_data[view_id].shape[0])
            np.random.shuffle(rand_didx)

            for batch_idx in range(batch_nums):
                didx = rand_didx[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]

                view_labs = self.to_one_hot(self.train_labels[view_id][didx])
                view_data = self.to_one_hot(self.train_data[view_id][didx])

                optimizer_dnet.zero_grad()

                data_fea = datanet(view_data)
                data_fea, view_labs = feature_augmentation(data_fea, view_labs, self.gamma)
                pred = data_fea.view([data_fea.shape[0], -1]).mm(W)
                    
                loss = lossmodule(data_fea, view_labs, pred, epoch + 1)

                loss.backward()
                optimizer_dnet.step()

                if ((epoch + 1) % self.sample_interval == 0) and (batch_idx == batch_nums - 1):
                    datanet.eval()
                    view_labs = self.to_one_hot(self.val_labels[view_id])
                    view_data = self.to_one_hot(self.val_data[view_id])

                    data_fea = datanet(view_data)
                    pred = data_fea.view([data_fea.shape[0], -1]).mm(W)
                    loss_val = lossmodule(data_fea, view_labs, pred, epoch + 1)
                    if loss_val < best_loss:
                        test_fea = datanet(self.to_one_hot(self.test_data[view_id]))
                        test_labs = self.test_labels[view_id]
                        test_pred = F.softmax(test_fea.view([test_fea.shape[0], -1]).mm(W), dim=1)

                        best_loss = loss_val
                        best_epoch = epoch + 1
                    print(('ViewID: %d, Epoch %d/%d, loss: %.4f, loss_val: %.4f') % (view_id, epoch + 1, self.epochs, loss, loss_val))
        print('best_epoch:', best_epoch)
        sio.savemat('features/' + self.datasets + '_' + str(view_id) + '.mat', {'test_fea':test_fea.cpu().detach().numpy(),
                                                                                'test_lab':test_labs,
                                                                                'test_pred':test_pred.cpu().detach().numpy()
                                                                                })
    def learning_best_prior(self, n_view):
        
        from to_seed import to_seed
        to_seed(seed=self.seed)
        import os
        import torch
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)

        from model import Data_Net
        max_acc = 0
        for view_id in range(n_view):
            print('cur_view_id: ', view_id)
            W = torch.tensor(self.W, requires_grad= True).cuda()
            W = torch.nn.Parameter(W)

            datanet = Data_Net(input_dim=self.input_shape[view_id], out_dim=self.output_shape).cuda()
            datanet.register_parameter('W', W)
            get_grad_params = lambda model: [x for x in model.parameters() if x.requires_grad]
            params_dnet = get_grad_params(datanet)
            optimizer_dnet = optim.Adam(params_dnet, self.lr_SPL, [0.5, 0.999])

            best_acc = 0
            best_epoch = 0

            for epoch in range(self.epochs):
                batch_nums = int(self.train_labels[view_id].shape[0] / float(self.batch_size))
                rand_didx = np.arange(self.train_data[view_id].shape[0])
                np.random.shuffle(rand_didx)
                for batch_idx in range(batch_nums):
                    didx = rand_didx[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]

                    view_labs = self.to_one_hot(self.train_labels[view_id][didx])
                    view_data = self.to_one_hot(self.train_data[view_id][didx])

                    optimizer_dnet.zero_grad()
                    data_fea = datanet(view_data)
                    pred = data_fea.view([data_fea.shape[0], -1]).mm(W)
            
                    # GCE
                    q = min(1., 0.01 * epoch)
                    loss = gce_loss(pred, view_labs, q)

                    loss.backward()
                    optimizer_dnet.step()

                    if ((epoch + 1) % self.sample_interval == 0) and (batch_idx == batch_nums - 1):
                        datanet.eval()
                        view_labs = self.to_one_hot(self.train_labels[view_id])
                        view_data = self.to_one_hot(self.train_data[view_id])

                        data_fea = datanet(view_data)
                        pred = data_fea.view([data_fea.shape[0], -1]).mm(W)
                        pred = F.softmax(pred, dim=1)
                        tmp = pred * view_labs
                        tmp = torch.sum(tmp, dim=1)
                        tmp = torch.mean(tmp)
                        if tmp > best_acc:
                            best_acc = tmp
                            bW = W
                            best_epoch = epoch + 1
                        print('view_id: %d,epoch: %d acc:%.4f'%(view_id, epoch+1, tmp))

                        test_labs = self.to_one_hot(self.test_labels[view_id])
                        test_data = self.to_one_hot(self.test_data[view_id])
                        data_fea = datanet(test_data)
                        pred = data_fea.view([data_fea.shape[0], -1]).mm(W)
                        pred = F.softmax(pred, dim=1)
                        tmp = pred * test_labs
                        tmp = torch.sum(tmp, dim=1)
                        tmp = torch.mean(tmp)
                        print('view_id: %d,epoch: %d acc_test:%.4f'%(view_id, epoch+1, tmp))
            print('best_epoch:', best_epoch)
            if best_acc > max_acc:
                max_acc = best_acc
                best_W = bW
                print('view_id: ', view_id)
            sio.savemat('PriorW/W_' + str(self.output_shape) + 'X' + str(self.num_classes) + self.datasets + str(view_id)  + '.mat', {'W': bW.cpu().detach().numpy()})
        sio.savemat('PriorW/W_' + str(self.output_shape) + 'X' + str(self.num_classes) + self.datasets  + '.mat', {'W': best_W.cpu().detach().numpy()})
        return best_W
    # def load_prior(self, view_id):
    #     W = sio.loadmat('PriorW/W_' + str(self.output_shape) + 'X' + str(self.num_classes) + self.datasets +str(view_id) + '.mat')['W']
    #     W = torch.tensor(W, requires_grad=False).cuda()
    #     return W