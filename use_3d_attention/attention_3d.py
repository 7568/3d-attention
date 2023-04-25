import pandas as pd

from basemodel_torch import BaseModelTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from torch import einsum
from einops import rearrange
from sklearn.metrics import mean_squared_error,mean_absolute_error
from lib.models.pretrainmodel import ATTENTION_3D as ATTENTION_3Dodel
from lib.data_openml import DataSetCatCon
from lib.augmentations import embed_data_mask, mixup_data, add_noise
from tqdm import tqdm
from torchmetrics.classification import BinaryF1Score
import os
import sys

sys.path.append(os.path.dirname("../../*"))
sys.path.append(os.path.dirname("../*"))
from library import util as lib_util
torch.multiprocessing.set_sharing_strategy('file_system')

'''
    batch内数据为一天内的数据
'''


class SELF_ATTENTION(BaseModelTorch):

    def __init__(self, params, args):
        super().__init__(params, args)
        if args.cat_idx:
            num_idx = list(set(range(args.num_features)) - set(args.cat_idx))
            # Appending 1 for CLS token, this is later used to generate embeddings.
            # cat_dims = np.append(np.array([1]), np.array(args.cat_dims)).astype(int)
            cat_dims = np.array(args.cat_dims).astype(int)
        else:
            num_idx = list(range(args.num_features))
            cat_dims = np.array([])

        # Decreasing some hyperparameter to cope with memory issues
        # dim = self.params["dim"] if args.num_features < 50 else 8
        # self.batch_size = self.args.batch_size if args.num_features < 50 else 64


        # print("Using dim %d and batch size %d" % (dim, self.batch_size))
        self.cat_dims = cat_dims
        if args.cat_idx is None:
            args.cat_idx=[]
        self.model = ATTENTION_3Dodel(
            categories=tuple(cat_dims),
            num_continuous=len(num_idx),
            dim=self.params["dim"],
            dim_out=2,
            depth=self.params["depth"],  # 6 # 8
            attn_dropout=self.params["dropout"],  # 0.1
            ff_dropout=self.params["dropout"],  # 0.1
            mlp_hidden_mults=(4, 2),
            cont_embeddings="MLP",
            attentiontype="colrow",
            final_mlp_style="sep",
            y_dim=args.num_classes, device=self.device,
            each_day_cat_feature_num=len(args.cat_idx)//5,each_day_feature_num=args.num_features//5
        )

        if self.args.data_parallel:
            print(f'self.args.data_parallel :{self.args.data_parallel}')
            self.model.transformer = nn.DataParallel(self.model.transformer, device_ids=self.args.gpu_ids)
            self.model.mlpfory = nn.DataParallel(self.model.mlpfory, device_ids=self.args.gpu_ids)

    def fit(self, X, y, X_val=None, y_val=None, training_trading_dates=None, validation_trading_dates=None):

        if self.args.objective == 'binary':
            criterion = nn.BCEWithLogitsLoss().to(self.device)
            # criterion = nn.BCELoss().to(self.device)
        elif self.args.objective == 'classification':
            criterion = nn.CrossEntropyLoss().to(self.device)
        elif self.args.objective == 'binary_f1':
            criterion = BinaryF1Score().to(self.device)
        else:
            criterion = nn.MSELoss().to(self.device)
        # torch_f1_score = BinaryF1Score().to(self.device)
        self.model.to(self.device)
        print(f'self.learning_rate : {self.args.learning_rate}')
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

        # ATTENTION_3D wants it like this...
        X_train = {'data': X}
        y_train = {'data': y.reshape(-1, 1)}
        # X_val = {'data': X_val, 'mask': np.ones_like(X_val)}
        # y_val = {'data': y_val.reshape(-1, 1)}

        train_ds = DataSetCatCon(X_train, y_train, self.args.cat_idx, self.args.objective, trading_dates=training_trading_dates)
        trainloader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)

        min_val_loss_idx = 0
        min_mse = float('inf')
        loss_history = []
        val_loss_history = []

        for epoch in range(self.args.epochs):
            self.model.train()
            loss_history = []
            rmses=[]
            for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):

                x_categ, x_cont, y_gts = data
                x_categ = x_categ.squeeze(0)
                x_cont = x_cont.squeeze(0)
                y_gts = y_gts.squeeze(0)

                x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
                _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont,self.model)
                reps = self.model.transformer(x_cont_enc,x_categ_enc)
                y_reps = reps[:, 0, :]
                y_outs = self.model.mlpfory(y_reps)

                # if self.args.objective == "binary":
                #     soft_max_y_outs = torch.sigmoid(y_outs)
                # elif self.args.objective == "classification":
                #     soft_max_y_outs = F.softmax(y_outs, dim=1)
                # else:
                #     soft_max_y_outs = y_outs

                if self.args.objective == "regression":
                    y_gts = y_gts.to(self.device)
                elif self.args.objective == "classification":
                    y_gts = y_gts.to(self.device).squeeze()
                else:
                    y_gts = y_gts.to(self.device).float()
                loss = criterion(y_outs, y_gts)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_history.append(loss.item())
                rmse = mean_squared_error(y_gts.detach().cpu(), y_outs.detach().cpu(), squared=False)
                rmses.append(rmse)
                # print("Loss", loss.item())

            print(f'epoche " {epoch} , average loss : {np.array(loss_history).mean()} , average rmses : {np.array(rmses).mean()}')
            if epoch<10:
                continue
            # train_mse = self.predict_helper(X_train['data'],training_trading_dates,y_train['data'],tag='training',need_reload_model=False)
            # print(f'train mse : {train_mse}')
            mse = self.predict_helper(X_val,validation_trading_dates,y_val,tag='validation',need_reload_model=False)
            print(f'validation mse : {mse}')

            if mse < min_mse:
                min_mse = mse
                min_val_loss_idx = epoch

                # Save the currently best model
                self.save_model(filename_extension=f"{self.args.model_name}_{self.args.learning_rate}_best", directory="tmp")

            if min_val_loss_idx + self.args.early_stopping_rounds < epoch:
                print("Validation loss has not improved for %d steps!" % self.args.early_stopping_rounds)
                print("Early stopping applies.")
                break

            self.predict_helper(self.testing_x, self.testing_trading_dates, self.testing_y,need_reload_model=False)

        # self.load_model(filename_extension="best", directory="tmp")
        return loss_history, val_loss_history


    def set_testing(self, x, y, testing_trading_dates=None):
        self.testing_x = x
        self.testing_y = y.reshape(-1, 1)
        self.testing_trading_dates = testing_trading_dates


    def predict_and_anasys_result(self,data_X, _trading_dates=None,data_y = None,tag='testing',need_reload_model=True,max_day=210):
        _data_X = {'data': data_X}
        if data_y is None:
            _data_y = {'data': self.testing_y}
        else:
            _data_y = {'data': data_y.reshape(-1, 1)}
        _ds = DataSetCatCon(_data_X, _data_y, self.args.cat_idx, self.args.objective, trading_dates=_trading_dates)
        dataloader = DataLoader(_ds, batch_size=1, shuffle=False, num_workers=1)
        print(f'need_reload_model : {need_reload_model}')
        if need_reload_model:
            self.load_model(filename_extension=f"{self.args.model_name}_{self.args.learning_rate}_best",
                            directory="tmp")
            # filename='/home/liyu/git/3d-attention/use_3d_attention/output/3d-attention/h_sh_300_option/tmp/m_3d-attention_0.0001_best.pt'
            # state_dict = torch.load(filename, map_location=torch.device(self.device))
            # self.model.load_state_dict(state_dict)
            self.model.to(self.device)
        predictions = []
        real_testing_y = []
        mses = []
        x_conts=[]
        self.model.eval()
        with torch.no_grad():

            for data in tqdm(dataloader, total=len(dataloader)):
                x_categ, x_cont, y_gts = data
                x_categ = x_categ.squeeze(0)
                x_cont = x_cont.squeeze(0)
                x_conts.append(x_cont)
                y_gts = y_gts.squeeze(0)

                x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
                _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, self.model)
                reps = self.model.transformer(x_cont_enc, x_categ_enc)
                y_reps = reps[:, 0, :]
                y_outs = self.model.mlpfory(y_reps)

                y_outs = y_outs.detach().cpu()

                real_testing_y.append(y_gts)
                predictions.append(y_outs)
                mse = mean_squared_error(y_gts, y_outs)
                mses.append(mse)
        print(np.array(mses).mean())
        self._get_score(real_testing_y, predictions)
        x_conts = np.concatenate(x_conts)
        print(x_conts.shape)
        underlying_scrt_close=x_conts[:,2]
        strike_price=x_conts[:,4]
        moneyness = underlying_scrt_close/strike_price
        remaining_term = np.round((x_conts[:,5] * 365))
        testing_df=pd.DataFrame()
        testing_df.loc[:, 'moneyness'] = moneyness
        testing_df.loc[:, 'RemainingTerm'] = remaining_term
        testing_df.loc[:, 'y_test_true'] = np.concatenate(real_testing_y)
        testing_df.loc[:, 'y_test_hat'] = np.concatenate(predictions)
        table_name = f'{self.args.dataset}_moneyness_maturity'
        lib_util.analysis_by_moneyness_maturity(testing_df, max_day, table_name)

    def predict_helper(self, data_X, _trading_dates=None,data_y = None,tag='testing',need_reload_model=True):
        _data_X = {'data': data_X}
        if data_y is None:
            _data_y = {'data': self.testing_y}
        else:
            _data_y = {'data': data_y.reshape(-1, 1)}
        _ds = DataSetCatCon(_data_X, _data_y, self.args.cat_idx, self.args.objective, trading_dates=_trading_dates)
        dataloader = DataLoader(_ds, batch_size=1, shuffle=False, num_workers=4)
        print(f'need_reload_model : {need_reload_model}')
        if need_reload_model:
            self.load_model(filename_extension=f"{self.args.model_name}_{self.args.learning_rate}_best",
                            directory="tmp")
            self.model.to(self.device)
        predictions = []
        real_testing_y = []
        mses=[]
        self.model.eval()
        with torch.no_grad():

            for data in tqdm(dataloader, total=len(dataloader)):
                x_categ, x_cont, y_gts = data
                x_categ = x_categ.squeeze(0)
                x_cont = x_cont.squeeze(0)
                y_gts = y_gts.squeeze(0)

                x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
                _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, self.model)
                reps = self.model.transformer(x_cont_enc, x_categ_enc)
                y_reps = reps[:, 0, :]
                y_outs = self.model.mlpfory(y_reps)

                y_outs = y_outs.detach().cpu()

                real_testing_y.append(y_gts)
                predictions.append(y_outs)
                mse = mean_squared_error(y_gts, y_outs)
                mses.append(mse)
        print(f'np.array(mses).mean() : {np.array(mses).mean()}')

        mse = self._get_score(real_testing_y, predictions)

        print(f'mse in {tag} : {mse}')

        if tag=='validation' or tag=='training':
            return mse
        else:
            return np.concatenate(predictions),np.concatenate(real_testing_y)

    def _get_score(self, y_true, y_prediction):
        y_true = np.concatenate(y_true)
        y_prediction = np.concatenate(y_prediction)

        # print(np.array(y_prediction))
        # print(np.array(y_true))
        print(y_prediction.shape)
        print(y_true.shape)
        mse = mean_squared_error(y_true, y_prediction)
        rmse = mean_squared_error(y_true, y_prediction, squared=False)
        mae = mean_absolute_error(y_true, y_prediction)
        print(f'mse : {mse} , rmse : {rmse} , mae : {mae}')
        return mse
