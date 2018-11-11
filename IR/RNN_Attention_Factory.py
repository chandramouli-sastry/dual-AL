from __future__ import division,print_function

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import random
from torch.optim import Adam
import numpy as np
torch.set_default_dtype(torch.float64)

def xavier(shape):
    w = torch.Tensor(*shape)
    return torch.nn.init.xavier_normal_(w)


class GRU:
    def __init__(self, input_size, hidden_units, activation, derivative_needed = False):
        self.gru = torch.nn.GRUCell(input_size = input_size,hidden_size=hidden_units)
        self.hidden_units = hidden_units
        self.activation = F.leaky_relu
        self.derivative_needed = derivative_needed

        self.gate_kernel = Variable(0.01*xavier(shape=[input_size + hidden_units, 2 * hidden_units]), requires_grad=True)
        self.gate_bias = Variable(torch.zeros(2 * hidden_units), requires_grad=True)
        self.candidate_kernel = Variable(0.01*xavier(shape=[input_size + hidden_units, hidden_units]), requires_grad=True)
        self.candidate_bias = Variable(torch.zeros(hidden_units), requires_grad=True)

        self.variables = [self.gate_bias,self.gate_kernel,self.candidate_bias,self.candidate_kernel] + list(self.gru.parameters())

    def parameters(self):
        return self.variables

    def zero_state(self, batch_size):
        return torch.zeros(batch_size, self.hidden_units)

    def one_step_forward(self, input, state, for_out):

        gate_inputs = torch.matmul(torch.cat([input, state], 1), self.gate_kernel)
        gate_inputs = gate_inputs + self.gate_bias

        value = F.sigmoid(gate_inputs)
        r, u = torch.chunk(value, chunks=2, dim=1)

        r_state = r * state

        candidate = torch.matmul(torch.cat([input, r_state], 1), self.candidate_kernel)
        candidate = candidate + self.candidate_bias

        c = self.activation(candidate)
        new_h = u * state + (1 - u) * c

        if for_out or not(self.derivative_needed):
            return new_h,None

        u = u.data
        c = c.data
        state = state.data
        value = value.data

        start = torch.ones([1,self.hidden_units])
        du = torch.mul(start, (state))
        # dstate = torch.mul(start, (u))
        du = du - torch.mul(start, (c))
        dc = torch.mul(start, (1 - u))  # 50 x 50
        if self.activation == F.leaky_relu:
            dcandidate = (dc * torch.where(candidate > 0, torch.ones_like(candidate),
                                    0.01 * torch.ones_like(candidate)))  # 50x50 * 1x50
        else:
            dcandidate = (dc * (1 - c ** 2))
        dinputs_rstate = torch.matmul(dcandidate, torch.transpose(self.candidate_kernel,dim0=1,dim1=0))  # 50 x 350
        dinputs = dinputs_rstate[:, :100]
        drstate = dinputs_rstate[:, 100:]
        # dstate = dstate + r * drstate
        dr = state * drstate
        dru = torch.cat([dr, du], dim=1)
        dgateinputs = dru * (value * (1 - value))
        dinputs_state = torch.matmul(dgateinputs, torch.transpose(self.gate_kernel,dim0=1,dim1=0))
        dinputs = dinputs + dinputs_state[:, :100]
        # dstate = dstate + dinputs_state[:, 300:]
        return new_h,dinputs

    def forward(self, inputs, for_out, reverse=False):
        states = []
        grads = []
        prev_state = self.zero_state(batch_size=inputs.shape[0])

        if for_out:
            mask1 = Variable(torch.bernoulli(inputs[:, 0, :].data.new(inputs[:, 0, :].data.size()).fill_(1)))
            mask2 = Variable(torch.bernoulli(prev_state.data.new(prev_state.data.size()).fill_(1)))
            mask3 = Variable(torch.bernoulli(prev_state.data.new(prev_state.data.size()).fill_(1)))
        else:
            mask1 = Variable(torch.bernoulli(inputs[:, 0, :].data.new(inputs[:, 0, :].data.size()).fill_(0.7)))/0.7
            mask2 = Variable(torch.bernoulli(prev_state.data.new(prev_state.data.size()).fill_(0.7)))/0.7
            mask3 = Variable(torch.bernoulli(prev_state.data.new(prev_state.data.size()).fill_(0.7)))/0.7

        if reverse:
            for t in range(inputs.shape[1] - 1, -1, -1):
                prev_state, grad = self.one_step_forward(inputs[:, t, :], prev_state, mask1=mask1, mask2=mask2,
                                                         for_out=for_out)
                states.append(prev_state)
                grads.append(grad)
            states = states[::-1]
            grads = grads[::-1]
        else:
            for t in range(inputs.shape[1]):
                prev_state, grad = self.one_step_forward(inputs[:, t, :]*mask1, prev_state*mask2, for_out=for_out)
                states.append(prev_state)
                grads.append(grad)
        if for_out:
            return torch.stack(states).transpose(dim0=1, dim1=0), None

        if not(self.derivative_needed):
            states = [state*mask3 for state in states]
            return torch.stack(states).transpose(dim0=1, dim1=0),None

        return torch.stack(states).transpose(dim0=1, dim1=0), torch.stack(grads).transpose(dim0=1, dim1=0)



def softmax_padding(acts,padding,dim):
    exps = torch.exp(acts-acts.max(dim=dim,keepdim=True)[0].detach()) * padding
    return exps / exps.sum(dim=dim, keepdim=True)

def look_for_90(attentions,user_terms,att_max,regularize=False):
    between_01 = attentions
    num_seen = (between_01 * user_terms).sum()
    t = between_01 * user_terms
    if regularize:
        return (num_seen - 0.5)**2 + (F.relu((between_01-att_max))).sum()
    else:
        return (num_seen - 0.5)**2

class RNN_Attention_Factory:
    def get_class(self,use_attention=False,att_max=0.8):
        class RNN_Prior:
            def initialize_variables(self):
                hidden_size = 50
                context_size = 20

                self.bidi = GRU(input_size=self.vector_size,
                               hidden_units=hidden_size,
                               activation=F.leaky_relu,
                               derivative_needed=use_attention)

                self.attention_weight = Variable(xavier(shape=[hidden_size, context_size]), requires_grad=True)
                self.context_vector = Variable(xavier(shape=[context_size, 1]), requires_grad=True)

                self.relevance_weight = Variable(xavier(shape=[hidden_size, 2]), requires_grad=True)
                self.relevance_bias = Variable(xavier(shape=[1, 2]), requires_grad=True)
                self.variables = [self.relevance_weight,
                                  self.relevance_bias,
                                  self.attention_weight,
                                  self.context_vector] + list(self.bidi.parameters())
                self.opt = Adam(self.variables,amsgrad=True)


            def __init__(self,word_vector_size):
                self.vector_size = word_vector_size
                self.initialize_variables()
                self.training = False

                def get_out(inputs,padding,only_pre_max=False,for_output=False,output=None):
                    inputs = Variable(inputs,requires_grad=True)
                    hidden_rel,grad_input_proxy = self.bidi.forward(inputs,for_output) # None x None x (fw_size + bw_size)
                    reg_hid = (hidden_rel**2).sum()

                    pre_context = F.tanh(torch.matmul(hidden_rel,self.attention_weight)) # None x None x context_size
                    pre_attention = torch.squeeze(torch.matmul(pre_context,self.context_vector),dim=2)
                    attention_unnormalized = torch.exp(pre_attention) * padding
                    attention = attention_unnormalized / attention_unnormalized.sum(dim=1, keepdim=True) # None x None x 1

                    vec = torch.mul(torch.unsqueeze(attention,dim=2),hidden_rel).sum(dim=1)
                    pre_max_real = torch.matmul(vec, self.relevance_weight) + self.relevance_bias

                    if only_pre_max:
                        return pre_max_real

                    out = F.softmax(pre_max_real)

                    if for_output:
                        return out,attention,attention,attention,None

                    if use_attention:
                        grad_attention_proxy = (grad_input_proxy * inputs).sum(dim=2)
                        grad_attention_proxy = softmax_padding(grad_attention_proxy, padding, dim=1)
                    else:
                        grad_attention_proxy = attention

                    return out,grad_attention_proxy,grad_input_proxy,attention,pre_max_real,reg_hid

                def get_loss(inputs, padding, user_terms, expected):
                    out,grad_attention_proxy,_,attention,pre_max_real,reg_hid = self.get_out(inputs,padding,output=expected)

                    prediction_error = -(expected*torch.log(out[:,1]+10**-5) + (1-expected)*torch.log(out[:,0]+10**-5)).sum()
                    self.reg_hid = reg_hid

                    attention_error = 0
                    if use_attention:
                        wa = {}
                        for idx, w in enumerate(self.doc.words):
                            wa[w] = wa.get(w, 0) + grad_attention_proxy[0][idx]

                        attention_error = attention_error + expected*look_for_90(grad_attention_proxy,user_terms,regularize=True,att_max=att_max)
                        for w in wa:
                            attention_error += F.relu((wa[w] - 0.4))

                    if use_attention:
                        wa = {}
                        for idx,w in enumerate(self.doc.words):
                            wa[w] = wa.get(w,0)+attention[0][idx]
                        attention_error += expected*(look_for_90(attention, user_terms,regularize=True,att_max=att_max) + F.relu((0.01-attention)*user_terms).sum())

                        for w in wa:
                            attention_error += F.relu(expected*(wa[w] - 0.4) + (1-expected)*(wa[w] - 0.1))
                    return prediction_error + attention_error

                self.get_out = get_out
                self.get_loss = get_loss


            def get_reg(self):
                reg = 0.0001 * ((self.attention_weight ** 2).sum()
                                + (self.context_vector ** 2).sum()
                                + (self.relevance_weight ** 2).sum())
                for par in self.bidi.parameters():
                    reg += 0.0001 * ((par ** 2).sum())
                return reg + 10**-4*self.reg_hid

            def get_feed_dict_multiple(self,docs):
                if self.training:
                    assert len(docs) == 1
                    doc = docs[0]
                    set_words = set(doc.words)
                    chosen_words = random.sample(set_words,int(1*len(set_words)))
                    doc.chosen_vectors = [(doc.vectors[i] if doc.words[i] in chosen_words else 0*doc.vectors[i]) for i in range(len(doc.vectors))]
                    maximum = max([len(doc.words) for doc in docs])
                    return {    "vectors": torch.DoubleTensor(np.array([doc.chosen_vectors[:maximum]+[[0]*(self.vector_size)]*(maximum-len(doc.vectors[:maximum])) for doc in docs],dtype=np.float64).reshape([-1,maximum,self.vector_size])),
                            "output": torch.DoubleTensor(np.array([[doc.class_*1] for doc in docs],dtype=np.float64)),
                            "user_terms": torch.DoubleTensor(np.array(
                                [doc.ut2[:maximum] + [0] * (maximum - len(doc.ut2[:maximum])) for doc in
                                 docs],dtype=np.float64).reshape([-1, maximum])),
                            "padding": torch.DoubleTensor(np.array([[1]*len(doc.vectors[:maximum]) + [0]*(maximum-len(doc.vectors[:maximum])) for doc in docs],dtype=np.float64).reshape([-1,maximum])) }
                maximum = max([len(doc.words) for doc in docs])
                return {"vectors": torch.DoubleTensor(np.array([doc.vectors[:maximum] + [[0] * (self.vector_size)] * (maximum - len(doc.vectors[:maximum])) for doc in docs], dtype=np.float64).reshape([-1, maximum, self.vector_size])),
                        "output": torch.DoubleTensor(np.array([[doc.class_ * 1] for doc in docs], dtype=np.float64)),
                        "user_terms": torch.DoubleTensor(np.array([doc.user_terms[:maximum] + [0] * (maximum - len(doc.user_terms[:maximum])) for doc in docs], dtype=np.float64).reshape([-1, maximum])),
                        "padding": torch.DoubleTensor(np.array([[1] * len(doc.vectors[:maximum]) + [0] * (maximum - len(doc.vectors[:maximum])) for doc in docs], dtype=np.float64).reshape([-1, maximum]))}

            def train(self, docs):
                self.initialize_variables()
                self.training = True
                print("====")
                epochs = 150
                opt = self.opt

                random.shuffle(docs)
                last_10 = [100]*10
                last_5 = [100] * 5
                for epoch in range(epochs):
                    total_error = 0
                    all_nan = False
                    for doc in docs:
                        self.doc = doc
                        fd = self.get_feed_dict_multiple([doc])

                        loss = self.get_loss(inputs=fd["vectors"],
                                             user_terms=fd["user_terms"],
                                             expected=fd["output"],
                                           padding=fd["padding"])
                        if np.isnan(loss.detach()).any():
                            self.train(docs)
                            return

                        opt.zero_grad()
                        (loss+self.get_reg()).backward()
                        torch.nn.utils.clip_grad_norm(self.variables,max_norm=10)

                        update = True
                        for var in self.variables:
                            if var.grad is not None and np.isnan(var.grad.numpy()).all():
                                # print("CHECK")
                                update = False
                        all_nan = all_nan or update
                        if update:
                            opt.step()
                        opt.zero_grad()
                        total_error += loss.data

                    total_error = total_error / len(docs)

                    last_10.pop(0)
                    last_10.append(total_error)
                    if max(last_10)<0.01:
                        print("")
                        print("breaking")
                        break
                    print(".",end="")

                print(total_error)
                self.training = False

            def run(self,docs):
                for doc_s in [docs[i:i + 10] for i in range(0, len(docs), 10)]:
                    fd = self.get_feed_dict_multiple(doc_s)

                    rel, pos_attention, neg_attention, attention, _ = self.get_out(inputs=fd["vectors"],
                                                                                padding=fd["padding"],for_output=True)
                    rel = rel.data
                    pos_attention = pos_attention.data
                    neg_attention = neg_attention.data
                    attention = attention.data
                    self.opt.zero_grad()
                    for ind,doc in enumerate(doc_s):
                        d = {
                            "rel": rel[ind][1],
                            "attentions": attention[ind][:len(doc_s[ind].user_terms)].numpy(),
                            "pos_att": pos_attention[ind][:len(doc_s[ind].user_terms)].numpy(),
                            "neg_att": neg_attention[ind][:len(doc_s[ind].user_terms)].numpy()
                        }
                        doc.pred_class = 0 if d["rel"]<0.5 else 1
                        doc.parameters = d
        return RNN_Prior