from qutip import *
import numpy as np
from itertools import product
import random
import matplotlib.pyplot as plt
import simple_exact_diagonalization_routines as spd
import pandas as pd

class pauli_transfer_matrix():
    def __init__(self, ntls):
        self.ntls = ntls
        self.pauli_index = [0,1,2,3]
        self.pauli_index_ls = [self.pauli_index] * self.ntls
        self.u_perm = list(product(*self.pauli_index_ls))
        self.pauli_ls = [qeye(2), sigmax(), sigmay(), sigmaz()]
        self.identity = tensor([qeye(2) for i in range(self.ntls)])
        self.perm_pairs = self.get_perm_pairs()

    @staticmethod
    def delta(a, b):
        if a == b or a == 0 or b == 0:
            return 1
        else:
            return 0

    @staticmethod
    def pauli_prod_single(p1, p2):
        results = {
            (1,2): (1j, 3),
            (1,3): (-1j, 2),
            (2,1): (-1j, 3),
            (2,3): (1j, 1),
            (3,1): (1j, 2),
            (3,2): (-1j, 1)
        }
        if p1 == p2:
            return (1, 0)
        elif p1 == 0:
            return (1, p2)
        elif p2 == 0:
            return (1, p1)
        
        else:
            return results[(p1,p2)]

    def get_perm_pairs(self):
        pair_index_ls = [self.u_perm] * 2
        # For ntls = 2, len(pair_perm)=16^ntls=256
        perm_pair_uw = list(product(*pair_index_ls))
        return perm_pair_uw
    
    def perm_vec(self):
        # n^4 possible vectors in {0,1,2,3}^n
        return list(product(*self.pauli_index_ls))
    
    def pauli_gen(self, index):
        # Returns qutip operator with index
        return tensor([self.pauli_ls[index[i]] for i in range(len(index))])
    
    def pauli_qutip_ops(self):
        # Returns dictinary of qutip operators, call by pauli_qutip_ops[(index)]
        return {index: self.pauli_gen(index) for index in self.u_perm}

    def pauli_commute(self, u, w):
        # u, w: tuples, e.g. (0,1,2,2,3) for n=5
        # Determine 0 or 2
        sgn = 1
        for i in range(len(u)):
            sgn *= (-1) ** self.delta(u[i], w[i])
        if sgn == (-1) ** self.ntls:
            return (0, 0)
        else:
            (fact, op) = self.pauli_prod(u, w)
            return (2 * fact, op)
    
    def pauli_prod(self, u, w):
        new = np.zeros(len(u))
        fact_new = 1
        for i in range(len(u)):
            (fact, res) = self.pauli_prod_single(u[i], w[i])
            fact_new *= fact
            new[i] = res
        return (fact_new, tuple(new.astype(int)))  
    # Unused
    def f(self, u, D=None):
        f_ls = []
        if D == None:
            D = self.D
        for D_loc in D:
            (df, d) = D_loc
            (fact, v) = self.pauli_commute(u,d)
            f_ls.append((df*fact, v))
        return f_ls
        circuit_op = tensor(qeye(2), self.identity)
        for i in range(N):
            V_fj = self.V_fj(self.sample_vv(), self.sample_uw())
            circuit_op *= V_fj * np.exp(-1j* H * self.beta / N) * V_fj.dag()
        return circuit_op
        
class commutator_type_dynamics(pauli_transfer_matrix):
    def __init__(self, D):
        self.D = D
        self.ntls = len(self.D[0][1])
        super().__init__(self.ntls)
        self.please_be_verbose = False
        self.please_be_exhaustively_verbose = False
        self.J = None
        self.local_restriction = False
        
        self.beta = None
        self.gamma_uw_dict = None
        self.prob_uw = None
    
    # Here comments dictate whether function is universal for murao
    # routine or specific to the commutator type. For future changes
    # Such as option in Murao class given any function f()
    
    def run_PTM(self, D=None):
        if D == None:
            D = self.D
        gamma_comp = self.gamma_distribution(self.D)
        self.beta = gamma_comp['beta']
        self.gamma_uw_dict = gamma_comp['gamma_uw_dict']
        self.prob_uw = gamma_comp['prob_uw_dict']
        return {'beta': self.beta, 'gamma_uw_dict': self.gamma_uw_dict, 'prob_uw': self.prob_uw}
    
    # commute
    def gamma_u_w(self, u, w, D=None, J=None, local_restriction=False):
        # D = [(df, d)], df constant factor, d pauli vector (tuple)
        # [u,d]=v return 0/2 if w=v, else 0
        # d be an index for a Pauli matrix in diagonal D
        if D == None:
            D = self.D
        if J == None:
            J = self.J
        if local_restriction == None:
            local_restriction = self.local_restriction
            
        if self.please_be_exhaustively_verbose:
            print('Received D in gamma_uw',D)
            
        if local_restriction == True and J != None and u not in J:
            return 0
        
        for D_loc in D:
            (df, d) = D_loc
            (fact, v) = self.pauli_commute(u,d)
            # TODO count number of 0's in u and w truncte (return 0) if too little
            # QUESTION u AND w or u OR w or what - experiment
            if v != w:
                continue
            else:
                return df * fact
        return 0
    # commute
    def gamma_distribution(self, D=None, show_df=False, show_zeros=False):
        if D == None:
            D = self.D
        if self.please_be_exhaustively_verbose:
            print('Received D in gamma_distribution',D)
        perm_pair_uw = self.perm_pairs
        gamma = [self.gamma_u_w(pair_uw[0], pair_uw[1], D)
                 for pair_uw in perm_pair_uw]
        gamma_uw = {k: v for k,v in zip(perm_pair_uw, gamma)}
        beta = 2*np.sum(np.abs(gamma))
        prob_uw_dict = {k: v for k,v in zip(perm_pair_uw, 2*np.abs(gamma)/beta)}
        if show_df:
            self.show_prob_uw_distribution(prob_uw_dict, show_zeros=show_zeros)
        return {'beta':beta, 'gamma_uw_dict': gamma_uw, 'prob_uw_dict': prob_uw_dict}
    # murao
    def show_prob_uw_distribution(self, prob_uw=None, show_zeros=False):
        if prob_uw == None:
            prob_uw = self.prob_uw
        gamma_uw_df = pd.DataFrame([(k,val) for k,val in prob_uw.items()], columns=['Index','Gamma'])
        df_sort = gamma_uw_df.sort_values('Gamma')
        if show_zeros:
            display(gamma_uw_df)
        else:
            display(df_sort[df_sort['Gamma']!=0j])
    # commute
    def sample_uw(self, D=None, multi_sample=False):
        if D == None:
            perm_pair_uw = self.perm_pairs
            prob_uw = self.prob_uw
            gamma_uw = self.gamma_uw_dict
        else:
            gamma_cal = self.gamma_distribution(D)
            perm_pair_uw = self.perm_pairs
            gamma_uw = gamma_cal['gamma_uw_dict']
            prob_uw = gamma_cal['prob_uw_dict']
        if multi_sample:
            n_samples = 500000
            samples = np.random.choice(len(perm_pair_uw),n_samples, p=np.abs(list(prob_uw.values())))
            sample_stat = np.unique(samples, return_counts=True)
            unique_ind = list(sample_stat[0].astype(int))
            ind_occur = sample_stat[1]
            x_labels = [f'{perm_pair_uw[i]}' for i in unique_ind]
            freq = list(ind_occur/n_samples)
            y_ax = pd.Series(freq)
            plt.figure(figsize=(14,8))
            fig = y_ax.plot(kind='bar')
            fig.set_title(f'Sample frequencies out of {n_samples}')
            fig.set_ylabel('Frequency')
            fig.set_xlabel('Pair index')
            fig.set_xticklabels(unique_ind)
            rects = fig.patches
            for rect, label in zip(rects, x_labels):
                height = rect.get_height()
                fig.text(
                    rect.get_x() + rect.get_width() / 2, height , label, ha="center", va="bottom"
                )
            plt.show()
            plt.show()
            all_possible_uw = [(perm_pair_uw[i], prob_uw[perm_pair_uw[i]]) for i in unique_ind]
            return all_possible_uw
        else:
            sample = np.random.choice(len(perm_pair_uw), p=np.abs(list(prob_uw.values())))
            return (perm_pair_uw[sample][0], perm_pair_uw[sample][1], gamma_uw[perm_pair_uw[sample]])
    # murao
    def sample_vv(self):
        (v, vp) = (self.u_perm[random.randint(0,len(self.u_perm)-1)], 
           self.u_perm[random.randint(0,len(self.u_perm)-1)])
        return (v, vp)
    # murao
    def controlled_U(self, U, ntls=None):
        if ntls == None:
            ntls = self.ntls
        # |0><0|I + |1><1|U
        return tensor(ket("0")*bra("0"),self.identity) + tensor(ket("1")*bra("1"),U)
    # murao    
    def V_fj(self, sample_vv, sample_uw):
        (v, vp) = sample_vv
        (u, w, gamma_uw) = sample_uw
        hadamard_c = tensor(hadamard_transform(1), self.identity)
        sf = int((1 - np.real(np.sign(gamma_uw)))/2)
        if self.please_be_verbose:
            print(f'V_fj using (v, vp) = ({v},{vp}), (u, w) = ({u},{w}), gamma_uw = {gamma_uw}')
            print(f'Sign sf = {sf}')
        op = tensor(sigmax()**sf, self.identity) * hadamard_c * self.controlled_U(self.pauli_gen(w))\
             * tensor(qeye(2),self.pauli_gen(vp)) * self.controlled_U(self.pauli_gen(u))\
             * hadamard_c * self.controlled_U(self.pauli_gen(v))
        return op
    # murao
    def complete_circuit(self, N, H):
        circuit_op = tensor(qeye(2), self.identity)
        for i in range(N):
            V_fj = self.V_fj(self.sample_vv(), self.sample_uw())
            circuit_op *= V_fj * np.exp(-1j* H * self.beta / N) * V_fj.dag()
        return circuit_op
        