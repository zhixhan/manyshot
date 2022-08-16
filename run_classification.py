import argparse
from logging import raiseExceptions
from select import select
from tabnanny import verbose
from data_utils import loading_dataset
from utils import *
from itertools import permutations
import random
from copy import deepcopy
import warnings 

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_random_state
from sklearn.exceptions import ConvergenceWarning
from munkres import Munkres
from matplotlib import pyplot as plt
import seaborn as sns

class GaussianMixturewTarget(GaussianMixture):

    def __init__(
        self,
        n_components=1,
        *,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weights_init=None,
        means_init=None,
        precisions_init=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
        
    ):
        super().__init__(
            n_components=n_components,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval,
            covariance_type = covariance_type,
            weights_init = weights_init,
            means_init = means_init,
            precisions_init = precisions_init
        )
    def fit_predict(self, X, y=None):
        
        X = self._validate_data(X, dtype=[np.float64, np.float32], ensure_min_samples=2)
        if X.shape[0] < self.n_components:
            raise ValueError(
                "Expected n_samples >= n_components "
                f"but got n_components = {self.n_components}, "
                f"n_samples = {X.shape[0]}"
            )
        self._check_initial_parameters(X)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.inf
        self.converged_ = False
       
        random_state = check_random_state(self.random_state)
        best_init = -1
        n_samples, _ = X.shape
        for init in range(n_init):
            
            self._print_verbose_msg_init_beg(init)
            if do_init:
                self._initialize_parameters(X, random_state)
           
            lower_bound = -np.inf if do_init else self.lower_bound_

            for n_iter in range(1, self.max_iter + 1):
                prev_lower_bound = lower_bound

                log_prob_norm, log_resp = self._e_step(X)
                self._m_step(X, log_resp)
                lower_bound = self._compute_lower_bound(log_resp, log_prob_norm)

                change = lower_bound - prev_lower_bound
                self._print_verbose_msg_iter_end(n_iter, change)

                if abs(change) < self.tol:
                    self.converged_ = True
                    break
            self._print_verbose_msg_init_end(lower_bound)
            
            means = self._get_parameters()[1]
          
            m = Munkres()
            indexes = m.compute(-means)
            profits = np.sum([means[index[0]][index[1]] for index in indexes])

            if profits > max_lower_bound or max_lower_bound == -np.inf:
                max_lower_bound = profits
                best_params = self._get_parameters()
                best_n_iter = n_iter
                best_init = init
        print("best_init:",best_init)
        if not self.converged_:
            warnings.warn(
                "Initialization %d did not converge. "
                "Try different init parameters, "
                "or increase max_iter, tol "
                "or check for degenerate data." % (init + 1),
                ConvergenceWarning,
            )

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound
        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
        _, log_resp = self._e_step(X)

        return log_resp.argmax(axis=1)

def main(models, datasets, all_shots, num_seeds, start_seed, subsample_test_set, bs, method, gmm_train_estimate_scale):
    """
    Run experiment or load past results, print accuracy
    """

    default_params = {
        'subsample_test_set': subsample_test_set,
        'bs': bs,
        'method': method,
        'gmm_train_estimate_scale': gmm_train_estimate_scale
    }
    
    # list of all experiment parameters to run
    all_params = []
    for model in models:
        for dataset in datasets:
            for num_shots in all_shots:
                    for seed in range(start_seed, num_seeds+start_seed):                 
                        p = deepcopy(default_params)
                        p['model'] = model
                        p['dataset'] = dataset
                        p['seed'] = seed
                        p['num_shots'] = num_shots
                        if 'gmm' in method:
                            p['expr_name'] = f"{p['dataset']}_{p['model']}_{p['num_shots']}shot_{repr(p['subsample_test_set'])}_subsample_seed{p['seed']}_method_{p['method']}_gmm_train_estimate_scale{p['gmm_train_estimate_scale']}"
                        else:
                            p['expr_name'] = f"{p['dataset']}_{p['model']}_{p['num_shots']}shot_{repr(p['subsample_test_set'])}_subsample_seed{p['seed']}_method_{p['method']}"
                        all_params.append(p)

   
    # query the model and save the responses
    
    save_results(all_params)


def save_results(params_list, freeze_test_set=True):
    """
    Run the model and save its responses and the rest of configs into a pickle file
    """
    # set training examples order in params_list
    
    result_tree = dict()
    
    for param_index, params in enumerate(params_list):
        print("\nExperiment name:", params['expr_name'])
        ### load data
        ### sample few-shot training examples
        all_train_sentences, all_train_labels, all_test_sentences, all_test_labels = loading_dataset(params)
        print('Train/Test set scale: {tr}, {te}.'.format(tr=len(all_train_labels), te=len(all_test_labels)))
        params_check(params)
        
        np.random.seed(params['seed'])
        if params['dataset']=='amazon_polarity' and 'gpt2' in params['model']:
            # avoid to exceeding the maximum sequence length for gpt2 on amazon_polarity
            train_sentences, train_labels = random_sampling(all_train_sentences, all_train_labels, params['num_shots'], max_length=500)
        else:
            train_sentences, train_labels = random_sampling(all_train_sentences, all_train_labels, params['num_shots'])
        
        print(train_sentences, train_labels)
       

        maxlength = 950 if (params['dataset'] in ['rte','agnews','dbpedia','amazon_polarity'] and 'gpt2' in params['model']) else None
        print("maxlength-thres:",maxlength)
        if params['dataset'] in ['amazon_polarity','agnews','dbpedia']:
            params['subsample_test_set']=2000

        ### sample test set
        if params['subsample_test_set'] is None:
            test_sentences, test_labels = all_test_sentences, all_test_labels
            print(f"selecting full test set ({len(all_test_labels)} examples)")
        else:
            if freeze_test_set:
                np.random.seed(0) # always use seed 0 result if freeze
            else:
                np.random.seed(params['seed'])
            
            test_sentences, test_labels = random_sampling(all_test_sentences, all_test_labels, params['subsample_test_set'],max_length=maxlength)
            print(f"selecting {len(test_labels)} subsample of test set")

        #test_sentences, test_labels = all_train_sentences, all_train_labels
        #print(len(test_labels))
        if params['dataset']=='rte' and 'gpt2' in params['model'] and params['num_shots']>=8:
            inds = [index for index in range(len(test_sentences)) if len(test_sentences[index])<=1000]
            test_sentences = [test_sentences[index] for index in inds]
            test_labels = [test_labels[index] for index in inds]

        ### Evaluate the performance and save all results
        # obtaining model's response on test examples
        print(f"getting raw resp for {len(test_sentences)} test sentences")
        _, all_label_probs = get_model_response(params, train_sentences, train_labels, test_sentences)
       
       
        if params['method']=='calibrate':
            content_free_inputs = ["N/A", "", "[MASK]"]
            # calculate P_cf
            with torch.no_grad():
                p_cf = get_p_content_free(params, train_sentences, train_labels, content_free_inputs=content_free_inputs)
            acc = eval_accuracy(all_label_probs, test_labels, mode="diagonal_W", p_cf=p_cf)
            print(f"p_cf      : {p_cf}")

        elif params['method']=='ori':
            acc = eval_accuracy(all_label_probs, test_labels)
            
        elif  params['method']=="gmm_test_estimate":
            acc = eval_accuracy(np.log(all_label_probs), test_labels,  gmm_estimate="gmm_test_estimate")

        elif params['method']=="gmm_train_estimate":
            assert params['gmm_train_estimate_scale'] is not None
            gmm_train_estimate_scale = params['gmm_train_estimate_scale']
            
            np.random.seed(params['seed']+1)
            estimate_train_sentences, _ = random_sampling(all_train_sentences, all_train_sentences, gmm_train_estimate_scale, max_length=maxlength)
            print("Estimating GMM using {len} train sentences....".format(len=len(estimate_train_sentences)))
            
            _, all_label_probs_train = get_model_response(params, train_sentences, train_labels, estimate_train_sentences)
            all_label_probs_train = np.asarray(np.log(all_label_probs_train))
            

            num_class = all_label_probs_train.shape[1]
            
            gmm = GaussianMixturewTarget(n_components=num_class, n_init=100, random_state=0, covariance_type='full').fit(all_label_probs_train)
            print("gmm ori weight: ",gmm.weights_)
            gmm.weights_ = np.asarray([0.5] * num_class)
            print("gmm converged: ", gmm.converged_)
            
            print("Test sentences using GMM with log-probs.")
            acc = eval_accuracy(np.log(all_label_probs), test_labels,  gmm_estimate="gmm_train_estimate", gmm=gmm)
                

        accuracies = [acc]
        
        print(f"Accuracies: {accuracies}")
        

        # add to result_tree
        keys = [params['dataset'], params['model'], params['num_shots']]
        node = result_tree # root
        for k in keys:
            if not (k in node.keys()):
                node[k] = dict()
            node = node[k]
        node[params['seed']] = accuracies
        '''
        # save to file
        result_to_save = dict()
        params_to_save = deepcopy(params)
        result_to_save['params'] = params_to_save
        result_to_save['train_sentences'] = train_sentences
        result_to_save['train_labels'] = train_labels
        result_to_save['test_sentences'] = test_sentences
        result_to_save['test_labels'] = test_labels
        result_to_save['all_label_probs'] = all_label_probs
        result_to_save['accuracies'] = accuracies
        if 'prompt_func' in result_to_save['params'].keys():
            params_to_save['prompt_func'] = None
        #save_pickle(params, result_to_save)
        '''
    print_results(result_tree)

def eval_accuracy(all_label_probs, test_labels, mode=None, p_cf=None, localE=False, globalE=False, gmm_estimate=None, gmm=None):
    # evaluate the accuracy with and without contextual calibration
    num_classes = all_label_probs.shape[1]
    
    pred_list = []
    correctness_list = []
    assert len(all_label_probs) == len(test_labels)
    if gmm_estimate is None:
        if p_cf is None:
            # do not calibrate
            W = np.identity(num_classes)
            b = np.zeros([num_classes, 1])
        else:
            # calibrate
            if mode == "diagonal_W":
                W = np.linalg.inv(np.identity(num_classes) * p_cf)
                b = np.zeros([num_classes, 1])
            elif mode == "identity_W":
                W = np.identity(num_classes)
                b = -1 * np.expand_dims(p_cf, axis=-1)
            else:
                assert False

        for label_probs, true_label in zip(all_label_probs, test_labels):
            
            calibrate_label_probs = np.matmul(W, np.expand_dims(label_probs, axis=-1)) + b
            
            ans_label = np.argmax(calibrate_label_probs)
            pred_list.append(ans_label)
            if ans_label == true_label:
                correctness_list.append(1)
            else:
                correctness_list.append(0)
    
    else:
        if gmm_estimate=="gmm_test_estimate":
            probs = np.asarray(all_label_probs)
            gmm = GaussianMixturewTarget(n_components=num_classes, n_init=100, random_state=0, covariance_type='full').fit(all_label_probs)
            gmm.weights_ = np.asarray([0.5] * num_classes)
            m = Munkres()
            indexes = m.compute(1-gmm.means_)
            label_mapping = [index[1] for index in indexes]
            
            print(label_mapping)
            print(gmm.means_)
            
            gmm_predicts = gmm.predict(probs)
            gmm_predicts = [label_mapping[p] for p in gmm_predicts]
            for gmm_pred, true_label in zip(gmm_predicts, test_labels):
                pred_list.append(gmm_pred)
                if gmm_pred == true_label:
                    correctness_list.append(1)
                else:
                    correctness_list.append(0)

        elif gmm_estimate=="gmm_train_estimate":
            assert gmm is not None
            probs = np.asarray(all_label_probs)
            m = Munkres()
            indexes = m.compute(1-gmm.means_)
            label_mapping = [index[1] for index in indexes]
            print(label_mapping)
            print(gmm.means_)

            gmm_predicts = gmm.predict(probs)
            gmm_predicts = [label_mapping[p] for p in gmm_predicts]
            for gmm_pred, true_label in zip(gmm_predicts, test_labels):
                if gmm_pred == true_label:
                    correctness_list.append(1)
                else:
                    correctness_list.append(0)
        else:
            raise NotImplementedError("Wrong GMM Estimate Method!")
       
        #with open('/home/v-zhixhan/data/zhixhan/local_pt_logs/gpt2-medium-agnews-pseudos/pt-0-gmm-4demons.pkl','wb') as f:
        #    pickle.dump(np.asarray(pred_list), f)
    return np.mean(correctness_list)


def get_p_content_free(params, train_sentences, train_labels, content_free_inputs=('N/A',)):
    """Query model with content free input, return its prediction probability for each label"""

    _, all_p_y = get_model_response(params, train_sentences, train_labels, content_free_inputs, normalize=False)

    p_y = np.mean(np.array(all_p_y), axis=0)
    p_y = p_y / np.sum(p_y) # normalize
    return p_y





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--models', dest='models', action='store', required=True, help='name of model(s), e.g., GPT2-XL')
    parser.add_argument('--datasets', dest='datasets', action='store', required=True, help='name of dataset(s), e.g., agnews')
    parser.add_argument('--num_seeds', dest='num_seeds', action='store', required=True, help='num seeds for the training set', type=int)
    parser.add_argument('--start_seed', dest='start_seed', action='store', required=True, help='start seed for the training set', type=int)

    parser.add_argument('--all_shots', dest='all_shots', action='store', required=True, help='num training examples to use')
    # other arguments
    parser.add_argument('--subsample_test_set', dest='subsample_test_set', action='store', required=False, type=int,
                        default=None, help='size of test set to use to speed up eval. None means using all test set')
    parser.add_argument('--bs', dest='bs', action='store', required=False, type=int, default=None,
                        help='batch size for model queries. For OpenAI API, capped at 20. For local running, set this to max out your GPU memory.')
    # flags
    parser.add_argument('--method', dest='method', type=str, required=True, default="gmm")
    parser.add_argument('--gmm_train_estimate_scale', dest='gmm_train_estimate_scale', action='store', type=int, default=None)

    
    args = parser.parse_args()
    args = vars(args)

    # simple processing
    def convert_to_list(items, is_int=False, is_float=False):
        if is_int:
            return [int(s.strip()) for s in items.split(",")]
        elif is_float:
            return [float(s.strip()) for s in items.split(",")]
        else:
            return [s.strip() for s in items.split(",")]

    args['models'] = convert_to_list(args['models'])
    args['datasets'] = convert_to_list(args['datasets'])
    args['all_shots'] = convert_to_list(args['all_shots'], is_int=True)

    main(**args)