## Import packages
!pip install lifelines

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from lifelines import CoxPHFitter
from lifelines import WeibullAFTFitter
from lifelines import LogNormalAFTFitter
from lifelines import LogLogisticAFTFitter
from lifelines import KaplanMeierFitter

## function for step 1 - null model
def mod_sel1(data, time_var, event_var, fun):
    fitter = fun
    mod1 = fitter.fit(data[[time_var] + [event_var]],
                      duration_col = time_var, event_col = event_var)
    mod1.print_summary()
    
## Function for step 2 - add one variable at a time, compare each individual model to null model
## data: preprocessed dataset
## time_var: input variable that represents time
## event_var: input variable that represents outcome event (i.e status)
## terms: input a list of the rest of the variables in the dataset
## fun: input CoxPHFitter if you want to fit a Cox Proportional model, or any AFT Fitter if you want to fit an AFT Model
## alpha: 0.1, df(degrees of freedom): 1 are default parameters for chi-square values in Goodness of Fit tests for each variable
## print_model: False if you want results printed immeditaly, other wise the function will print the selection process for step 1

def mod_sel2(data, time_var, event_var, terms, fun, alpha = 0.1,
             df = 1, print_model = False):
    
    print('Model Selection using ' + str(fun) + '\n')
    
    fitter = fun
    threshold = chi2.ppf(1 - alpha, df) ## critical point of chi-squared test
    
    mod1 = fitter.fit(data[[time_var] + [event_var]],
                      duration_col= time_var, event_col= event_var) ## fit null model as in step 1
    ll1 = mod1.log_likelihood_ ## extract log likelihood coefficient
    print('Step 2\n')
    
    ll_diffs = [] 
    p_vals = []
    for term in terms:
        mod2 = fitter.fit(data[[time_var] + [event_var] + [term]],
                          duration_col= time_var, event_col= event_var) ## fit model for individual variable 
        ll2 = mod2.log_likelihood_
        res = (2*ll2) - (2*ll1) ## Get likelihood ratio
        p = chi2.sf(res, 1) ## get p-value
        ll_diffs.append(res) ## list of likelihood ratios
        p_vals.append(p) ## list of p-values
        
    step2_mod = []
    for j in range(len(ll_diffs)):
        if ll_diffs[j] > threshold: ## add variable if its likelihood ratio is greater than the chi-square critical point
          ## print variable name, likelihood ratio, and p-value to demonstrate why it's added
            print('Variable Fitted: ' + terms[j])
            print('LL difference: ' + str(ll_diffs[j])) 
            print('p-value: ' + str(p_vals[j]))
            print('Add ' + terms[j])
            step2_mod.append(terms[j])
            print(' ')
        else:
          ## print variable name, likelihood ratio, and p-value to demonstrate why it's not added
            print('Variable Fitted: ' + terms[j])
            print('LL difference: ' + str(ll_diffs[j]))
            print('p-value: ' + str(p_vals[j]))
            print('Do not add ' + terms[j])
            print(' ')
    print('Step 2 Variables: ' + str(step2_mod) + '\n') ## print list of variables for step 2 model
    
    if print_model == True: ## print model process if set to True
        model = fitter.fit(data[[time_var] + [event_var] + step2_mod],
                           duration_col = time_var, event_col= event_var)
        model.print_summary()
        
## Step 3: Add one of the dropped variables at a time to model in step 3
## terms: input variables from model in step 3

def mod_sel3(data, time_var, event_var, terms, fun,
             alpha = 0.1, df= 1, print_model=False): 
    
    print('Step 3 Model Selection using ' + str(fun) + '\n')
    
    fitter = fun
    threshold = chi2.ppf(1 - alpha, df)
    ## model 1: fit model from step 2
    mod1 = fitter.fit(data[[time_var] + [event_var] + terms],
                      duration_col= time_var, event_col= event_var)
    ll1 = mod1.log_likelihood_
    
    
    term_lists = [] ## this creates a list of lists - each list consists of the provided input list minus iteratively one variable
    for x in range(len(terms)):
        term_l = list(terms)
        term_l.pop(x)
        term_lists.append(term_l)
    
    ll_diffs = []
    p_vals = []
    for term_list in term_lists: 
      ## model 2: fit models that iteratively exclude one variable from model 1
        mod2 = fitter.fit(data[[time_var] + [event_var] + term_list],
                          duration_col = time_var, event_col= event_var)
        ll2 = mod2.log_likelihood_
        res = (2*ll1) - (2*ll2) 
        p = chi2.sf(res, 1)
        ll_diffs.append(res)
        p_vals.append(p)
        
    step3_mod = []
    for k in range(len(ll_diffs)): 
        if ll_diffs[k] > threshold: ## do not drop particular variable if likelihood ratio is greater than critical point
            print('Variables Fitted: ' + str(term_lists[k]))
            print('LL difference: ' + str(ll_diffs[k]))
            print('p-value: ' + str(p_vals[k]))
            print('Do not drop ' + terms[k])
            step3_mod.append(terms[k])
            print(' ')
        else:
          ## drop particular variable othervwise
            print('Variables Fitted: ' + str(term_lists[k]))
            print('LL difference: ' + str(ll_diffs[k]))
            print('p-value: ' + str(p_vals[k]))
            print('Drop ' + terms[k])
            print(' ')
    print('Step 3 Variables: ' + str(step3_mod) + '\n')
    
    if print_model == True:
        model = fitter.fit(data[[time_var] + [event_var] + step3_mod],
                           duration_col = time_var, event_col= event_var)
        model.print_summary()
        
 ## step 4: iter
## terms1: variables from the last step
## terms2: add variables that were not added in step 2 and dropped in step 3

def mod_sel4(data, time_var, event_var, terms1, terms2, fun,
             alpha = 0.1, df = 1, print_model=False): 
    print('Step 4 Model Selection using ' + str(fun) + '\n')
    
    fitter = fun
    threshold = chi2.ppf(1 - alpha, df)
    
    ## model 1: fit model from step 3
    mod1 = fitter.fit(data[[time_var] + [event_var] + terms1],
                      duration_col= time_var, event_col= event_var)
    ll1 = mod1.log_likelihood_
    
    ll_diffs = []
    p_vals = []
    for term in terms2: ## fit one dropped variable at a time to step 3 model
      ## model 2: fit model with variables in model 1 plus one variable from terms2
        mod2 = fitter.fit(data[[time_var] + [event_var] + terms1 + [term]],
                          duration_col= time_var, event_col= event_var)
        ll2 = mod2.log_likelihood_
        res = (2*ll2) - (2*ll1) 
        p = chi2.sf(res, 1)
        ll_diffs.append(res)
        p_vals.append(p)
        
    step4_mod = []
    for j in range(len(ll_diffs)):
        if ll_diffs[j] > threshold: ## like step 2: add dropped variable if likelihood ratio is greater than the critical value
            print('Variable Fitted: ' + terms2[j])
            print('LL difference: ' + str(ll_diffs[j]))
            print('p-value: ' + str(p_vals[j]))
            print('Add ' + terms2[j])
            step4_mod.append(terms2[j])
            print(' ')
        else:
            print('Variable Fitted: ' + terms2[j])
            print('LL difference: ' + str(ll_diffs[j]))
            print('p-value: ' + str(p_vals[j]))
            print('Do not add ' + terms2[j])
            print(' ')
    print('Step 4 Variables: ' + str(terms1 + step4_mod) + '\n')
    
    if print_model == True:
        model = fitter.fit(data[[time_var] + [event_var] + terms1 + step4_mod],
                           duration_col= time_var, event_col= event_var)
        model.print_summary()
        
## step 5: drop one variable from step 4 model at a time
## terms: list of variables from step 4 model

def mod_sel5(data, time_var, event_var, terms, fun, alpha = 0.1, df= 1): 
    print('Step 4 Model Selection using ' + str(fun) + '\n')
    
    fitter = fun
    threshold = chi2.ppf(1 - alpha, df)
    mod1 = fitter.fit(data[[time_var] + [event_var] + terms],
                      duration_col= time_var, event_col= event_var)
    ll1 = mod1.log_likelihood_
    
    term_lists = []  ## create list of lists of terms minus one variable like in step 3
    for x in range(len(terms)):
        term_list = list(terms)
        term_list.pop(x)
        term_lists.append(term_list)
    
    ll_diffs = []
    p_vals = []
    for term_l in term_lists:
        mod2 = fitter.fit(data[[time_var] + [event_var] + term_l],
                          duration_col = time_var, event_col= event_var)
        ll2 = mod2.log_likelihood_
        res = (2*ll1) - (2*ll2)
        p = chi2.sf(res, 1)
        ll_diffs.append(res)
        p_vals.append(p)
        
    final_mod = []
    for k in range(len(ll_diffs)):
        if ll_diffs[k] > threshold:
            print('Variables Fitted: ' + str(term_lists[k]))
            print('LL difference: ' + str(ll_diffs[k]))
            print('p-value: ' + str(p_vals[k]))
            print('Do not drop ' + terms[k])
            final_mod.append(terms[k])
            print(' ')
            
        else:
            print('Variables Fitted: ' + str(term_lists[k]))
            print('LL difference: ' + str(ll_diffs[k]))
            print('p-value: ' + str(p_vals[k]))
            print('Drop ' + terms[k])
            print(' ')
            
    print('Step 5 Variables: ' + str(final_mod) + '\n')
    mod_final = fitter.fit(data[[time_var] + [event_var] + final_mod],
                           duration_col = time_var, event_col= event_var)
    ## print final model of the 5-step model selectio process
    mod_final.print_summary()
        
