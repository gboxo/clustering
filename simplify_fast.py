import pdb
import sys
import operator
from collections import OrderedDict
import subprocess
import numpy as  np
import json
import math
from transformers import BertTokenizer
import sys
import random
import time
import os
import tqdm
from multiprocessing import Pool, cpu_count

SINGLETONS_TAG  = "_singletons_ "
EMPTY_TAG = "_empty_ "
OTHER_TAG = "OTHER"
AMBIGUOUS = "AMB"
MAX_VAL = 20
TAIL_THRESH = 10
SUBWORD_COS_THRESHOLD = .2
MAX_SUBWORD_PICKS = 20

UNK_ID = 1
IGNORE_CONTINUATIONS=True
USE_PRESERVE=True

try:
    from subprocess import DEVNULL  # Python 3.
except ImportError:
    DEVNULL = open(os.devnull, 'wb')



def read_embeddings(embeds_file):
    with open(embeds_file) as fp:
        embeds_list = json.loads(fp.read())
    arr = np.array(embeds_list)
    return arr

def calculate_scores(args):
        term, terms, get_embedding_index, calc_inner_prod, tokenize = args
        scores = []

        index_term = get_embedding_index(term, tokenize)
        for other_term in terms:
            if term == other_term:
                continue

            index_other = get_embedding_index(other_term, tokenize)
            val = calc_inner_prod(index_term, index_other)
            scores.append(val)
        
        mean_score = sum(scores) / len(scores)
        std_dev = math.sqrt(sum((score - mean_score) ** 2 for score in scores) / len(scores))

        return term, mean_score, std_dev, scores

def consolidate_labels(existing_node,new_labels,new_counts):
    """Consolidates all the labels and counts for terms ignoring casing

    For instance, egfr may not have an entity label associated with it
    but eGFR and EGFR may have. So if input is egfr, then this function ensures
    the combined entities set fo eGFR and EGFR is made so as to return that union
    for egfr
    """
    new_dict = {}
    existing_labels_arr = existing_node["label"].split('/')
    existing_counts_arr = existing_node["counts"].split('/')
    new_labels_arr = new_labels.split('/')
    new_counts_arr = new_counts.split('/')
    assert(len(existing_labels_arr) == len(existing_counts_arr))
    assert(len(new_labels_arr) == len(new_counts_arr))
    for i in range(len(existing_labels_arr)):
        new_dict[existing_labels_arr[i]] = int(existing_counts_arr[i])
    for i in range(len(new_labels_arr)):
        if (new_labels_arr[i] in new_dict):
            new_dict[new_labels_arr[i]] += int(new_counts_arr[i])
        else:
            new_dict[new_labels_arr[i]] = int(new_counts_arr[i])
    sorted_d = OrderedDict(sorted(new_dict.items(), key=lambda kv: kv[1], reverse=True))
    ret_labels_str = ""
    ret_counts_str = ""
    count = 0
    for key in sorted_d:
        if (count == 0):
            ret_labels_str = key
            ret_counts_str = str(sorted_d[key])
        else:
            ret_labels_str += '/' +  key
            ret_counts_str += '/' +  str(sorted_d[key])
        count += 1
    return {"label":ret_labels_str,"counts":ret_counts_str}




def read_labels(labels_file):
    terms_dict = OrderedDict()
    lc_terms_dict = OrderedDict()
    with open(labels_file,encoding="utf-8") as fin:
        count = 1
        for term in fin:
            term = term.strip("\n")
            term = term.split()
            if (len(term) == 3):
                terms_dict[term[2]] = {"label":term[0],"counts":term[1]}
                lc_term = term[2].lower()
                if (lc_term in lc_terms_dict):
                     lc_terms_dict[lc_term] = consolidate_labels(lc_terms_dict[lc_term],term[0],term[1])
                else:
                     lc_terms_dict[lc_term] = {"label":term[0],"counts":term[1]}
                count += 1
            else:
                print("Invalid line:",term)
                assert(0)
    print("count of labels in " + labels_file + ":", len(terms_dict))
    return terms_dict,lc_terms_dict


def read_entities(terms_file):
    ''' Read bootstrap entities file

    '''
    terms_dict = OrderedDict()
    with open(terms_file,encoding="utf-8") as fin:
        count = 1
        for term in fin:
            term = term.strip("\n")
            if (len(term) >= 1):
                nodes = term.split()
                assert(len(nodes) == 2)
                lc_node = nodes[1].lower()
                if (lc_node in terms_dict):
                    pdb.set_trace()
                    assert(0)
                    assert('/'.join(terms_dict[lc_node]) == nodes[0])
                terms_dict[lc_node] = nodes[0].split('/')
                count += 1
    print("count of entities in ",terms_file,":", len(terms_dict))
    return terms_dict



def read_terms(terms_file):
    terms_dict = OrderedDict()
    with open(terms_file,encoding="utf-8") as fin:
        count = 1
        for term in fin:
            term = term.strip("\n")
            if (len(term) >= 1):
                terms_dict[term] = count
                count += 1
    print("count of tokens in ",terms_file,":", len(terms_dict))
    return terms_dict

def is_subword(key):
        return True if str(key).startswith('#')  else False

def is_filtered_term(key): #Words selector. skiping all unused and special tokens
    if (IGNORE_CONTINUATIONS):
        return True if (is_subword(key) or str(key).startswith('[')) else False
    else:
        return True if (str(key).startswith('[')) else False

def filter_2g(term,preserve_dict):
    if (USE_PRESERVE):
        return True if  (len(term) <= 2 and term not in preserve_dict) else False
    else:
        return True if  (len(term) <= 2 ) else False

class BertEmbeds:
    def __init__(self, model_path,do_lower, terms_file,embeds_file,cache_embeds,normalize,labels_file,stats_file,preserve_2g_file,glue_words_file,bootstrap_entities_file):
        do_lower = True if do_lower == 1 else False
        self.tokenizer = BertTokenizer.from_pretrained(model_path,do_lower_case=do_lower)
        self.terms_dict = read_terms(terms_file)
        self.labels_dict,self.lc_labels_dict = read_labels(labels_file)
        self.stats_dict = read_terms(stats_file) #Not used anymore
        self.preserve_dict = read_terms(preserve_2g_file)
        self.gw_dict = read_terms(glue_words_file)
        self.bootstrap_entities = read_entities(bootstrap_entities_file)
        self.embeddings = read_embeddings(embeds_file)
        self.dist_threshold_cache = {}
        self.dist_zero_cache = {}
        self.normalize = normalize
        self.similarity_matrix = np.array(self.cache_matrix(True))
        self.lookup_table = {key: index for index, key in enumerate(self.terms_dict.keys())}





    def cache_matrix(self,normalize):
        b_embeds = self
        print("Computing similarity matrix (takes approx 5 minutes for ~100,000x100,000 matrix ...)")
        start = time.time()
        #pdb.set_trace()
        vec_a = b_embeds.embeddings.T #vec_a shape (1024,)
        if (normalize):
            vec_a = vec_a/np.linalg.norm(vec_a,axis=0) #Norm is along axis 0 - rows
            vec_a = vec_a.T #vec_a shape becomes (,1024)
            similarity_matrix = np.inner(vec_a,vec_a)
        end = time.time()
        time_val = (end-start)*1000
        print("Similarity matrix computation complete.Elapsed:",time_val/(1000*60)," minutes")
        return similarity_matrix


    def get_embedding_index(self,text,tokenize=False):
            if (tokenize):
                assert(0)
                tokenized_text = self.tokenizer.tokenize(text)
            else:
                if (not text.startswith('[')):
                    tokenized_text = text.split()
                else:
                    tokenized_text = [text]
            indexed_tokens = self.lookup_table[tokenized_text[0]]
            #assert(len(indexed_tokens) == 1)
            return indexed_tokens
    

    #def calc_inner_prod(self,text1,text2,tokenize):
    #        assert(tokenize == False)
    #        index1 = self.get_embedding_index(text1)
    #        index2 = self.get_embedding_index(text2)
    #        return self.similarity_matrix[index1][index2]

    #def get_terms_above_threshold(self,term1,threshold,tokenize):
    #        final_dict = {}
    #        for k in self.terms_dict:
    #            term2 = k.strip("\n")
    #            val = self.calc_inner_prod(term1,term2,tokenize)
    #            val = round(val,2)
    #            if (val > threshold):
    #                final_dict[term2] = val
    #        sorted_d = OrderedDict(sorted(final_dict.items(), key=lambda kv: kv[1], reverse=True))
    #        return sorted_d
    

    #----------------

    def calc_inner_prod(self, index1, index2):
        return self.similarity_matrix[index1, index2]

    def get_terms_above_threshold(self, term1, threshold, tokenize=False):
        assert not tokenize, "Tokenize should be False"
        
        index1 = self.get_embedding_index(term1, tokenize)
        final_dict = {}
        
        for term2,k in self.terms_dict.items():
            index2 = self.get_embedding_index(term2.strip("\n"), tokenize)
            val = self.calc_inner_prod(index1, index2)
            val = round(val, 2)
            if val > threshold:
                final_dict[term2] = val
                
        return OrderedDict(sorted(final_dict.items(), key=lambda kv: kv[1], reverse=True))
    #----------------
    def labeled_term(self,k):
            if (k not in self.bootstrap_entities):
                return False
            labels = self.bootstrap_entities[k]
            if (len(labels) > 1):
                return True
            assert(len(labels) == 1)
            if (labels[0] == "UNTAGGED_ENTITY"):
                return False
            return True
    def create_entity_labels_file(self,full_entities_dict):
        with open("labels.txt","w") as fp:
            for term in self.terms_dict:
                if (term not in full_entities_dict and term.lower() not in self.bootstrap_entities):
                    fp.write("OTHER 0 " + term + "\n")
                    continue
                if (term not in full_entities_dict): #These are vocab terms that did not show up in a cluster but are present in bootstrap list
                    lc_term = term.lower()
                    counts_str = len(self.bootstrap_entities[lc_term])*"0/"
                    fp.write('/'.join(self.bootstrap_entities[lc_term]) + ' ' + counts_str.rstrip('/') + ' ' + term + '\n') #Note the term output is case sensitive. Just the indexed version is case insenstive
                    continue
                out_entity_dict = {}
                for entity in full_entities_dict[term]:
                    assert(entity not in out_entity_dict)
                    out_entity_dict[entity] = full_entities_dict[term][entity]
                sorted_d = OrderedDict(sorted(out_entity_dict.items(), key=lambda kv: kv[1], reverse=True))
                entity_str = ""
                count_str = ""
                for entity in sorted_d:
                    if (len(entity_str) == 0):
                        entity_str = entity
                        count_str =  str(sorted_d[entity])
                    else:
                        entity_str += '/' +  entity
                        count_str +=  '/' + str(sorted_d[entity])
                if (len(entity_str) > 0):
                    fp.write(entity_str + ' ' + count_str + ' ' + term + "\n")


    def subword_clustering(self):
            '''
                Generate clusters for terms in vocab
                This is used for unsupervised NER (with subword usage)
            '''
            tokenize = False
            count = 1
            total = len(self.terms_dict)
            pivots_dict = OrderedDict()
            singletons_arr = []
            full_entities_dict = OrderedDict()
            untagged_items_dict = OrderedDict()
            empty_arr = []
            total = len(self.terms_dict)
            dfp = open("adaptive_debug_pivots.txt","w")
            esupfp = open("entity_support.txt","w")
            for key in tqdm.tqdm(self.terms_dict):
                if (key.startswith('[') or len(key) < 2):
                    count += 1
                    continue
                count += 1
                #print(":",key)
                sorted_d = self.get_terms_above_threshold(key,SUBWORD_COS_THRESHOLD,tokenize)
                arr = []
                labeled_terms_count = 0
                for k in sorted_d:
                    if (self.labeled_term(k.lower())):
                        labeled_terms_count += 1
                    arr.append(k)
                    if (labeled_terms_count >= MAX_SUBWORD_PICKS):
                        break
                #print("Processing: ",key,"count:",count," of ",total)
                if (len(arr) >  0):
                    max_mean_term,max_mean, std_dev,s_dict = self.find_pivot_subgraph(arr,tokenize)
                    if (max_mean_term not in pivots_dict):
                        new_key  = max_mean_term
                    else:
                        #print("****Term already a pivot node:",max_mean_term, "key  is :",key)
                        new_key  = max_mean_term + "++" + key
                    #pivots_dict[new_key] = {"key":new_key,"orig":key,"mean":max_mean,"terms":arr}
                    pivots_dict[key] = {"key":new_key,"orig":key,"mean":max_mean,"terms":arr}
                    entity_type,entity_counts,curr_entities_dict = self.get_entity_type(arr,new_key,esupfp)
                    self.aggregate_entities_for_terms(arr,curr_entities_dict,full_entities_dict,untagged_items_dict)
                    #print(entity_type,entity_counts,new_key,max_mean,std_dev,arr)
                    dfp.write(entity_type + " " + entity_counts + " " + new_key + " " + new_key + " " + new_key+" "+key+" "+str(max_mean)+" "+ str(std_dev) + " " +str(arr)+"\n")
                else:
                    #print("***Empty arr for term:",key)
                    empty_arr.append(key)

            dfp.write(SINGLETONS_TAG + str(singletons_arr) + "\n")
            dfp.write(EMPTY_TAG + str(empty_arr) + "\n")
            with open("pivots.json","w") as fp:
                fp.write(json.dumps(pivots_dict))
            with open("pivots.txt","w") as fp:
                for k in pivots_dict:
                    fp.write(k + '\n')
            dfp.close()
            esupfp.close()
            self.create_entity_labels_file(full_entities_dict)
    def aggregate_entities_for_terms(self,arr,curr_entities_dict,full_entities_dict,untagged_items_dict):
            if (len(curr_entities_dict) == 0):
                return
            for term in arr:
                if term not in full_entities_dict: #This is case sensitive. We want vocab entries eGFR and EGFR to pick up separate weights for their entities
                    full_entities_dict[term] = OrderedDict()
                for entity in curr_entities_dict:
                    #if  (entity not in term_entities): #aggregate counts only for entities present for this term in original manual harvesting list(bootstrap list)
                    #    continue
                    if (entity not  in full_entities_dict[term]):
                        full_entities_dict[term][entity] = curr_entities_dict[entity]
                    else:
                        full_entities_dict[term][entity] += curr_entities_dict[entity]


    def get_entity_type(self,arr,new_key,esupfp):
            e_dict = {}
            #print("GET:",arr)
            for term in arr:
                term = term.lower() #bootstrap entities is all lowercase.
                if (term in self.bootstrap_entities):
                    entities = self.bootstrap_entities[term]
                    for entity in entities:
                        if (entity in e_dict):
                                #print(term,entity)
                                e_dict[entity] += 1
                        else:
                                #print(term,entity)
                                e_dict[entity] = 1
            ret_str = ""
            count_str = ""
            entities_dict = OrderedDict()
            if (len(e_dict) >= 1):
                sorted_d = OrderedDict(sorted(e_dict.items(), key=lambda kv: kv[1], reverse=True))
                #print(new_key + ":" + str(sorted_d))
                esupfp.write(new_key + ' ' + str(sorted_d) + '\n')
                count = 0
                for k in sorted_d:
                    if (len(ret_str) > 0):
                        ret_str += '/' + k
                        count_str += '/' + str(sorted_d[k])
                    else:
                        ret_str = k
                        count_str = str(sorted_d[k])
                    entities_dict[k] = int(sorted_d[k])
                    count += 1
            if (len(ret_str) <= 0):
                ret_str = "OTHER"
                count_str = str(len(arr))
            #print(ret_str)
            count_str += '/' + str(len(arr))
            return ret_str,count_str,entities_dict


    """ def find_pivot_subgraph(self,terms,tokenize):
            max_mean = 0
            std_dev = 0
            max_mean_term = None
            means_dict = {}
            if (len(terms) == 1):
                return terms[0],1,0,{terms[0]:1}
            for i in terms:
                full_score = 0
                count = 0
                full_dict = {}
                index_i = self.get_embedding_index(i.strip("\n"), tokenize)
                for j in terms:
                    
                    if (i != j):
                        index_j = self.get_embedding_index(j.strip("\n"), tokenize)

                        val = self.calc_inner_prod(index_i,index_j)
                        #print(i+"-"+j,val)
                        full_score += val
                        full_dict[count] = val
                        count += 1
                if (len(full_dict) > 0):
                    mean  =  float(full_score)/len(full_dict)
                    means_dict[i] = mean
                    #print(i,mean)
                    if (mean > max_mean):
                        #print("MAX MEAN:",i)
                        max_mean_term = i
                        max_mean = mean
                        std_dev = 0
                        for k in full_dict:
                            std_dev +=  (full_dict[k] - mean)*(full_dict[k] - mean)
                        std_dev = math.sqrt(std_dev/len(full_dict))
                        #print("MEAN:",i,mean,std_dev)
            #print("MAX MEAN TERM:",max_mean_term)
            sorted_d = OrderedDict(sorted(means_dict.items(), key=lambda kv: kv[1], reverse=True))
            return max_mean_term,round(max_mean,2),round(std_dev,2),sorted_d  """
 

#----------------------
    
 
    def find_pivot_subgraph(self, terms, tokenize):
        terms = [term.strip("\n") for term in terms]

        if len(terms) == 1:
            return terms[0], 1, 0, {terms[0]: 1}

        max_mean = 0
        max_mean_term = None
        means_dict = {}
        std_dev = 0

        # Parallelize the calculation
        with Pool(processes=cpu_count()) as pool:
            args = [(term, terms, self.get_embedding_index, self.calc_inner_prod, tokenize) for term in terms]
            results = pool.map(calculate_scores, args,chunksize=1000)
        
        # Post-process results
        for term, mean_score, term_std_dev, scores in results:
            means_dict[term] = mean_score
            if mean_score > max_mean:
                max_mean_term = term
                max_mean = mean_score
                std_dev = term_std_dev

        sorted_means = OrderedDict(sorted(means_dict.items(), key=lambda kv: kv[1], reverse=True))

        return max_mean_term, round(max_mean, 2), round(std_dev, 2), sorted_means

#---------------------

def main():
    b_embeds =BertEmbeds(os.getcwd(),0,"vocab.txt","bert_vectors.txt",True,True,"results/labels.txt","results/stats_dict.txt","preserve_1_2_grams.txt","glue_words.txt","bootstrap_entities.txt") #True - for cache embeds; normalize - True
    display_threshold = .4
    b_embeds.subword_clustering()



if __name__ == '__main__':
    main()


