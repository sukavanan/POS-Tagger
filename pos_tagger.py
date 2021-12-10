import numpy as np
import pandas as pd
import json
import operator

vocab_dict = {}            #dictionary for the vocabulary and its frequency, key - word, value - frequency
pos_freq_dict = {}         #dictionary for the pos tags and its frequence, key - pos tag, value - frequency
transition = {}            #dictionary to maintain t(s'|s), key - (s, s'), value - t(s'|s)
emission = {}              #dictionary to maintain e(s|x), key - (s, x), value - e(s|x)
pos_idx_dict = {}          #dictionary to maintain the index to pos tag matching, key - index, value - pos tag
pos_idx_dict_inv = {}      #inverse of above dictionary, key - pos tag, value - index
vocab_idx_dict = {}        #dictionary to maintain the index to word matching, key - index, value - word
vocab_idx_dict_inv = {}    #inverse of above dictionary, key - word, value - index
sentences = 0
threshold = 1
unknown_count = 0

#list of numbers in words, used to assign the <num> tag
numbers = ['one','two','three','four','five', 'six','seven','eight','nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'zero', 'hundred', 'thousand', 'million', 'billion', 'trillion', 'quadrillion', 'quintillion', 'sextillion', 'septillion', 'octillion', 'nonillion', 'decillion']

#file paths
train_path = "data/train"
dev_path = "data/dev"
test_path = "data/test"
vocab_path = "data/vocab.txt"
hmm_path = "data/hmm.json"
greedy_path = "outputs/greedy.out"
viterbi_path = "outputs/viterbi.out"

#function to identify if a given word is a number
def is_number(s):
    try:
        if "," in s:
            s = s.replace(",", "")
        if ":" in s:
            s = s.replace(":", "")
        if "\/" in s:
            s = s.replace("\/", "")
        if s.lower() in numbers:
            return True
        float(s)
        return True
    except ValueError:
        return False

#function to identify if a given word is a compound word
def is_compound(s):
    if "-" in s:
        s = s.replace("-", "")
        if s.isalnum():
            return True
        else:
            return False
    else:
        return False

#creating the vocabulary
with open(train_path, "r") as file_obj:
    lines = file_obj.readlines()
line_list = [line.rstrip().split("\t") for line in lines]
line_list = [x for x in line_list if x != ['']]

for index, word, pos in line_list:
    if index == '1':
        sentences += 1
    if is_number(word):
        word = "<num>"
    elif is_compound(word):
        word = "<cmp>"
    if word in vocab_dict:
        vocab_dict[word] += 1
    else:
        vocab_dict[word] = 1
    if pos in pos_freq_dict:
        pos_freq_dict[pos] += 1
    else:
        pos_freq_dict[pos] = 1

max_pos_tag = max(pos_freq_dict, key = pos_freq_dict.get)

#replacing rare words with <unk>
for word, freq in list(vocab_dict.items()):
    if freq <= threshold:
        unknown_count += freq
        del vocab_dict[word]

vocab_dict["<unk>"] = unknown_count
vocab_dict = dict(sorted(vocab_dict.items(), key = lambda item: item[1], reverse = True))

#creating the vocab.txt file
index = 1
with open(vocab_path, "w") as fp:
    fp.write("<unk>\t{}\t{}\n".format(index, str(unknown_count)))
    for word, freq in vocab_dict.items():
        index += 1
        if word != "<unk>":
            fp.write("{}\t{}\t{}\n".format(word, index, freq))

print("Threshold for rare words: <=", threshold)
print("Size of vocabulary: ", len(vocab_dict))
print("Total no.of sentences: ", sentences)
print("Total number of occurunces of <unk>: ",vocab_dict["<unk>"])

#creating the transition and emission dictionaries
for i in range(len(line_list) - 1):
    pos = line_list[i][2]
    pos_ = line_list[i+1][2]
    word = line_list[i][1]
    
    if is_number(word):
        word = "<num>"
    elif is_compound(word):
        word = "<cmp>"
    
    if word not in vocab_dict:
        word = "<unk>"
    
    denom = pos_freq_dict[pos]
    
    if line_list[i][0] == '1':
        if ("<s>", pos) in transition:
            transition[("<s>", pos)] += (1 / sentences)
        else:
            transition[("<s>", pos)] = (1 / sentences)
    else:
        if (pos, pos_) in transition:
            transition[(pos, pos_)] += (1 / denom)
        else:
            transition[(pos, pos_)] = (1 / denom)
    
    if (pos, word) in emission:
        emission[(pos, word)] += (1 / denom)
    else:
        emission[(pos, word)] = (1 / denom)

#dropping 0 probabilities
for key, val in list(transition.items()):
    if val == 0:
        del transition[key]

for key, val in list(emission.items()):
    if val == 0:
        del emission[key]

#converting the keys to string so that it could be stored as json file
str_transition = {key[0] + "," + key[1]: val for key, val in transition.items()}
str_emission = {key[0] + "," + key[1]: val for key, val in emission.items()}

print("Total no.of transmission probabilities: ",len(transition))
print("Total no.of emission probabilities: ",len(emission))

json_dict = {"transition":str_transition, "emission":str_emission}
with open(hmm_path, "w") as fp:
    json.dump(json_dict, fp)

#function to predict pos tag based on the greedy decoding of hmm algorithm
def greedy_decoding(pos, word):
    temp_trans = {key[1]: val for key, val in transition.items() if key[0] == pos}
    temp_emiss = {key[0]: val for key, val in emission.items() if key[1] == word}
    max_val = 0.0
    pos_tag = max_pos_tag
    flag = 0
    
    trans_set = set(temp_trans)
    emiss_set = set(temp_emiss)
    for key in trans_set.intersection(emiss_set):
        flag = 1
        res = temp_trans[key] * temp_emiss[key]
        if res > max_val:
            max_val = res
            pos_tag = key

#flag variable used to check if there is a common pos tag between the transition set and the emission set
#if no common pos tag found, and emission set is not empty then the max valued pos tag from the emission set is returned
#else, max values pos tag from transition set is returned
#if both are empty the most common pos tag in the whole dataset is returned
    if flag == 0:
        if len(temp_emiss) != 0:
            return max(temp_emiss, key = temp_emiss.get)
        elif len(temp_trans) != 0:
            return max(temp_trans, key = temp_trans.get)
        else:
            return pos_tag
        
    return pos_tag

#function that calls the greedy_decoding, unknown, number and compound words are replaced with their respective tags
def predict(dataset):
    correct = 0
    for i in range(len(dataset)):
        word = dataset[i][1]
        pos = dataset[i][2]
        if is_number(word):
            word = "<num>"
        elif is_compound(word):
            word = "<cmp>"
        if word not in vocab_dict:
            word = "<unk>"
        if dataset[i][0] == '1':
            pred_tag = greedy_decoding("<s>", word)
        else:
            prev_pos = dataset[i-1][2]
            pred_tag = greedy_decoding(prev_pos, word)
        if pred_tag == pos:
            correct += 1
        elif pred_tag == -1:
            print(i, pos)
            break
    return (correct / len(dataset)) * 100

#function to create the greedy.out file for the test dataset
def test(dataset):
    with open(greedy_path, "w") as fp:
        pos = ""
        for i in range(len(dataset)):
            index = dataset[i][0]
            word = dataset[i][1]
            if index == '1':
                pos = "<s>"
            if is_number(word):
                word = "<num>"
            elif is_compound(word):
                word = "<cmp>"
            if word not in vocab_dict:
                word = "<unk>"
            pred_tag = greedy_decoding(pos, word)
            pos = pred_tag
            fp.write("{}\t{}\t{}\n".format(index, dataset[i][1], pred_tag))
            if i != len(dataset) - 1 and dataset[i+1][0] == '1':
                fp.write("\n")

#creating the index-pos tag (and its inverse) and the index-word (and its inverse) dictionaries
idx = 0
for key, val in pos_freq_dict.items():
    pos_idx_dict[idx] = key
    pos_idx_dict_inv[key] = idx
    idx += 1

idx = 0
for key, val in vocab_dict.items():
    vocab_idx_dict[idx] = key
    vocab_idx_dict_inv[key] = idx
    idx += 1

#creating the prior, transmission and emission matrices for the viterbi decoding algorithm
prior = np.zeros((len(pos_idx_dict),), dtype = float)
for i in range(len(pos_idx_dict)):
    if ("<s>", pos_idx_dict[i]) in transition:
        prior[i] = transition[("<s>", pos_idx_dict[i])]
    else:
        prior[i] = 0

tm = np.zeros((len(pos_idx_dict), len(pos_idx_dict)), dtype = float)
for i in range(len(pos_idx_dict)):
    for j in range(len(pos_idx_dict)):
        if (pos_idx_dict[i], pos_idx_dict[j]) in transition:
            tm[i][j] = transition[(pos_idx_dict[i], pos_idx_dict[j])]
        else:
            tm[i][j] = 0

em = np.zeros((len(pos_idx_dict), len(vocab_idx_dict)), dtype = float)
for i in range(len(pos_idx_dict)):
    for j in range(len(vocab_idx_dict)):
        if (pos_idx_dict[i], vocab_idx_dict[j]) in emission:
            em[i][j] = emission[(pos_idx_dict[i], vocab_idx_dict[j])]
        else:
            em[i][j] = 0

#function to predict pos tag based on the viterbi decoding for hmm algorithm
def viterbi_decoding(sent_data):
    samples = len(sent_data)
    pos_tags = len(pos_idx_dict)
    viterbi = np.zeros((pos_tags, samples), dtype = float) #the dynamic matrix
    path = np.zeros((pos_tags, samples), dtype = int)      #matrix to hold the previous max tag for a tag, word pair
    result_path = np.zeros(samples, dtype = int)           #array that gives the indices of the pos tags that maximizes the probability 
    scale = np.zeros(samples, dtype = float)               #scale factor used to prevent underflow of probabilities
    
    word = sent_data[0]
    viterbi[:, 0] = prior.T * em[:, vocab_idx_dict_inv[word]]
    
    denom = np.sum(viterbi[:, 0])
    if denom != 0:
        scale[0] = 1.0 / denom
        viterbi[:, 0] = scale[0] * viterbi[:, 0]
    
    #the loops, to calculate [i,j]th value of the viterbi matrix
    for i in range(1, samples):
        word = sent_data[i]
        for j in range(pos_tags):
            temp_trans = viterbi[:, i-1] * tm[:, j]
            path[j, i], viterbi[j, i] = max(enumerate(temp_trans), key = operator.itemgetter(1))
            viterbi[j, i] = viterbi[j, i] * em[j, vocab_idx_dict_inv[word]]
        
        denom = np.sum(viterbi[:, i])
        if denom != 0:
            scale[i] = 1 / denom
            viterbi[:, i] = scale[i] * viterbi[:, i]
    
    #backtracking through the path matrix to find the highest probability resulting pos tags
    result_path[samples - 1] = viterbi[:, samples - 1].argmax()
    for i in range(samples - 1, 0, -1):
        result_path[i - 1] = path[result_path[i], i]
    
    return result_path

#function that calls the viterbi_decoding function, replaces unknown, number and compound words with their respective tags
def predict_viterbi(dataset):
    correct = 0
    sent_data = []
    sent_pos = []
    for i in range(len(dataset)):
        index = dataset[i][0]
        word = dataset[i][1]
        pos = dataset[i][2]

        if is_number(word):
            word = "<num>"
        elif is_compound(word):
            word = "<cmp>"
        if word not in vocab_dict:
            word = "<unk>"
        
        #dividing the whole dataset into sentences and sending it to the viterbi_decoding function
        sent_data.append(word)
        sent_pos.append(pos)        
        if i == len(dataset) - 1:
            pred_tag = []
            path = viterbi_decoding(sent_data)
            pred_tag = [pos_idx_dict[path[i]] for i in range(len(path))]
            correct += (np.array(pred_tag) == np.array(sent_pos)).sum()
        
        elif dataset[i+1][0] == '1':
            pred_tag = []
            path = viterbi_decoding(sent_data)
            pred_tag = [pos_idx_dict[path[i]] for i in range(len(path))]
            correct += (np.array(pred_tag) == np.array(sent_pos)).sum()
            sent_data = []
            sent_pos = []
    return (correct / len(dataset)) * 100

#function to predict the pos tags for the test dataset
def test_viterbi(dataset):    
    with open(viterbi_path, "w") as fp:
        sent_data = []
        orig_word = []
        for i in range(len(dataset)):
            index = dataset[i][0]
            word = dataset[i][1]
            orig_word.append(word)
            
            if is_number(word):
                word = "<num>"
            elif is_compound(word):
                word = "<cmp>"
            if word not in vocab_dict:
                word = "<unk>"

            sent_data.append(word)        
            if i == len(dataset) - 1:
                pred_tag = []
                path = viterbi_decoding(sent_data)
                pred_tag = [pos_idx_dict[path[i]] for i in range(len(path))]
                for j in range(len(sent_data)):
                    fp.write("{}\t{}\t{}\n".format(j+1, orig_word[j], pred_tag[j]))
            
            elif dataset[i+1][0] == '1':
                pred_tag = []
                path = viterbi_decoding(sent_data)
                pred_tag = [pos_idx_dict[path[i]] for i in range(len(path))]
                for j in range(len(sent_data)):
                    fp.write("{}\t{}\t{}\n".format(j+1, orig_word[j], pred_tag[j]))
                fp.write("\n")
                sent_data = []
                orig_word = []

print("Shape of Prior array: ", prior.shape)
print("Shape of Transition Matrix: ", tm.shape)
print("Shape of Emission Matrix: ", em.shape)

with open(dev_path, "r") as fp:
    dev_lines = fp.readlines()
dev_lines = [line.rstrip().split("\t") for line in dev_lines]
dev_lines = [x for x in dev_lines if x != ['']]
accuracy = predict(dev_lines)
print("Dev Set accuracy for Greedy Decoding of HMM: ",accuracy)
accuracy = predict_viterbi(dev_lines)
print("Dev Set accuracy for Viterbi Decoding of HMM: ",accuracy)

with open(test_path, "r") as fp:
    test_lines = fp.readlines()
test_lines = [line.rstrip().split("\t") for line in test_lines]
test_lines = [x for x in test_lines if x != ['']]
test(test_lines)
print("Created greedy.out file...")
test_viterbi(test_lines)
print("Created viterbi.out file...")

