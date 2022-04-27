
import nltk
from nltk.corpus import brown
import random
from nltk.tokenize import word_tokenize

# all_index function return all occurrence of an element in a list
def all_index(list,ele):
    list_indexs=[]
    for i in range(len(list)):
        if list[i]==ele:
            list_indexs.append(i)
    return list_indexs

from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
# to_baseform function convert a verb to its base form of verb, for example : speaking,spoken -> speak
def to_baseform(list_word):
    new_list=[]
    for word in list_word:
        new_list.append(lemmatizer.lemmatize(word, 'v'))
    return new_list

list_tag=brown.tagged_words()
words=brown.words()
type_adj=['JJ']    # adjective
type_adverb=['RB'] # adverb

# list_post_adj function return a list of index of adjective that follow an adverb in the adverb list(that would be defined in the main function)
def list_post_adj(pos):  # pos=very, highly, ...
    index_adj = all_index(words, pos)
    list_after_pos = []
    list_index_after_pos = []
    type_adj = ['JJ']
    for index in index_adj:
        if list_tag[index + 1][1] in type_adj:
            if '-' not in list_tag[index + 1][0] and list_tag[index + 1][0] not in list_after_pos:
                list_index_after_pos.append(index + 1)
                list_after_pos.append(list_tag[index + 1][0])
    return list_index_after_pos,list_after_pos

# list_post_adj function return a list of index of verb that follow an adverb in the adverb list(that would be defined in the main function)
def list_pos_verb(pos):
    index_very = all_index(words, pos)
    list_after_pos= []
    list_index_after_pos = []
    type_verb = ['VB', 'VBG', 'VBN', 'VBP', 'VBZ']
    for index in index_very:
        if list_tag[index + 1][1] in type_verb:
            verb=lemmatizer.lemmatize(list_tag[index + 1][0],'v')
            if '-' not in verb and verb not in list_after_pos:
                list_index_after_pos.append(index + 1)
                list_after_pos.append(verb)
    return list_index_after_pos, list_after_pos

from nltk.corpus import wordnet

#get_synonyms_list function return a list of synonyms of the word in parameter
def get_synonyms_list(word,type):
    syns=wordnet.synsets(word)
    synonyms=[]
    for i in range(len(syns)):
        syn=syns[i]
        type_of_syn = syn.name().split('.')[1]
        if type_of_syn in type:
            for l in syn.lemmas():
                synonyms.append(l.name())

    list_syn=list(set(synonyms))
    list_syn_with_occurrence=[]
    for ele in list_syn:
        if ele in words and ele != word:
            list_syn_with_occurrence.append(ele)
    return list_syn_with_occurrence

# adverb_attach_frequency function is used to measured the intensity of each word
# by measuring the frequency of its combining with an adverb in context
# higher score mean weaker-intensity word
def adverb_attach_frequency(word):
    list_index=all_index(words,word)
    occurrence_count=len(list_index)
    adverb_attach_count=0
    for index in list_index:
        if list_tag[index-1][1] in type_adverb :
            adverb_attach_count+=1
    return adverb_attach_count/occurrence_count

# sorting the synonyms of a word basing on their scores given by function adverb_attach_frequency
def sorted_contrasted_synonym(word,type):
    list_synonym=get_synonyms_list(word,type)
    list_frequency_of_synonym=[]
    for synonym in list_synonym:
        frequent=adverb_attach_frequency(synonym)
        list_frequency_of_synonym.append(frequent)
    for i in range(len(list_frequency_of_synonym)):
        for j in range(i,len(list_frequency_of_synonym)):
            if list_frequency_of_synonym[i] > list_frequency_of_synonym[j]:
                tam=list_frequency_of_synonym[i]
                list_frequency_of_synonym[i]=list_frequency_of_synonym[j]
                list_frequency_of_synonym[j]=tam

                tam2=list_synonym[i]
                list_synonym[i]=list_synonym[j]
                list_synonym[j]=tam2
    return list_synonym,list_frequency_of_synonym

# from the list of synonyms for each word, the function get_most_constrated_synonym would choose synonyms
# with strongest intensity basing on their score given by function adverb_attach_frequency
def get_most_constrated_synonym(word,type,threshold):
    list_sorted_synonym, list_sorted_frequency_of_synonym = sorted_contrasted_synonym(word,type)
    most_constrasted_synonyms=[]
    frequency_of_most_constrasted_synonyms=[]
    if len(list_sorted_synonym) >0:
        i=0
        while  i <= len(list_sorted_frequency_of_synonym)-1:
            if list_sorted_frequency_of_synonym[i] < threshold:
                most_constrasted_synonyms.append(list_sorted_synonym[i])
                frequency_of_most_constrasted_synonyms.append(list_sorted_frequency_of_synonym[i])
                i += 1
            else:
                break
    return most_constrasted_synonyms,frequency_of_most_constrasted_synonyms

# save2file to save file to path
def save2file(list,path_save):
    with open(path_save,'w') as f:
        for s in list:
            f.write(str(s)+'\n')

# load_file to load content of a file and return the list of content of each line
def load_file(path):
    with open(path,'r') as f:
        list = [eval(line.rstrip('\n')) for line in f]
        return list

#main function would create pairs of verbs and adjectives that are synonyms but constrasted in expression
def main():
    #This is the list of adverbs that would be included to show the relationship of expression
    adverbs_list = ['extremely', 'very', 'too', 'really', 'awfully', 'utterly', 'completely', 'absolutely', 'entirely',
                    'enormously', 'intensely', 'incredibly', 'intensely', 'highly', 'remarkably']
    list_pairs=[]

    #threshold_adj and threshold_verb to set a threshold, so that program would take the new pairs to list or not
    threshold_adj=0.1
    threshold_verb=0.1
    for adverb in adverbs_list:
        list_index_after_pos_adj, list_after_pos_adj=list_post_adj(adverb)
        # we are doing for adjectives
        for word in list_after_pos_adj:
            list_most_constrasted_synonyms, list_frequency_of_most_constrasted_synonyms=get_most_constrated_synonym(word,['a','s'],threshold_adj)
            for i in range(min(len(list_most_constrasted_synonyms),1)):
                list_pairs.append(tuple((adverb,word,list_most_constrasted_synonyms[i],list_frequency_of_most_constrasted_synonyms[i])))
        # next, we do for verbs
        list_index_after_pos_verb, list_after_pos_verb = list_pos_verb(adverb)
        for word in list_after_pos_verb:
            list_most_constrasted_synonyms, list_frequency_of_most_constrasted_synonyms = get_most_constrated_synonym(word,['v'],threshold_verb)
            for i in range(min(1,len(list_most_constrasted_synonyms))):
                list_pairs.append(tuple((adverb,word,list_most_constrasted_synonyms[i],list_frequency_of_most_constrasted_synonyms[i])))
    #shuffle the list before saving to file
    random.shuffle(list_pairs)
    save2file(list_pairs, 'total_pairs.csv')
main()  # this is to run the main function

# the function get50_pairs would return the initial 50 pairs in the saved file
# to get this list, call get50_pairs('total_pairs.csv')
def get50_pairs(path):
    list_pairs=load_file(path)
    return list_pairs[:50]
