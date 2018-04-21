import re, random
#import spacy
#nlp = spacy.load('en')
import csv

#todo:shall we cleanup the word? we can handle multiple words with it
#def spacy_tokenize(text):
#    return [token.text for token in nlp.tokenizer(text)]

#todo: implement diff distance calc algos
def dameraulevenshtein(seq1, seq2):
    """Calculate the Damerau-Levenshtein distance between sequences."""
    oneago = None
    thisrow = list(range(1, len(seq2) + 1)) + [0]
    for x in range(len(seq1)):
        # Python lists wrap around for negative indices, so put the
        # leftmost column at the *end* of the list. This matches with
        # the zero-indexed strings and saves extra calculation.
        twoago, oneago, thisrow = (oneago, thisrow, [0] * len(seq2) + [x + 1])
        for y in range(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
            # This block deals with transpositions
            if (x > 0 and y > 0 and seq1[x] == seq2[y - 1]
                    and seq1[x - 1] == seq2[y] and seq1[x] != seq2[y]):
                thisrow[y] = min(thisrow[y], twoago[y - 2] + 1)
    return thisrow[len(seq2) - 1]


class MyFuzzySearch:
    def __init__(self, max_lev_distance=3):
        self.max_lev_distance = max_lev_distance
        self.dictionary = {}
        self.longest_word_length = 0

    def get_deletes_list(self, w):
        """For a given w, get strings with up to max_lev_distance characters deleted in it"""
        words_from_delete = []
        queue = [w]
        for del_count in range(self.max_lev_distance):
            temp_queue = []
            for word in queue:
                if len(word) > 1:
                    for c in range(len(word)):
                        word_without_c = '%s%s' % (word[:c], word[c + 1:])
                        if word_without_c not in words_from_delete:
                            words_from_delete.append(word_without_c)
                        if word_without_c not in temp_queue:
                            temp_queue.append(word_without_c)
            queue = temp_queue

        return words_from_delete

    def create_dictionary_entry(self, w, f):
        '''add word and anything from the word into our dict'''
        # check if word is already in dictionary
        # dictionary entries are in the form: (list of suggested corrections,
        # frequency of word in corpus)
        #todo: for each word we have a list of possible words that can either arise from it(completion), or changes from it(correction)
        #todo: how to add frequency in our dict?
        #todo: here when a word from corpus comes, it increases f, we can directly set f since our corpus has unique words + f
        #todo: corrections get f = 0 if its added newly to dict, it doesnt change if already there, f is replaced by f from corpus if added later.
        #todo:
        #todo: why only deletes and no edits of a word?
        #todo: we wont return new_real_word_added, our case no return
        #todo:
        new_real_word_added = False
        if w in self.dictionary:
            # increment count of word in corpus
            self.dictionary[w] = (self.dictionary[w][0], f)
        else:
            self.dictionary[w] = ([], f)
            self.longest_word_length = max(self.longest_word_length, len(w))

        #add corrections only for popular words, we need to set a threshold to consider popular ones
        #todo: try bottom_15 avg
        if self.dictionary[w][1] > 100059294:
            deletes = self.get_deletes_list(w)
            for item in deletes:
                if item in self.dictionary:
                    # add (correct) word to delete's suggested correction list
                    self.dictionary[item][0].append(w)
                else:
                    # note frequency of word in corpus is not incremented
                    self.dictionary[item] = ([w], 0)


    def create_dictionary(self, fname):
        with open(fname, 'rb') as csvfile:
            wordreader = csv.reader(csvfile, delimiter='\t')
            for row in wordreader:
                self.create_dictionary_entry(row[0], row[1])

        #todo: also analyse and find common and rare thresholds
        self.threshold_deduct = 1000000
        self.threshold_add = 100000000
        return self.dictionary

    def get_possible_words(self, string):
        """return list of possible words related to string"""
        if (len(string) - self.longest_word_length) > self.max_lev_distance:
            return []

        possible_words_dict = {}
        min_suggest_len = float('inf')

        queue = [string]
        processed_words = {}

        while len(queue) > 0:
            item_from_queue = queue[0]
            queue = queue[1:]
            #todo: if length of word is over length of a word in dict by 2, such words are not compared?
            #todo: if the word type is misspelt, its in our dict, but its f will be 0 or low.
            #todo: generally a misspelt word will likely match a delete of a real word or atleast lies close to it.
            #todo: so we can get away from keeping substitutions of a read word in dict!
            #todo: should we ever suggest exact match? isn't it pointless?
            #todo:
            #todo: matches at the start of a word should be ranked higher
            #todo: Short words should rank higher, but we should consider its f, for each extra letter,
            #todo: the suggested_word loses 'x' from its f. need a smart way to determine 'x'
            #todo: this will penalize larger words in somewhat reasonable way
            #todo: exact match should be first if matched with a real word with non 0 f
            #todo: need to match with strings much longer



            if (item_from_queue in self.dictionary):
                #todo: make it greater than a threshold!
                if self.dictionary[item_from_queue][1] > 0:
                    if len(item_from_queue) < (len(string)-2) and len(item_from_queue) <= 3:
                        continue
                    item_dist = dameraulevenshtein(item_from_queue, string)
                    if string not in item_from_queue and abs(item_dist) >= len(item_from_queue)-1:
                        continue
                    possible_words_dict[item_from_queue] = (self.dictionary[item_from_queue][1],
                                            len(string) - len(item_from_queue), string in item_from_queue, item_from_queue.startswith(string))

                    if (len(string) - len(item_from_queue)) < min_suggest_len:
                        min_suggest_len = len(string) - len(item_from_queue)

                #todo if sc_item is not popular, add it to queue so that we could find something popular from it
                #if written word is substring, dont care about lev_dist

                for matched_item in self.dictionary[item_from_queue][0]:
                    if matched_item not in possible_words_dict:
                        #not interested in shorter possibilities
                        if len(matched_item) <= len(item_from_queue):
                            continue

                        # pointless
                        if matched_item == string:
                            continue

                        # calculate Damerau-Levenshtein distance
                        item_dist = dameraulevenshtein(matched_item, string)

                        if item_dist <= self.max_lev_distance or string in matched_item:
                            if matched_item not in self.dictionary:
                                continue
                            if len(matched_item) < (len(string)-2) and len(matched_item) <= 3:
                                continue

                            if string not in matched_item and abs(item_dist) >= len(matched_item)-1:
                                continue
                            possible_words_dict[matched_item] = (self.dictionary[matched_item][1], item_dist, string in matched_item, matched_item.startswith(string))
                            if item_dist < min_suggest_len:
                                min_suggest_len = item_dist

                            if matched_item not in processed_words and len(string) <= len(matched_item):
                                queue.append(matched_item)
                                processed_words[matched_item] = None


            # we check for deletes only when the item from queue is equal or smaller, if its already larger dont bother
            if len(string) < len(item_from_queue):
                continue

            if (len(string) - len(item_from_queue)) < self.max_lev_distance and len(item_from_queue) > 1:
                for c in range(len(item_from_queue)):  # character index
                    word_without_c = '%s%s' % (item_from_queue[:c], item_from_queue[c + 1:])
                    if word_without_c not in processed_words:
                        queue.append(word_without_c)
                        processed_words[word_without_c] = None

        # return (possible_word, (fs, lev_distance)):
        as_list = possible_words_dict.items()
        outlist = sorted(as_list,
                         key=lambda x:
                         (int(x[1][0]) - (abs(x[1][1]) * self.threshold_deduct) +
                         (int(x[1][2]) * self.threshold_add) + (int(x[1][3]) * 2 * self.threshold_add)),
                         reverse=True)

        return outlist


    def autocomplete_search(self, word):
        results = []
        possible_words = self.get_possible_words(word)
        #results = possible_words
        results = [x[0] for x in possible_words[0:25]]
        return results

if __name__ == '__main__':
    # threshold_deduct = 10000
    # threshold_add = 100000
    # as_list = [('envision', ('1225007', -1, False)), ('nison', ('15481', 2, False)), ('environet', ('19131', -2, True)), ('environments', ('13331567', -5, True)), ('environmen', ('130857', -3, True)), ('environement', ('35429', -5, True)), ('evison', ('14478', 1, False)), ('enviro', ('576114', 1, False)), ('nevron', ('54852', 2, False)), ('environmentally', ('3610401', -8, True)), ('environics', ('32677', -3, True)), ('dbenvironment', ('14916', -6, True)), ('environme', ('37361', -2, True)), ('ension', ('15773', 2, False)), ('evron', ('23050', 2, False)), ('viron', ('12974', 2, False)), ('environews', ('20736', -3, True)), ('nevon', ('69625', 2, False)), ('nevison', ('21891', 0, False)), ('evros', ('27913', 2, False)), ('nevis', ('4527077', 2, False)), ('vison', ('50332', 2, False)), ('netiron', ('46283', 0, False)), ('envio', ('52330', 2, False)), ('ention', ('14798', 2, False)), ('eniro', ('46588', 2, False)), ('environmental', ('81596309', -6, True)), ('envios', ('29734', 1, False)), ('nevin', ('236998', 2, False)), ('envis', ('33946', 2, False)), ('aviron', ('14479', 2, False)), ('neron', ('17901', 2, False)), ('enron', ('3048638', 2, False)), ('environnement', ('421835', -6, True)), ('environs', ('565770', -1, True)), ('environment', ('101959294', -4, True)), ('enviros', ('34326', 0, False)), ('netio', ('29098', 2, False)), ('environ', ('3870498', 0, True)), ('etron', ('50676', 2, False)), ('environm', ('62219', -1, True)), ('everon', ('15546', 2, False)), ('envir', ('182940', 2, False))]
    # outlist = sorted(as_list, key=lambda x: (int(x[1][0]) - (abs(x[1][1]) * threshold_deduct) + (int(x[1][2]) * threshold_add)), reverse=True)
    # results = [x[0] for x in outlist[0:5]]
    # print results

    fs = MyFuzzySearch(max_lev_distance=2)
    #print dameraulevenshtein('environ', 'environment')
    #print dameraulevenshtein('environment', 'environ')
    #fs.create_dictionary('/home/ec2-user/word_search.tsv')
    fs.create_dictionary('/Users/sudhi/Downloads/word_search.tsv')
    #print fs.dictionary['environ']
    #print fs.dictionary['environme']
    #tokens = spacy_tokenize(sample_text)
    test_word = 'arch'
    #test_word = 'environ'
    print('getting results...')
    print(fs.autocomplete_search(test_word)[0:50])

    print('results provided')