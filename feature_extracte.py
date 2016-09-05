#!/usr/bin/env python

import sys
import pickle
from tf import tf_idf,tfidf_similarity

file = open(sys.argv[1],"r")
#test_file = open(sys.argv[2], "r")
to_write =open(sys.argv[2],"w+")



feature_map = {}
feature_index = 0


HASH_SIZE = 1000000






def process_id_feature(prefix, id):
	global feature_map
	global feature_index

	str = prefix + "_" + id
	if str in feature_map:
		return feature_map[str]
	else:
		feature_index = feature_index + 1
		feature_map[str] = feature_index
	return feature_index


def hash_feature(prefix, id):
	str = prefix + "_" + id
	return hash(str)% HASH_SIZE


def extract_feature(seg):
	list = []
   
        list.append(process_id_feature("url",seg[1]))

        list.append(process_id_feature("ad",seg[2]))

        list.append(process_id_feature("ader",seg[3]))

        list.append(process_id_feature("depth",seg[4]))

        list.append(process_id_feature("pos",seg[5]))

        list.append(process_id_feature("query",seg[6]))

        list.append(process_id_feature("keyword",seg[7]))

        list.append(process_id_feature("title",seg[8]))

        list.append(process_id_feature("desc", seg[9]))

        list.append(process_id_feature("user",seg[10]))
	
        
	
	return list


def extract_feature2(seg):

	depth = float(seg[4])
	pos = float(seg[5])
	relative_pos = int(pos*10/depth)
	return process_id_feature("pos_ratio", str(relative_pos))

def extract_combination_feature(seg):
	list = []
	if(len(seg) >= 16):
		#str = seg[2] + "_" + seg[15]
		list.append(process_id_feature("gender", seg[15]))
         	
		#str1 = seg[15] + "_" + seg[16]
		list.append(process_id_feature("age", seg[16]))
         
		#str2 = seg[10] + "_" +seg[16]
		#list.append(process_id_feature("user_age", str2))
         
	#str3 = seg[2] + "_" + seg[6]
	#list.append(process_id_feature("query_ad", str3))

	#str4 = seg[2] + "_" + seg[10]
	#list.append(process_id_feature("ad_user",str4))
    
		#str5 = seg[6] + "_" + seg[10]
		#list.append(process_id_feature("query_user",str5))

		#str6 = seg[6] + "_" + seg[9]
		#list.append(process_id_feature("query_desc",str6))
	return list

def extract_numerical_feature(seg):
	list = []
	
	num_query = len(seg[11].strip().split("|"))
	num_keyword = len(seg[12].strip().split("|"))	
	num_title = len(seg[13].strip().split("|"))
	num_description = len(seg[14].strip().split("|"))
	list.append(str(process_id_feature("num_query", " ")) + ":" + str(num_query))
	list.append(str(process_id_feature("num_keyword", " ")) + ":" + str(num_keyword))
	list.append(str(process_id_feature("num_title", " ")) + ":" + str(num_title))
	list.append(str(process_id_feature("num_description", " ")) + ":" + str(num_description))
	
	corpus = seg[11:15]
	tfidf = tf_idf(corpus)

    
	query_similar_keyword = tfidf_similarity(tfidf[0],tfidf[1])
	query_similar_tile = tfidf_similarity(tfidf[0],tfidf[2])
	query_similar_description = tfidf_similarity(tfidf[0],tfidf[3])
	keyword_similar_title = tfidf_similarity(tfidf[1],tfidf[2])
	keyword_similar_description = tfidf_similarity(tfidf[1],tfidf[3])
	title_similar_description = tfidf_similarity(tfidf[2],tfidf[3])

	list.append(str(process_id_feature("query_similar_keyword", " ")) + ":" + str(query_similar_keyword) )
	list.append(str(process_id_feature("query_similar_tile", " ")) + ":" + str(query_similar_tile) )
	list.append(str(process_id_feature("query_similar_description", " ")) + ":" + str(query_similar_description ) )
	list.append(str(process_id_feature("keyword_similar_title", " ")) + ":" + str(keyword_similar_title ) )
	list.append(str(process_id_feature("keyword_similar_description", " ")) + ":" + str(keyword_similar_description ) )
	list.append(str(process_id_feature("title_similar_description", " ")) + ":" + str(title_similar_description ) )
    
	#list.append(str(process_id_feature("sum_idf_query", " ")) + ":" + str(sum(tfidf[0]) ) )
	#list.append(str(process_id_feature("sum_idf_keyword", " ")) + ":" + str( sum(tfidf[1]) ))
	#list.append(str(process_id_feature("sum_idf_title", " ")) + ":" + str(sum(tfidf[2]) ) )
	#list.append(str(process_id_feature("sum_idf_description", " ")) + ":" + str(sum(tfidf[3]) ) )

    
    
	depth = float(seg[4])
	postion = float(seg[5])
	relative_pos = float((depth-postion)*10.0/depth)
                  
	#list.append(str(process_id_feature("depth_num", " ")) + ":" + str(depth))
	#list.append(str(process_id_feature("postion_num", " ")) + ":" + str(postion))
	list.append(str(process_id_feature("relative_pos_num", " ")) + ":" + str(relative_pos))
    
	"""
	raw_query = int(seg[6])
	raw_user = int(seg[10])

	list.append(str(process_id_feature("raw_query"," ")) + ":" + str(raw_query))
	list.append(str(process_id_feature("raw_user"," ")) + ":" + str(raw_user))
	"""
    

	return list



def cate_to_str(label, list):
	line = label
	for i in list:
		line = line + "\t" + str(i) + ":1"
	return line

def numer_to_str(numer_list):
	return "\t".join(numer_list)

for line in file:

	seg = line.strip().split("\t")
	list_cate = extract_feature(seg)
	list_cate.append(extract_feature2(seg))
	list_cate.extend(extract_combination_feature(seg))
	list_numer = extract_numerical_feature(seg)
    
        to_write.write(cate_to_str(seg[0], list_cate) + "\t" + numer_to_str(list_numer) + "\n")

to_write.close()
