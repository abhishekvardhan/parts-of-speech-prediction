import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict 
import math
from nltk.tokenize import word_tokenize
ans=[]
app = Flask(__name__)
def transission_mat(tag_tag_count,tag_count,tags_l,alpha,vocab_l):
    numtags=len(tags_l)
    A=np.zeros((numtags,numtags))
    c=0
    k=0
    for i in range(numtags):
        c=tag_count[tags_l[i]]+(numtags*alpha)
        for j in range(numtags):
            if (tags_l[j],tags_l[j]) in tag_tag_count.keys():
                    k=tag_tag_count[(tags_l[i],tags_l[j])]
            #print((k+alpha)/c)
            A[i,j]=(k+alpha)/c
            k=0
    return A
def emmision_mat(word_tag_count,vocab_l,alpha,tags_l,tag_count,vocab):
        B=np.zeros((len(vocab),len(tags_l)))
        k=0
        for i in range(len(vocab_l)):
            for j in range(len(tags_l)):
                if (vocab_l[i],tags_l[j]) in word_tag_count.keys():
                    k=word_tag_count[(vocab_l[i],tags_l[j])]
                    
                t=tag_count[tags_l[j]]+(alpha*len(vocab))
                B[i,j]=(k+alpha)/t
                k=0
        return B
def viterbi_ini(A,B,tags_l,vocab_l,scent):
        c=np.zeros((len(tags_l),len(scent)))
        st="--init--"
        st_word=scent[0]
        for i in range(len(tags_l)):
            c[i,0]=math.log(A[st][tags_l[i]])+math.log(B[st_word][tags_l[i]])
        c=pd.DataFrame(c, index=tags_l, columns = scent )
        return c
def viterbi_back(A,B,c,d,tags_l,vocab_l,scent):
        max_prob=float("-inf")
        pos=[]

        pos.append(c[scent[-1]].idxmax(axis = 0, skipna = True))
        for i in range(len(scent)-1,-1,-1):
            pos.append(d[scent[i]][pos[-1]])
        pos=pos[::-1]
        pos=pos[1:]
        #print(len(pos),len(scent))
        for i in range(len(pos)):
            print(scent[i],pos[i])
        #print(pos)
        return pos
def viterbi_for(A,B,tags_l,vocab_l,scent):
        d=np.zeros((len(tags_l),len(scent)))
        d=pd.DataFrame(d, index=tags_l, columns = scent )
        c=viterbi_ini(A,B,tags_l,vocab_l,scent)
        
        for i in range(1,len(scent)):
            
            for j in range (len(tags_l)):
                max_prob=float("-inf")
                for k in range(len(tags_l)): 
                    prob=c[scent[i-1]][tags_l[k]]+math.log(A[tags_l[k]][tags_l[j]]) +math.log(B[scent[i]][tags_l[j]])
                    if max_prob<prob:
                        max_prob=prob
                        path=k
                d[scent[i]][tags_l[j]]=tags_l[path]
                c[scent[i]][tags_l[j]]=max_prob
        
        k=viterbi_back(A,B,c,d,tags_l,vocab_l,scent)
        
        return k
def pos_tagging(sentence,vocab_l,A,B,tags_l):
    scent=word_tokenize(sentence)
    for i in scent:
        if i not in vocab_l:
            return -1
    else:
        return viterbi_for(A,B,tags_l,vocab_l,scent)

@app.route('/')
def home():
    return render_template('index.html')
@app.template_global(name='zip')
def _zip(*args, **kwargs): 
    return __builtins__.zip(*args, **kwargs)

@app.route('/predict',methods=['POST'])
def predict():
    
    with open("./data/WSJ_02-21.pos", 'r') as f:
        R=f.readlines() 
    vocab=set()
    word_tag_count=defaultdict(int)
    tag_tag_count=defaultdict(int)
    tag_count=defaultdict(int)
    num_lines=0
    sent=[]
    prev="--init--"
    back="--back--"
    for i in R:
        if not i.split():
            tag_tag_count[(prev,back)]+=1
            prev="--init--"
            num_lines+=1
        else:
            word,tag=i.split()
            vocab.add(word)
            sent.append(word)
            word_tag_count[(word,tag)]+=1
            tag_tag_count[(prev,tag)]+=1
            tag_count[tag]+=1
            prev=tag 

    vocab_l=sorted(list(vocab))
    vocab_d={}
    for i,word in enumerate(sorted(vocab_l)):
        vocab_d[word]=i
    tag_count["--init--"]=0
    tags_l=sorted(list(tag_count.keys()))


    alpha=0.001

    A=transission_mat(tag_tag_count,tag_count,tags_l,alpha,vocab_l)
    A = pd.DataFrame(A, index=tags_l, columns = tags_l )

    for i in tags_l:
        A[i]["--init--"]=4.175880e-08

    B=emmision_mat(word_tag_count,vocab_l,alpha,tags_l,tag_count,vocab)
    B = pd.DataFrame(B, index=vocab_l, columns = tags_l )
    B=B.transpose()
    str1 = [x for x in request.form.values()]
    scent=pos_tagging(str1[0],vocab_l,A,B,tags_l)
    if scent!= -1:
        sl=str1[0].split()
        return render_template('index.html', prediction_text=zip(scent,sl))
    else:
        return render_template('index.html', prediction_text="Error")


if __name__ == "__main__":

    app.run(debug=True)