import pandas as pd
import csv
import numpy as np
import nltk
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import seaborn as sb
import os
from bs4 import BeautifulSoup

def read_all_files():
    for dir in os.listdir("./out"):
        print(dir)
        raw_filesNames = os.listdir("./out/"+ dir)

        i=1
        for name in raw_filesNames:
            file = open("./out/"+dir+"/"+name,encoding="utf8", errors='ignore')
            raw_contents = file.read()
            soup = BeautifulSoup(raw_contents, 'html.parser')

            content =[]
            p_sets=soup.find_all("p")
            for p in p_sets:
                content.append(p.text)

def contentFromHTML(name):
    dir = name[:2]
    file = open("./out/"+dir+"/"+name,encoding="utf8", errors='ignore')
    raw_contents = file.read()
    soup = BeautifulSoup(raw_contents, 'html.parser')

    content =[]
    p_sets=soup.find_all("p")
    for p in p_sets:
        if p:
            temp_p = (p.text).replace('\n', ' ')
            temp_p = temp_p.replace(',', '  ')
            content.append(temp_p)

    content_out=" ".join(content)
    if len(content_out)>=30000:
        content_out  = content_out[:2000]

    if soup.title:
        return (soup.title.text, content_out)
    else:
        return (" ", content_out)
def read_qrels():
    qrels_file = open("./trec-misinfo-resources/qrels/2020-derived-qrels/misinfo-qrels.3aspects", mode = "r")
    qrels = [line.rstrip() for line in qrels_file]


    # i=0
    # while i <=2000:
    #      i+=1
    #      print(qrels[i], i)
    return qrels


def write_csv():
    csv_file = open("./train.csv", mode='w',  encoding="utf8", errors='ignore')
    fieldnames = ['topic_ID', 'file_name', 'title', 'content', 'credibility']
    writer =  csv.DictWriter(csv_file, fieldnames=fieldnames)

    qrels  = read_qrels()

    writer.writeheader()
    for qrel in qrels:

        qrel = qrel.split()
        topic_ID = qrel[0]
        file_name = qrel[2]
        title, content = contentFromHTML(file_name)
        credibility = qrel[5]

        #print(topic_ID, file_name, title, content, credibility )
        if int(topic_ID) > 15:
            break
        #print(topic_ID, qrel)

        writer.writerow({'topic_ID':topic_ID, 'file_name':file_name, 'title':title, 'content':content, 'credibility':credibility})
def write_test_csv():
    csv_file = open("./test.csv", mode='w',  encoding="utf8", errors='ignore')
    fieldnames = ['topic_ID', 'file_name', 'title', 'content', 'credibility']
    writer =  csv.DictWriter(csv_file, fieldnames=fieldnames)

    qrels  = read_qrels()

    writer.writeheader()
    for qrel in qrels:

        qrel = qrel.split()
        topic_ID = qrel[0]
        file_name = qrel[2]
        title, content = contentFromHTML(file_name)
        credibility = qrel[5]

        #print(topic_ID, file_name, title, content, credibility )
        if (int(topic_ID) > 15) and (int(topic_ID)<=30):
            writer.writerow({'topic_ID':topic_ID, 'file_name':file_name, 'title':title, 'content':content, 'credibility':credibility})
        if int(topic_ID) >= 30:
            break
def write_single_csv():
        csv_file = open("./dataset.csv", mode='w',  encoding="utf8", errors='ignore')
        fieldnames = ['topic_ID', 'file_name', 'title', 'content', 'credibility']
        writer =  csv.DictWriter(csv_file, fieldnames=fieldnames)

        qrels  = read_qrels()

        writer.writeheader()
        for qrel in qrels:

            qrel = qrel.split()
            topic_ID = qrel[0]
            file_name = qrel[2]
            title, content = contentFromHTML(file_name)
            credibility = qrel[5]

            #print(topic_ID, file_name, title, content, credibility )

            writer.writerow({'topic_ID':topic_ID, 'file_name':file_name, 'title':title, 'content':content, 'credibility':credibility})
#write_csv()
#write_test_csv()
write_single_csv()
