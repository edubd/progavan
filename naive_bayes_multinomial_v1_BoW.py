# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 14:46:13 2026

@author: eduar
"""
import csv

class NaiveBayesMultinomial:
    """
    Classificador Naïve Bayes Multinomial para classificação de texto
    implementação da turma de Programação Avançada - ENCE 2026.1
    """
    
    def __init__(self):
        self.n = 0        # total de documentos da base de dados (BD) de treino
        self.V = set()    # vocabulário da BD
        self.d = 0        # tamanho do vocabulário (total de features ou tokens ou dimensões)
        self.Y = []       # lista com a classe de cada observacao
        self.bow = []     # representação Bag of Words da BD de treino
        self._simbolos = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        self.classes = set() # classes do problema
    
    
    def importar(self, bd_treino, pos_classe = 0, pos_texto = 1, 
                 cabecalho = True, sep = ",", aspas = '"', 
                 codificacao='utf-8'):
        """ importa uma base de treino no formato CSV p/ uma Bag of Words """

        with open(bd_treino, encoding = codificacao) as arq_csv:
            csv_reader = csv.reader(arq_csv, 
                                    delimiter = sep, 
                                    quotechar = aspas)
            
            # pula a linha de cabeçalho, caso exista
            if cabecalho: csv_reader.__next__()
            
            # loop sobre todas as linhas da BD
            for linha in csv_reader:
                texto = linha[pos_texto]   # pega o texto
                classe = linha[pos_classe] # pega a classe
                # print(texto, classe) # debug
                
                # gera a bow
                bow_atu = {}
                texto = texto.lower() # converte texto para minúsculo
                texto = texto.translate(str.maketrans("", "",  self._simbolos)) # remove pontuações
                tokens = texto.split()
                
                for token in tokens:
                    if token not in bow_atu: bow_atu[token] = 1
                    else: bow_atu[token] += 1
                    
                    if token not in self.V: self.V.add(token)
                
                # adiciona bow atual à bow geral
                self.bow.append(bow_atu)
                
                # adiciona classe atual à Y
                self.Y.append(classe)
                
            # gera os atributos n, d e classes
            self.n = len(self.bow)
            self.d = len(self.V)
            self.classes = set(self.Y)
        
        
    def head(self, N=5):
        """ retorna as N primeiras BoWs """
        if self.n > N:
            return self.bow[:N], self.Y[:N]
        else:
            return self.bow[:self.n], self.Y[:self.n]
            
        
    def tail(self, N=5):
        """ retorna as N últimas BoWs """
        if self.n > N:
            return self.bow[-N:], self.Y[-N:]
        else:
            return self.bow[-self.n:], self.Y[-self.n:]
            
            

if __name__ == "__main__":
    modelo = NaiveBayesMultinomial()
    
    modelo.importar('review_filmes.csv', pos_texto=0, pos_classe=1)
    #modelo.importar('sentiment140.csv')
    #modelo.importar('Train3Classes.csv', pos_texto=1, pos_classe=3)
    #modelo.importar('documents.csv', pos_texto=2, pos_classe=0)
    #modelo.importar('HateBR.csv', pos_texto=1, pos_classe=5)
    #modelo.importar('dataset_sentimentos_augmented.csv', pos_texto=1, pos_classe=2)
    #modelo.importar('dataset_label_pos_neg.csv', pos_texto=1, pos_classe=2)
    
    print('número de observacoes:', modelo.n)
    print('tamanho do vocabulário:', modelo.d)
    print('classes:', modelo.classes)
    print(modelo.head(2))
    print(modelo.tail())
    
    
    
        
