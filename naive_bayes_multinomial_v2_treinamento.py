# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:44:53 2026

@author: ecgrj
"""
import csv 


class NaiveBayesMultinomial:
    """
    Classificador Naïve Bayes Multinomial para classificação de texto
    implementação da turma de Programação Avançada - ENCE 2026.1
    
    
    Atributos:
    ----------
    _n: int
        total de documentos da base de treinamento

    _V: lista
        vocabulário - lista contendo todas as palavras da base de treinamento 
        armazenadas em ordem alfabética
    
    _d: int
        tamanho do vocabulário    

    _q: int
        número de classes

    _simbolos: string
        relação de caracteres que serão descartados no processo de tokenização

    _bow: lista 2d
        representação Bag of Words da base de treinamento

    _Y: lista
        lista contendo as classes de cada documento de treinamento
    
    _classes: lista
        rótulos de classe do problema (ex: "negative", "positive"), armazenados
        em ordem alfabética
    
    _freqs_classes: dicionário
        dicionário que armazena a contagem de frequência de cada classe
        Ex.: {"negative": 3, "positive": 2}

    _freqs_tokens: dicionário
        dicionário que armazena a contagem de frequência de cada token
        para cada classe. Junto com "_freqs_classes" forma o modelo NB !!!

    """     
    def __init__(self):
        """ 
        o método construtor apenas inicializa os atributos, todos definidos 
        com o modificador de acesso "protegido" (nomes iniciados por _) 
        """
        self._n = 0 
        self._V = []
        self._d = 0
        self._q = 0
        self._simbolos = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        self._bow = []
        self._Y = [] 
        self._classes = set()
        self._freqs_classes = {}
        self._freqs_tokens = {}


        """
        self._map_classes = {}
        
        #propriedades (modelo)
        self._apriori = {}   #probs apriori de cada classe
        self._conds = {}     #lista 2D com todas as probs condicionais P(X|y)
        self._denominador = {}
        """

    
    def _tokeniza(self, texto):
        """ 
        retorna a lista de tokens de um texto (string) passado como parâmetro
        
        Na implementação atual, os símbolos serão descartados e apenas letras
        e números serão considerados tokens (modifique se desejar!)

        Parâmetros:
        -----------
        texto: string
            texto de onde será gerada a lista de tokens

        Retorno
        -------
        tokens : lista
            lista contendo os tokens do texto

        """
        # converte texto para minúsculo
        texto = texto.lower() 
        
        # remove os símbolos contidos no atributo _simbolos
        texto = texto.translate(str.maketrans("", "",  self._simbolos)) 
        
        # retorna a lista de tokens
        return texto.split()
        

    def importar(self, bd_treino, pos_classe = 0, pos_texto = 1, 
                 cabecalho = True, sep = ",", aspas = '"', codificacao='utf-8'):
        """ importa base de dados de treinamento no formato CSV para uma
            Bag of Words em memória        

        Parâmetros:
        -----------
        bd_treino: string
            caminho da base de dados CSV de treinamento. A base pode
            conter um número qualquer de colunas
        
        pos_classe: int, default = 0
            posição da coluna que contém o atributo classe no arquivo CSV
            (0 se for a 1a coluna, 1 se for a 2a coluna etc.) 
            
        pos_texto: int, default = 1
            posição da coluna que contém os textos no arquivo CSV
            (0 se for a 1a coluna, 1 se for a 2a coluna etc.) 
            
        cabecalho: bool, default = True
            indica se arquivo CSV possui linha de cabeçalho 
            (True = possui, False = não possui)

        sep: string, default = ","
            caractere utilizado como separador no arquivo CSV 

        aspas: string, default = "
            caractere utilizado que arquivo CSV utiliza como aspas nos textos
            que contém o separador (a maioria usa ", mas alguns usam ')
            
        codificacao: string, default = "utf-8"
            encoding do arquivo CSV. Na maioria das vezes é "utf-8", porém
            pode também ser "latin_1", "windows-1252", "ansi", entre outros
            (se a importação falhar, tente uma das outras opções acima) 

        """
        # (1). abre base de dados usando o pacote 'csv'
        # tutorial sobre esse pacote: https://realpython.com/python-csv/        
        with open(bd_treino, encoding = codificacao) as arq_csv:
            csv_reader = csv.reader(arq_csv, 
                                    delimiter = sep, 
                                    quotechar = aspas)
            
            # (1.1) pula a linha de cabeçalho, caso exista
            if cabecalho: csv_reader.__next__()
            
            # (2) loop sobre todas as linhas da BD
            self._V = set()
            self._bow = []
            self._Y = []
            for linha in csv_reader:
                texto = linha[pos_texto]   # pega o texto
                classe = linha[pos_classe] # pega a classe
                # print(texto, classe) # debug
                
                # (3) gera a Bag of Words do texto atual (bow_atu) 
                #     e a adiciona à BoW geral (_bow)
                bow_atu = {}
                tokens = self._tokeniza(texto)
                
                for token in tokens:
                    if token not in bow_atu: bow_atu[token] = 1
                    else: bow_atu[token] += 1
                    
                    self._V.add(token)
                
                # (3.1) adiciona bow atual à bow geral (_bow)
                self._bow.append(bow_atu)
                
                # (3.2) adiciona classe atual à _Y
                self._Y.append(classe)
                
            # (4) gera os atributos _n, _d, _classes e _q
            self._n = len(self._bow)
            self._d = len(self._V)
            self._classes = list(set(self._Y))
            self._classes.sort() # apenas para ordenar alfabeticamente
            self._q = len(self._classes)
            
            # (5) converte V para uma lista e ordena alfabeticamente
            self._V = list(self._V)
            self._V.sort()
            

    def treinar(self):
        """ 
        realiza o treinamento do modelo NaiveBayesMultinomial a partir
        de _bow (Bag of Words dos textos) e _Y (classes dos textos)
        
        obviamente, para treinar o modelo é preciso antes ter importado
        alguma base de dados de treinamento
        """
        
        # (1) gera _freqs_classes (frequências para cada classe)
        
        # (1.1) inicializa a estrutura
        self._freq_classes = {}
        for classe in self._classes: self._freqs_classes[classe] = 0 

        # (1.2) contabiliza as frequências
        for classe in self._Y: self._freqs_classes[classe] += 1
    
        # (2) gera _freqs_tokens (frequências de cada token nos documentos de
        #                         cada classe)
        
        # (2.1) inicializa a estrutura
        self._freqs_tokens = {}
        for classe in self._Y: self._freqs_tokens[classe] = {}
        
        for classe in self._classes: 
            for token in self._V:
                self._freqs_tokens[classe][token] = 0
        
        # (2.2) contabiliza as frequências
        for i in range(self._n):
            bow    = self._bow[i] # pega a i-ésima BoW
            classe = self._Y[i]   # pega a classe da i-ésima BoW
            
            # para cada token da BoW atual, faz a atualização da
            # contagem de frequências
            for token in bow: 
                freq_token = bow[token]
                self._freqs_tokens[classe][token] += freq_token
    

# código de teste
if __name__ == '__main__':
    
    # instancia um objeto da classe NaiveBayesMultinomial
    modelo = NaiveBayesMultinomial()

    # importa um dataset para a memória (gera BoW e Y)
    nome_dataset = "review_filmes.csv"
    modelo.importar(nome_dataset, pos_classe = 1, pos_texto = 0)
    
    print('vocabulário: ', modelo._V, ' ------ tamanho:', len(modelo._V))
    print('-' * 60)
    print("-> BoW: ", modelo._bow)
    print("-> Y: ", modelo._Y)
    print('-' * 60)

    # treina o modelo
    modelo.treinar()
    print("* * * modelo:")
    print('-> freqs_classes:', modelo._freqs_classes)
    print('-> freqs_tokens:', modelo._freqs_tokens)
    
    