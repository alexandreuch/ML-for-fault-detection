# KNN for fault detection :computer:

Esse código foi baseado no artigo Detection of defective embedded bearings by sound analysis: a machine learning approach - Mario 2014. Nele o autor 
testa filtros em bases de dados para tentar melhorar a eficácia de detecção de erros em engrenagens usando no final classificador 1NN.

Primeiramente coleta-se a base A. Após aplicam-se os filtros formando novas bases e verificando se eles conseguem melhorar a acuracia do método.
Para realizar meus testes coletei as amostras de áudio em uma pasta, cada arquivo de áudio seguia o seguinte formato: sweep_C_vol_A_lab_2s_N.wave,
onde C é a classe (0 para não-ruidos e 1 para ruidosos), A refere-se ao volume e N a numeração da partiçao de áudio.

Os algoritmos funcionam da seguinte forma. Todas as amostras de áudio são em duas listas, uma com seus valores com Fast Fourier Transform Z-Normalizado e a outra com a classe de cada um segmento. Seguindo o código faz quatro testes com método o 1NN, sendo a 1° Utiliza apenas a base padrão sem filtragem, a 2° modifica a base com no coeficiente de Silhuetas,a 3° Modifica a base com o MAD,e a 4° Modifica a base aplicando o coeficiente de silhuetas + MAD.
A diferença entre o fault-detecton e fault-2b-detecton são as entradas. O primeiro aceita apenas uma base de dados, e usa a variavel corte para definir a base de testes e a de treino. Já o segundo utiliza a primera base como treino e a segunda como teste.
