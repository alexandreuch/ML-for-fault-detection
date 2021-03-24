import re
import os
os.system('cls' if os.name == 'nt' else 'clear')
import matplotlib.pyplot as plt
import time
import sklearn
import numpy as np
import pandas as pd
from pydub import AudioSegment
from scipy import stats
from scipy.fftpack import fft
import scipy.stats
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.io import wavfile

def atoi(text):
    """
    Return the char converted to integer, if it's a digit,
    otherwise, return the char.
    :param text: the string to be analysed.
    :return: char converted or char
    """
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def load_audios_from_folder(folder_path):
    """
    Generates a list with all audio file names,
    order the list accordingly with the number
    in the file name and add the path folder to
    each file name.
    :param folder_path: folder path
    :return: list_folder_path_plus_files_in_dir
    """

    # list with the files contained in the directory or folder
    list_files_in_dir = os.listdir(folder_path)
    # print('List of files in folder: ', list_files_in_dir)

    # order the list
    list_files_in_dir.sort(key=natural_keys)
    # print('Sorted list: ', list_files_in_dir)

    # list with the folder path plus file name ordered
    list_folder_path_plus_files_in_dir = [folder_path + file_name for file_name in list_files_in_dir]
    # print('Path plus folder:', list_folder_path_plus_files_in_dir)

    return list_folder_path_plus_files_in_dir

def read_audio_using_audiosegment(audio_file_path):
    """
    Read audio file using AudioSegment from PyDub
    :param audio_file_path: audio file path
    :return: audio_samples
    """

    audio_samples = AudioSegment.from_file(audio_file_path)
    # print('Audio segment sample rate: ', audio_samples.frame_rate)
    #audio_length = len(audio_samples)
    # print('Length of audio file - audio segment object (in milliseconds):\n', audio_length)
    # print('Number of channels: ', audio_samples.channels)

    return audio_samples

def audio_samples_and_classes(folder_path):
    audios_list = load_audios_from_folder(folder_path)
    audio_class_list = []
    audio_samples_list_of_lists = []
    
    for audio_file_path in audios_list:
        sample_rate, audio_signal_data = wavfile.read(audio_file_path)
        
        try:
            number_channels_audio_data = audio_signal_data.shape[1]

            if number_channels_audio_data:
                audio_data_channel_L = audio_signal_data[:, 0]
                audio_samples_list_of_lists.append(audio_data_channel_L)

        except IndexError:
                audio_samples_list_of_lists.append(audio_signal_data)

        audio_file_path_split_content = natural_keys(audio_file_path)
        audio_class = audio_file_path_split_content[5]
        audio_class_list.append(audio_class)    

    assert (len(audio_samples_list_of_lists) > 2) and (0 in audio_class_list) and (1 in audio_class_list)

    audio_samples_normalized = Z_and_norm(audio_samples_list_of_lists)
    
    return audio_samples_normalized,audio_class_list

def Z_and_norm(audio_samples_list_of_lists):
    audio_inst = []
    for i in range(0,len(audio_samples_list_of_lists)):
        amp = stats.zscore(audio_samples_list_of_lists[i])
        amp = np.abs(fft(amp))
        audio_inst.append(amp)
    
    return audio_inst

def silhueta(array_db, array_class):
    default_metrics = metrics.silhouette_samples(array_db, array_class)

    remove = []
    for i in range(0, len(default_metrics)):
        if(default_metrics[i] < -0.04): remove.append(i)
    
    #print(remove)
    return remover(array_db,array_class,remove)

def remover(array_db,array_class,array_remover):
    if(len(array_remover) > 0):
        bff_db = []
        bff_clss = []
        for i in range(0,len(array_db)):
            if(i not in (array_remover)):
                bff_db.append(array_db[i])
                bff_clss.append(array_class[i])
        return bff_db,bff_clss
    
    else:
        return array_db,array_class

def MAD(array_db,array_class):
    media = 0
    zero_elements = 0
    one_elements = 0

    for i in range(0,len(array_class)):
        media += np.mean(array_db[i])
        if(array_class[i] == 0): 
            zero_elements+=np.mean(array_db[i])
        else:
            one_elements+=np.mean(array_db[i])
    
    media/=len(array_db)
    zeros = zero_elements/array_class.count(0)
    ones = one_elements/array_class.count(1)
    limiar = abs(zeros-ones) 

    """
    print("Media:",media)
    print("Ones:",ones)
    print("Zeros:",zeros)
    print("Limiar:",limiar) 
    """

    cont = 0
    buffer = 0
    element = 0
    remove = []
    sub_array_len = len(array_db[i]) #88200
    len_arr_db = len(array_db)

    for i in range(0,sub_array_len):
        while cont < len_arr_db: 
            buffer += array_db[cont][element]
            cont+=1  
        buffer /= len_arr_db
        if(buffer < limiar): remove.append(element)
        element += 1
        cont = 0
        buffer = 0
    
    remove = sorted(remove,reverse=True)
    mascara = np.ones(len(array_db[0]))
    mascara[remove] = 0
    
    vet = []
    result = []
    for i in range(0,len(array_db)):
        for j in range(0,len(mascara)):
            if(mascara[j] == 1): vet.append(array_db[i][j])
        result.append(vet)
        vet=[]

    return(result)
    
def results1NN(X_test,Y_test,X_train,Y_train, tipo):
    kNN_classifier = KNeighborsClassifier(n_neighbors=1)
    kNN_classifier.fit(X_train, Y_train)

    labels_prediction_test_data = kNN_classifier.predict(X_test)
    
    kNN_accuracy_score = accuracy_score(Y_test, labels_prediction_test_data)
    recall_score = sklearn.metrics.recall_score(Y_test,labels_prediction_test_data)
    precision_score = sklearn.metrics.precision_score(Y_test, labels_prediction_test_data)

    print("Accuracy: %.4f" % kNN_accuracy_score)
    print("Precision: %.4f" % precision_score)
    print("Recall Score: %.4f" % recall_score)

def main():
    folder_path_v3 = '/home/pingulino/Alexandre/Thanos/segmented_database_1_v3/'
    folder_path_v4 = '/home/pingulino/Alexandre/Thanos/segmented_database_1_v4/'
    
    audio_inst_v3,audio_class_list_v3 = audio_samples_and_classes(folder_path_v3)
    audio_inst_v4,audio_class_list_v4 = audio_samples_and_classes(folder_path_v4)

    X_test = audio_inst_v3
    Y_test = audio_class_list_v3
    
    X_Default_test = audio_inst_v4
    Y_Default_test = audio_class_list_v4
 
    print("\n--------------------------------------------")
    print("Quantidade de elementos para teste:", len(Y_test))
    print("Quantidade de elementos para treino:", len(Y_Default_test))
    print("--------------------------------------------")

    print("                   Default\n")
    inicio = time.time()
    print("Quantidade de elementos:", len(Y_Default_test))
    results1NN(X_test,Y_test,X_Default_test,Y_Default_test,"Default")
    fim = time.time() - inicio
    print("Tempo por elemento: %.4fs" %(fim/(len(Y_Default_test))))
    print("Tempo de execução: %.4fs" %fim)
    print("--------------------------------------------")
    
    print("                   Silhueta\n")
    inicio = time.time()
    X_silhueta, Y_Silhueta = silhueta(X_Default_test,Y_Default_test)
    print("Quantidade de elementos:", len(Y_Silhueta))
    results1NN(X_test,Y_test,X_silhueta,Y_Silhueta,"Silhueta")
    fim = time.time() - inicio
    print("Tempo por elemento: %.4fs" %(fim/(len(Y_Default_test))))
    print("Tempo de execução: %.4fs" %fim)
    print("--------------------------------------------")
    
    print("                      MAD\n")
    corte = len(audio_inst_v4)
    X_all = audio_inst_v4 + audio_inst_v3
    Y_all = audio_class_list_v4 + audio_class_list_v3
    inicio = time.time()
    X_mad = MAD(X_all,Y_all)
    X_mad_teste = X_mad[corte:]
    X_mad_treino = X_mad[:corte]
    print("Quantidade de elementos:", len(Y_Default_test))
    results1NN(X_mad_teste,Y_test,X_mad_treino,Y_Default_test,"MAD")    
    fim = time.time() - inicio
    print("Tempo por elemento: %.4fs" %(fim/(len(Y_Default_test))))
    print("Tempo de execução: %.4fs" %fim)
    print("--------------------------------------------")
   
    print("                Silhueta + MAD\n")
    inicio = time.time()
    X_silhueta, Y_Silhueta = silhueta(X_Default_test,Y_Default_test)
    new_x = X_silhueta + X_test 
    new_y = Y_Silhueta + Y_test 
    X_madSil = MAD(new_x,new_y)
    new_corte = len(X_silhueta)
    X_ms_teste = X_madSil[new_corte:]
    X_ms_treino = X_madSil[:new_corte]
    print("Quantidade de elementos:", len(Y_Silhueta))
    results1NN(X_ms_teste,Y_test,X_ms_treino,Y_Silhueta,"Silhueta + MAD")
    fim = time.time() - inicio
    print("Tempo por elemento: %.4fs" %(fim/(len(Y_Default_test))))
    print("Tempo de execução: %.4fs" %fim)
    print("--------------------------------------------")
 
if __name__ == "__main__":
    main()