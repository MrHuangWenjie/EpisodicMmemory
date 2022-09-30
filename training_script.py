import os
# import agent
import factory
from audio_monitor import load_audio
import numpy as np
import torch
import cv2
import struct
import wave
import pyaudio
import wandb
from agent import Visual, Auditory
import librosa
from audiomentations import Compose, AddGaussianNoise, PitchShift, HighPassFilter
import soundfile as sf

device0 = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def sort_keys_files_img(name):
    return int(name[5:-4])


def sort_keys_files_audio(name):
    return int(name[5:-4])


def binary2hex(x):
    hexadecimal = []
    for i in range(8):
        temp = x[i*8: i*8+8]
        d = 0
        for j in range(8):
            if temp[j] == 1:
                d += pow(2, 7-j)
        hexadecimal.append(d)
    return hexadecimal


def find_d(x):
    for i in range(len(x)):
        temp = x[i]
        for j in range(len(x)):
            if sum(abs(x[j] - temp)) == 0 and i != j:
                print(j)
        print('########')


def save_rep(filename, array):
    np.savetxt('rep_speech/' + filename, array)


def raw_speech():
    # apple banana book car clock mouse phone sofa lemon toy
    sentences = []
    objects = ['apple', 'banana', 'book', 'car', 'clock', 'lemon', 'mouse', 'phone', 'sofa', 'toy']
    for obj in objects:
        sentences.append('howdoes' + obj + 'move')
        sentences.append('whatelsearearound' + obj)
        sentences.append('whatisthecolorof' + obj)
        # sentences.append('how' + obj + 'move')
        # sentences.append('what' + obj + 'color')
        # sentences.append('whataround' + obj)
    # sentences.append('whatisthis')
    print(len(sentences))
    num = 1
    for file_name in sentences:
        print(file_name)
        if len(file_name) != 0:
            save_file = file_name + '.txt'
            name_len = len(file_name)
            files = os.listdir('recorded_voices/speeches/')
            files.sort()
            speeches_mfcc_list = list()
            speeches_mfcc_list_test = list()
            for file in files:
                if len(file) >= name_len and file[:name_len] == file_name:
                    if file[-9:-4] != '00001' and file[-9:-4] != '00002' and file[-9:-4] != '00003':
                        try:
                            speech = load_audio('./recorded_voices/speeches/' + file)
                        except FileNotFoundError:
                            speech = []
                        print('the shape of speech:', np.shape(speech))
                        x, max_amplitude = hearing.signal2int(speech)
                        x = hearing.get_mfcc(x)
                        speeches_mfcc_list.append(x)
                    elif file[-9:-4] == '00001' or file[-9:-4] == '00002' or file[-9:-4] == '00003':
                        try:
                            speech = load_audio('./recorded_voices/speeches/' + file)
                        except FileNotFoundError:
                            speech = []
                        x, max_amplitude = hearing.signal2int(speech)
                        x = hearing.get_mfcc(x)
                        speeches_mfcc_list_test.append(x)
            print('to encode...')
            speeches_mfcc_list = torch.cat(speeches_mfcc_list, dim=0)
            speeches_mfcc_list_test = torch.cat(speeches_mfcc_list_test, dim=0)
            print(speeches_mfcc_list.size())
            print(speeches_mfcc_list_test.size())
            print(file_name)
            print(num)
            num += 1
            # hearing.encoder = torch.nn.DataParallel(hearing.encoder, device_ids=[1, 0])
            # hearing.encoder.to(device1)

            # rep = list()
            # batch_speeches_mfcc_list = speeches_mfcc_list[:len(speeches_mfcc_list//2)]
            # batch_speeches_mfcc_list = batch_speeches_mfcc_list.to(device1)
            # rep.append(torch.reshape(hearing.encoder(batch_speeches_mfcc_list).round(),
            #                          [len(speeches_mfcc_list), -1]).detach().cpu().numpy())
            # batch_speeches_mfcc_list = speeches_mfcc_list[len(speeches_mfcc_list // 2):]
            # batch_speeches_mfcc_list = batch_speeches_mfcc_list.to(device1)
            # rep.append(torch.reshape(hearing.encoder(batch_speeches_mfcc_list).round(),
            #                          [len(speeches_mfcc_list), -1]).detach().cpu().numpy())
            # rep = np.concatenate(rep, axis=0)
            # speeches_mfcc_list_test = speeches_mfcc_list_test.to(device1)
            rep = torch.reshape(hearing.encoder(speeches_mfcc_list).round(),
                                [len(speeches_mfcc_list), -1]).detach().cpu().numpy()
            rep_test = torch.reshape(hearing.encoder(speeches_mfcc_list_test).round(),
                                     [len(speeches_mfcc_list_test), -1]).detach().cpu().numpy()
            print('to save ...')
            # rep_index = 0
            # save_batch_size = 1000
            # while rep_index < len(rep):
            #     save_rep_portion = rep[rep_index: rep_index + save_batch_size]
            #     np.savetxt('rep_speech/' + save_file + str(rep_index//1000).zfill(3), save_rep_portion)
            np.savetxt('rep_speech/' + save_file, rep)
            np.savetxt('rep_speech/' + save_file[:-4] + '_test.txt', rep_test)


def speech_cat():
    speech_sentences = []
    objects = ['apple', 'banana', 'book', 'car', 'clock', 'lemon', 'mouse', 'phone', 'sofa', 'toy']
    for obj in objects:
        speech_sentences.append('how ' + obj + ' move')
        speech_sentences.append('what ' + obj + ' color')
        speech_sentences.append('what around ' + obj)
    for file_names in speech_sentences:
        speeches_mfcc_list = list()
        save_path = list()
        if len(file_names) != 0:
            words = list()
            audio_files = file_names.strip().split(' ')

            # load the audios of speech words with variants, each word has 5 speech variants
            for audio in audio_files:
                save_path += audio
                word_variants = list()
                for i in range(5):
                    try:
                        tmp = load_audio('./recorded_voices/newWords/' + audio + str(i+1).zfill(3) + '.wav')
                        word_variants.append(tmp)
                    except FileNotFoundError:
                        print('======================== not enough variant of audio ===========================')
                        continue
                words.append(word_variants)

            print(np.shape(words))
            # re-concatenate three words into a speech, 5*5*5=125 speeches in total
            sentences = list()
            for i in range(5):
                for j in range(5):
                    for k in range(5):
                        tmp = list()
                        tmp.append(words[0][i])
                        tmp.append(words[1][j])
                        tmp.append(words[2][k])
                        sentences.append(tmp)
            print('size of sentences:', np.shape(sentences))
            # process words list into a single speech
            for elements in sentences:
                sentence = list()
                for ele in elements:
                    sentence = sentence + ele
                x, max_amplitude = hearing.signal2int(sentence)
                x = hearing.get_mfcc(x)
                speeches_mfcc_list.append(x)

            print('to encode...')
            # encode speeches into representations and save them into speech data
            speeches_mfcc_list = torch.cat(speeches_mfcc_list, dim=0)
            print(speeches_mfcc_list.size())
            rep = torch.reshape(hearing.encoder(speeches_mfcc_list).round(), [len(speeches_mfcc_list), -1]).detach().cpu().numpy()
            np.savetxt('rep_speech/' + ''.join(save_path) + '_.txt', rep)


def concept_rep():
    concepts = ['apple', 'banana', 'book', 'car', 'clock', 'lemon', 'mouse', 'phone', 'sofa', 'toy', 'red', 'yellow', 'silver', 'black']
    for concept in concepts:
        tmp = load_audio('./recorded_voices/newWords/' + concept + '001.wav')
        x, max_amplitude = hearing.signal2int(tmp)
        rep = torch.reshape(hearing.encoder(hearing.get_mfcc(x)).round(), [1, -1]).squeeze(dim=0).detach().cpu().numpy()
        np.savetxt('rep_speech/' + concept + '.txt', rep)


def image_rep():
    files = os.listdir('rep_dete/')
    mouses = list()
    phones = list()
    apples = list()
    lemons = list()
    sofas = list()
    bananas = list()
    books = list()
    clocks = list()
    cars = list()
    toys = list()

    for file in files:
        sub_img = np.load('/home/wenjie/PycharmProjects/episodicMemory/rep_dete/' + file)
        sub_img = np.reshape(sub_img, [64, 64, 5]).transpose([2, 0, 1])
        label = file.split('_')[1]
        if label == 'mouse':
            mouses.append(sub_img)
        elif label == 'phone':
            phones.append(sub_img)
        elif label == 'apple':
            apples.append(sub_img)
        elif label == 'banana':
            bananas.append(sub_img)
        elif label == 'lemon':
            lemons.append(sub_img)
        elif label == 'sofa':
            sofas.append(sub_img)
        elif label == 'book':
            books.append(sub_img)
        elif label == 'clock':
            clocks.append(sub_img)
        elif label == 'car':
            cars.append(sub_img)
        elif label == 'toy':
            toys.append(sub_img)
        else:
            continue
    mouses_rep = torch.reshape(vision.encoder(
        torch.tensor(mouses, dtype=torch.float32)).round().squeeze(dim=0), [len(mouses), -1]).detach().numpy()
    phones_rep = torch.reshape(vision.encoder(
        torch.tensor(phones, dtype=torch.float32)).round().squeeze(dim=0), [len(phones), -1]).detach().numpy()
    apples_rep = torch.reshape(vision.encoder(
        torch.tensor(apples, dtype=torch.float32)).round().squeeze(dim=0), [len(apples), -1]).detach().numpy()
    bananas_rep = torch.reshape(vision.encoder(
        torch.tensor(bananas, dtype=torch.float32)).round().squeeze(dim=0), [len(bananas), -1]).detach().numpy()
    lemons_rep = torch.reshape(vision.encoder(
        torch.tensor(lemons, dtype=torch.float32)).round().squeeze(dim=0), [len(lemons), -1]).detach().numpy()
    sofas_rep = torch.reshape(vision.encoder(
        torch.tensor(sofas, dtype=torch.float32)).round().squeeze(dim=0), [len(sofas), -1]).detach().numpy()
    books_rep = torch.reshape(vision.encoder(
        torch.tensor(books, dtype=torch.float32)).round().squeeze(dim=0), [len(books), -1]).detach().numpy()
    clocks_rep = torch.reshape(vision.encoder(
        torch.tensor(clocks, dtype=torch.float32)).round().squeeze(dim=0), [len(clocks), -1]).detach().numpy()
    cars_rep = torch.reshape(vision.encoder(
        torch.tensor(cars, dtype=torch.float32)).round().squeeze(dim=0), [len(cars), -1]).detach().numpy()
    huaweitoys_rep = torch.reshape(vision.encoder(
        torch.tensor(toys, dtype=torch.float32)).round().squeeze(dim=0), [len(toys), -1]).detach().numpy()
    print(np.shape(mouses_rep))
    np.savetxt('rep_object/mouses.txt', mouses_rep)
    np.savetxt('rep_object/phones.txt', phones_rep)
    np.savetxt('rep_object/apples.txt', apples_rep)
    np.savetxt('rep_object/bananas.txt', bananas_rep)
    np.savetxt('rep_object/lemons.txt', lemons_rep)
    np.savetxt('rep_object/sofas.txt', sofas_rep)
    np.savetxt('rep_object/books.txt', books_rep)
    np.savetxt('rep_object/clocks.txt', clocks_rep)
    np.savetxt('rep_object/cars.txt', cars_rep)
    np.savetxt('rep_object/toys.txt', huaweitoys_rep)


def color_rep():
    files = os.listdir('rep_dete/')
    red = list()
    yellow = list()
    black = list()
    silver = list()

    for file in files:
        sub_img = np.load('/home/wenjie/PycharmProjects/episodicMemory/rep_dete/' + file)
        sub_img = np.reshape(sub_img, [64, 64, 5]).transpose([2, 0, 1])
        color = file.split('_')[2]
        if color == 'red.npy':
            red.append(sub_img)
        elif color == 'yellow.npy':
            yellow.append(sub_img)
        elif color == 'black.npy':
            black.append(sub_img)
        elif color == 'silver.npy':
            silver.append(sub_img)
        else:
            continue
    red_rep = torch.reshape(vision.encoder(
        torch.tensor(red, dtype=torch.float32)).round().squeeze(dim=0), [len(red), -1]).detach().numpy()
    yellow_rep = torch.reshape(vision.encoder(
        torch.tensor(yellow, dtype=torch.float32)).round().squeeze(dim=0), [len(yellow), -1]).detach().numpy()
    black_rep = torch.reshape(vision.encoder(
        torch.tensor(black, dtype=torch.float32)).round().squeeze(dim=0), [len(black), -1]).detach().numpy()
    silver_rep = torch.reshape(vision.encoder(
        torch.tensor(silver, dtype=torch.float32)).round().squeeze(dim=0), [len(silver), -1]).detach().numpy()
    print(np.shape(red_rep))
    np.savetxt('rep_color/reds.txt', red_rep)
    np.savetxt('rep_color/yellows.txt', yellow_rep)
    np.savetxt('rep_color/blacks.txt', black_rep)
    np.savetxt('rep_color/silvers.txt', silver_rep)


def audio_augmentation():
    chunk = 1024
    data_format = pyaudio.paInt16
    channels = 1
    rate = 4096
    record_seconds = 10

    # augment = Compose([AddGaussianNoise(min_amplitude=0.1, max_amplitude=0.2, p=1),
    #                    PitchShift(min_semitones=-8, max_semitones=8, p=1),
    #                    HighPassFilter(min_cutoff_freq=2000, max_cutoff_freq=4000, p=1)])
    augment = Compose([AddGaussianNoise(min_amplitude=0.02, max_amplitude=0.1, p=0.5),
                       PitchShift(min_semitones=-4, max_semitones=4, p=0.5)])

    # p = pyaudio.PyAudio()

    files = os.listdir('recorded_voices/speeches/')
    files.sort()
    for file in files:
        if file[-9:-4] == '00001' or file[-9:-4] == '00002' or file[-9:-4] == '00003' or file[-8] == '_':
            continue
        sig, sr = librosa.load('recorded_voices/speeches/' + file, sr=rate)
        for i in range(10):
            aug_sig = augment(sig, sr)
            sf.write('recorded_voices/speeches/' + file[:-4] + '_' + str(i+200).zfill(3) + '.wav', aug_sig, sr)
    print('audio augmentation is done !')


def shuffle_simul(a, b):
    array_size = len(a)
    sequence = []
    for i in range(array_size):
        sequence.append(i)
    np.random.shuffle(sequence)
    a_mirror = []
    b_mirror = []
    for i in range(len(sequence)):
        a_mirror.append(a[i])
        b_mirror.append(b[i])
    a = np.array(a)
    b = np.array(b)
    return a, b


if __name__ == '__main__':
    hearing = Auditory()
    vision = Visual()
    # >>>> prepare data for training >>>>
    # audio_augmentation()
    # try:
    #     while True:
    #         choice = int(input('please choose speech_rep, concept_rep, img_rep or color_rep with 1, 2, 3 or 4:'))
    #         if choice == 1:
    #             raw_speech()
    #         elif choice == 2:
    #             concept_rep()
    #         elif choice == 3:
    #             image_rep()
    #         elif choice == 4:
    #             color_rep()
    #         elif choice == 5:
    #             speech_cat()
    #         else:
    #             pass
    # except KeyboardInterrupt:
    #     exit(0)

    # >>> prepare the data for training move analyser >>>
    # files = os.listdir('rep_dete/')
    # pairs = []  # size of n, 8192*2, 1, 1
    # move_class = []  # size of n, 4
    # portion_num = 0
    # objects = []
    # for file in files:
    #     sub_img = np.load('/home/wenjie/PycharmProjects/episodicMemory/rep_dete/' + file)
    #     sub_img = np.reshape(sub_img, [64, 64, 5]).transpose([2, 0, 1])
    #     objects.append(sub_img)
    # for filtered_object_index_i in range(len(objects)):
    #     print(filtered_object_index_i)
    #     for filtered_object_index_j in range(filtered_object_index_i, len(objects)):
    #         move_class_instance = [0, 0, 0, 0]  # closer, farther, right, left
    #         first_filtered_object = objects[filtered_object_index_i]
    #         second_filtered_object = objects[filtered_object_index_j]
    #         first_filtered_object_position = [first_filtered_object[3][32][32], first_filtered_object[4][32][32]]
    #         second_filtered_object_position = [second_filtered_object[3][32][32], second_filtered_object[4][32][32]]
    #         move_class_factor = [second_filtered_object_position[0] - first_filtered_object_position[0],
    #                              second_filtered_object_position[1] - first_filtered_object_position[1]]
    #         if move_class_factor[0] > (100/480):
    #             move_class_instance[0] = 1
    #         elif move_class_factor[0] < -(100/480):
    #             move_class_instance[1] = 1
    #         if move_class_factor[1] > (100/640):
    #             move_class_instance[2] = 1
    #         elif move_class_factor[1] < -(100/640):
    #             move_class_instance[3] = 1
    #         encoded_object1 = torch.reshape(vision.encoder(torch.tensor(first_filtered_object,
    #                                                                     dtype=torch.float32).unsqueeze(dim=0))
    #                                         .squeeze(dim=0), [-1, 1, 1]).round().detach().numpy()
    #         encoded_object2 = torch.reshape(vision.encoder(torch.tensor(second_filtered_object,
    #                                                                     dtype=torch.float32).unsqueeze(dim=0))
    #                                         .squeeze(dim=0), [-1, 1, 1]).round().detach().numpy()
    #         pair_instance_1 = np.concatenate((encoded_object1, encoded_object2), axis=0)
    #         pair_instance_2 = np.concatenate((encoded_object2, encoded_object1), axis=0)
    #         pairs.append(pair_instance_1)
    #         move_class.append(move_class_instance)
    #         pairs.append(pair_instance_2)
    #         move_class.append([1 - move_class_instance_ele for move_class_instance_ele in move_class_instance])
    #         if len(pairs) >= 20000 or (filtered_object_index_i == len(objects)-1
    #                                    and filtered_object_index_j == len(objects)-1):
    #             print(len(pairs))
    #             pairs_to_store = np.reshape(pairs, [len(pairs), -1])
    #             np.save('rep_move_groups/pairs' + str(portion_num), pairs_to_store)
    #             np.save('rep_move_groups/move_class' + str(portion_num), move_class)
    #             portion_num += 1
    #             pairs = []
    #             move_class = []
    # pairs_to_store = np.reshape(pairs, [len(pairs), -1])
    # np.save('rep_move/pairs.txt', pairs_to_store)
    # np.save('rep_move/move_class.txt', move_class)
    # <<< prepare the data for training move analyser <<<
    # train move_analyser   in: concatenated_2_objects, out: move_description
    sig = 0
    train_data = list()
    test_data = list()
    train_sample_num = 0
    for i in range(15):
        print(sig)
        sig += 1
        pairs = np.load('rep_move/pairs' + str(i) + '.npy')
        move_class = np.load('rep_move/move_class' + str(i) + '.npy')
        pairs, move_class = shuffle_simul(pairs, move_class)
        if i < 14:
            # No.14 is still samples
            pairs = pairs[:len(pairs)//3]
            move_class = move_class[:len(move_class)//3]
        data_ele = np.concatenate((pairs, move_class), axis=1)
        np.random.shuffle(data_ele)
        train_data.append(data_ele[:-len(data_ele)//10])
        test_data.append(data_ele[-len(data_ele)//10:])
    train_data = np.concatenate(train_data, axis=0)
    np.random.shuffle(train_data)
    print('train data shuffle done!')
    test_data = np.concatenate(test_data, axis=0)
    print(len(train_data) + len(test_data))
    train_in = train_data[:, :16384]
    train_out = train_data[:, 16384:]
    test_in = test_data[:, :16384]
    test_out = test_data[:, 16384:]
    print('split done')
    train_in = torch.tensor(train_in, dtype=torch.float32).unsqueeze(dim=-1).unsqueeze(dim=-1).to(device1)
    train_out = torch.tensor(train_out, dtype=torch.float32).to(device1)
    test_in = torch.tensor(test_in, dtype=torch.float32).unsqueeze(dim=-1).unsqueeze(dim=-1).to(device1)
    test_out = torch.tensor(test_out, dtype=torch.float32).to(device1)
    print('unsqueeze done!')

    # ============================== train the models

    # train language.material    in: speech_rep, out: concept_rep
    # (this is not required now as the default source is memory)
    # train language.cue         in: speech_rep, out: concept_rep
    # speech_audio_files = os.listdir('rep_speech/')
    # speech_sentences = list()
    # objects = ['mouse', 'phone', 'apple', 'lemon', 'banana', 'sofa', 'book', 'clock', 'car', 'toy']
    # for i in range(len(objects)):
    #     speech_sentences.append('whatisthecolorof' + objects[i])
    #     speech_sentences.append('whatelsearearound' + objects[i])
    #     speech_sentences.append('howdoes' + objects[i] + 'move')
    # in_speeches = list()
    # test_speeches = list()
    # for i in range(len(speech_sentences)):
    #     in_speeches.append(np.loadtxt('rep_speech/' + speech_sentences[i] + '.txt'))
    #     test_speeches.append(np.loadtxt('rep_speech/' + speech_sentences[i] + '_test.txt'))
    # out_concepts = list()
    # for i in range(len(objects)):
    #     out_concepts.append(np.loadtxt('rep_speech/' + objects[i] + '.txt'))
    #
    # out_control_pattern1 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]  # color_inf 3, memory 1, non_temporal 2, single inf 5 for speech 1 - 10
    # out_control_pattern2 = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1]  # object_inf, memory, non_temporal, multiple inf for speech 11 - 20
    # out_control_pattern3 = [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1]  # move_model, memory, temporal, single inf for speech 21 - 30
    # out_control_patterns = [out_control_pattern1, out_control_pattern2, out_control_pattern3]
    #
    # train_data = list([[], [], [], [], [], [], [], [], [], [], [], [], [], [], []])
    # test_data = list()
    # sort_index = 0
    # while sort_index < len(speech_sentences):
    #     print(sort_index)
    #     in_data = in_speeches[sort_index]
    #     cue_cate = sort_index // 3
    #     question_cate = sort_index % 3
    #     target_data_cue = np.repeat(np.expand_dims(out_concepts[cue_cate], axis=0), len(in_data), axis=0)
    #     target_data_sig = np.repeat(np.expand_dims(out_control_patterns[question_cate], axis=0), len(in_data), axis=0)
    #     target_data = np.concatenate((in_data, target_data_cue, target_data_sig), axis=1)
    #     batch_size = len(target_data) // 15
    #     for train_data_batch_index in range(15):
    #         sub_batch = target_data[train_data_batch_index * batch_size: (train_data_batch_index + 1) * batch_size]
    #         np.random.shuffle(sub_batch)
    #         train_data[train_data_batch_index].append(sub_batch)
    #     test_in_data = test_speeches[sort_index]
    #     if len(np.shape(test_in_data)) == 1:
    #         test_in_data = np.expand_dims(test_in_data, axis=0)
    #     test_target_data_cue = np.repeat(np.expand_dims(out_concepts[cue_cate], axis=0), len(test_in_data), axis=0)
    #     test_target_data_sig = np.repeat(np.expand_dims(out_control_patterns[question_cate], axis=0), len(test_in_data), axis=0)
    #     test_target_data = np.concatenate((test_in_data, test_target_data_cue, test_target_data_sig), axis=1)
    #     test_data.append(test_target_data)
    #     sort_index += 1
    # print('load finished!')
    # test_data = np.concatenate(test_data, axis=0)
    # test_in = test_data[:, :8192]
    # test_out = test_data[:, 8192:]
    # print('data split completed')
    #
    # test_in = torch.tensor(test_in, dtype=torch.float32).to(device1).unsqueeze(dim=-1).unsqueeze(dim=-1)
    # test_out = torch.tensor(test_out, dtype=torch.float32).to(device1).unsqueeze(dim=-1).unsqueeze(dim=-1)
    # print(test_in.size(), test_out.size())
    # print('data unsqueeze completed')

    # train object inf           in: rep_object, out: rep_concept
    # objects = ['mouse', 'phone', 'apple', 'lemon', 'banana', 'sofa', 'book', 'clock', 'car', 'toy']
    # train_data = list()
    # test_data = list()
    # for i in range(len(objects)):
    #     in_data = np.loadtxt('rep_object/' + objects[i] + 's.txt')
    #     target_data = np.repeat(np.expand_dims(np.loadtxt('rep_speech/' + objects[i] + '.txt'),
    #                                            axis=0), len(in_data), axis=0)
    #     tmp_data = np.concatenate((in_data, target_data), axis=1)
    #     np.random.shuffle(tmp_data)
    #     train_data.append(tmp_data[:-len(tmp_data)//10])
    #     test_data.append(tmp_data[-len(tmp_data)//10:])
    # train_data = np.concatenate(train_data, axis=0)
    # test_data = np.concatenate(test_data, axis=0)
    # print(len(train_data) + len(test_data))
    # np.random.shuffle(train_data)
    # np.random.shuffle(test_data)
    # train_in = train_data[:, :8192]
    # train_out = train_data[:, 8192:]
    # test_in = test_data[:, :8192]
    # test_out = test_data[:, 8192:]
    #
    # train_in = torch.tensor(train_in, dtype=torch.float32).unsqueeze(dim=-1).unsqueeze(dim=-1).to(device1)
    # train_out = torch.tensor(train_out, dtype=torch.float32).unsqueeze(dim=-1).unsqueeze(dim=-1).to(device1)
    # test_in = torch.tensor(test_in, dtype=torch.float32).unsqueeze(dim=-1).unsqueeze(dim=-1).to(device1)
    # test_out = torch.tensor(test_out, dtype=torch.float32).unsqueeze(dim=-1).unsqueeze(dim=-1).to(device1)

    # train color inf            in: rep_color, out: rep_concept
    # data = []
    # colors = ['red', 'yellow', 'black', 'silver']
    # for i in range(len(colors)):
    #     in_data = np.loadtxt('rep_color/' + colors[i] + 's.txt')
    #     target_data = np.repeat(np.expand_dims(np.loadtxt('rep_speech/' + colors[i] + '.txt'),
    #                                            axis=0), len(in_data), axis=0)
    #     data.append(np.concatenate((in_data, target_data), axis=1))
    #
    # data = np.concatenate(data, axis=0)
    # np.random.shuffle(data)
    # train_data = data[:-len(data)//10]
    # test_data = data[-len(data)//10:]
    #
    # train_in = train_data[:, :8192]
    # train_out = train_data[:, 8192:]
    # test_in = test_data[:, :8192]
    # test_out = test_data[:, 8192:]
    #
    # train_in = torch.tensor(train_in, dtype=torch.float32).unsqueeze(dim=-1).unsqueeze(dim=-1).to(device1)
    # train_out = torch.tensor(train_out, dtype=torch.float32).unsqueeze(dim=-1).unsqueeze(dim=-1).to(device1)
    # test_in = torch.tensor(test_in, dtype=torch.float32).unsqueeze(dim=-1).unsqueeze(dim=-1).to(device1)
    # test_out = torch.tensor(test_out, dtype=torch.float32).unsqueeze(dim=-1).unsqueeze(dim=-1).to(device1)

    for i in range(1):
        # three trials for each training
        # configure wandb for the visualisation of results
        wandb.init(project="Move_move_analyser")
        wandb.config = {
            "learning_rate": 0.01,
            "epochs": 500
        }

        # material_an = factory.Analyser().to(device)
        # cue_an = factory.Analyser().to(device)
        # control_an = factory.ControlAnalyser().to(device)
        # language_an = factory.Analyser().to(device1)
        # language_an = torch.nn.DataParallel(factory.Analyser(), device_ids=[1, 0])
        # language_an = language_an.to(device1).train()
        # color_inf = factory.ColorClassifier('color').to(device1)
        # object_inf = factory.ObjectClassifier('object').to(device1)
        move_an = factory.MoveAnalyser().to(device1)

        # material_an = factory.Analyser().to(device)
        # cue_an = torch.load('trained_models/language/cue_analyser').to(device).train()
        # control_an = torch.load('trained_models/language/control_analyser').to(device).train()
        # color_inf = torch.load('trained_models/knowledge/color_inf').to(device).train()
        # object_inf = torch.load('trained_models/knowledge/object_inf').to(device).train()
        # move_an = torch.load('trained_models/move/move_an').to(device).train()

        # optimizer = torch.optim.SGD(params=cue_an.parameters(), lr=0.00001, momentum=0.9)
        # optimizer = torch.optim.SGD(params=control_an.parameters(), lr=0.01, momentum=0.9)
        # optimizer = torch.optim.SGD(params=object_inf.parameters(), lr=0.001, momentum=0.9)
        # optimizer = torch.optim.Adam(params=object_inf.parameters(), lr=0.001, weight_decay=0.000001)
        # optimizer = torch.optim.SGD(params=color_inf.parameters(), lr=0.0001, momentum=0.9)
        # optimizer = torch.optim.SGD(params=move_an.parameters(), lr=0.0001, momentum=0.9)
        optimizer = torch.optim.Adam(params=move_an.parameters(), lr=0.001, weight_decay=0.000001)
        # optimizer = torch.optim.Adam(params=language_an.parameters(), lr=0.001, weight_decay=0.000001)
        # optimizer = torch.optim.SGD(params=language_an.parameters(), lr=0.001, momentum=0.9)
        # print(train_in.size(), train_out.size(), test_in.size(), test_out.size())

        batch_size = 2048
        batch_num = int(np.ceil(len(train_in)/2048))
        acc_his = list()
        for epoch in range(500):
            epoch_loss = 0.
            for batch_index in range(batch_num):
                optimizer.zero_grad()
                end_index = batch_index*batch_size+batch_size
                if end_index > len(train_in):
                    end_index = len(train_in)
                input_data = train_in[batch_index*batch_size: end_index]
                output_data = train_out[batch_index*batch_size: end_index]
                out = move_an(input_data)
                loss = torch.sum((out - output_data) ** 2)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().cpu().numpy()
            print('epoch %d, loss: %f' % (epoch, epoch_loss))
            wandb.log({'loss': epoch_loss})
            # Optional
            # wandb.watch(cue_an)

            if (epoch+1) % 50 == 0:
                move_an.eval()
                # training acc
                out = move_an(train_in).round()
                acc_train = 0
                acc_train_fuzzy = 0
                for i in range(len(out)):
                    dif = torch.sum((out[i] - train_out[i]) ** 2)
                    if dif == 0:
                        acc_train += 1
                    if dif < 50:
                        acc_train_fuzzy += 1

                print('train acc: ', acc_train, len(train_in), acc_train/len(train_in), acc_train_fuzzy/len(train_in))

                # testing acc
                out = move_an(test_in).round()
                acc_test = 0
                acc_test_fuzzy = 0
                for i in range(len(out)):
                    dif = torch.sum((out[i] - test_out[i]) ** 2)
                    if dif == 0:
                        acc_test += 1
                    if dif < 50:
                        acc_test_fuzzy += 1
                print('test acc: ', acc_test, len(test_in), acc_test/len(test_in), acc_test_fuzzy/len(test_in))

                # validate acc
                # if len(validate_data) > 0:
                #     out = cue_an(validate_data_in.to(device)).round()
                #     print(out.size())
                #     print(validate_data_out.size())
                #     acc_val = 0
                #     acc_val_fuzzy = 0
                #     for i in range(len(out)):
                #         dif = torch.sum((out[i] - validate_data_out[i].to(device)) ** 2)
                #         if dif == 0:
                #             acc_val += 1
                #         if dif < 50:
                #             acc_val_fuzzy += 1
                #     print('validation acc: ', acc_val, len(validate_data_in),
                #           acc_val / len(validate_data_in), acc_val_fuzzy/len(validate_data_in))
                acc_his.append([acc_train / len(train_in), acc_test / len(test_in),
                                acc_train_fuzzy / len(train_in), acc_test_fuzzy / len(test_in)])
                wandb.log({'acc_train': acc_train / len(train_in), 'acc_test': acc_test / len(test_in),
                           'acc_train_fuzzy': acc_train_fuzzy / len(train_in),
                           'acc_test_fuzzy': acc_test_fuzzy / len(test_in)})
                # acc_his.append([acc_test / len(test_in), acc_test_fuzzy / len(test_in)])
                # wandb.log({'acc_test': acc_test / len(test_in), 'acc_test_fuzzy': acc_test_fuzzy / len(test_in)})
                move_an.train()

        # >>>> for model with big datasets >>>>
        # acc_his = list()
        # for epoch in range(500):
        #     acc_train = 0
        #     acc_train_fuzzy = 0
        #     train_sample_num = 0
        #     if (epoch + 1) % 50 == 0:
        #         test_flg = 1
        #     else:
        #         test_flg = 0
        #     epoch_loss = 0.
        #     for batch_index in range(len(train_data)):
        #         optimizer.zero_grad()
        #         # for language_an
        #         # train_data_batch = np.concatenate(train_data[batch_index], axis=0)
        #         # input_data = torch.tensor(train_data_batch[:, :8192], dtype=torch.float32).unsqueeze(
        #         #     dim=-1).unsqueeze(dim=-1).to(device1)
        #         # output_data = torch.tensor(train_data_batch[:, 8192:], dtype=torch.float32).unsqueeze(
        #         #     dim=-1).unsqueeze(dim=-1).to(device1)
        #         # for move_an
        #         input_data = torch.tensor(train_data[batch_index][:, :8192*2], dtype=torch.float32).unsqueeze(dim=-1).unsqueeze(dim=-1).to(device1)
        #         output_data = torch.tensor(train_data[batch_index][:, 8192*2:]).to(device1)
        #         out = move_an(input_data)
        #         loss = torch.sum((out - output_data) ** 2)
        #         if test_flg == 1:
        #             # out_copy = out.copy().round().detach()
        #             train_sample_num += len(input_data)
        #             for out_index in range(len(out)):
        #                 # dif = torch.sum((out_copy[i] - output_data[i]) ** 2)
        #                 dif = torch.sum((out[i].round() - output_data[i]) ** 2)
        #                 if dif == 0:
        #                     acc_train += 1
        #                 if dif < 50:
        #                     acc_train_fuzzy += 1
        #         loss.backward()
        #         optimizer.step()
        #         epoch_loss += loss.detach().cpu().numpy()
        #     print('epoch %d, loss: %f' % (epoch, epoch_loss))
        #     wandb.log({'loss': epoch_loss})
        #     # Optional
        #     # wandb.watch(cue_an)
        #
        #     if test_flg == 1:
        #         move_an.eval()
        #         # testing acc
        #         out = move_an(test_in).round()
        #         acc_test = 0
        #         acc_test_fuzzy = 0
        #         for out_index in range(len(out)):
        #             dif = torch.sum((out[out_index] - test_out[out_index]) ** 2)
        #             if dif == 0:
        #                 acc_test += 1
        #             if dif < 50:
        #                 acc_test_fuzzy += 1
        #         print('train acc: ', acc_train, train_sample_num,
        #               acc_train / train_sample_num, acc_train_fuzzy / train_sample_num)
        #         print('test acc: ', acc_test, len(test_in), acc_test/len(test_in), acc_test_fuzzy/len(test_in))
        #         acc_his.append([acc_train / train_sample_num, acc_test / len(test_in)])
        #         wandb.log({'acc_train': acc_train / train_sample_num,
        #                    'acc_train_fuzzy': acc_train_fuzzy / train_sample_num,
        #                    'acc_test': acc_test / len(test_in),
        #                    'acc_test_fuzzy': acc_test_fuzzy / len(test_in)})
        #         move_an.train()

        for ele in acc_his:
            print(ele)

        if acc_his[-1][1] < 0.9:
            continue
        move_an.eval()
        # language_an = language_an.module.to(torch.device('cpu'))
        move_an = move_an.to(torch.device('cpu'))
        torch.save(move_an, 'trained_models/move/move_an_bk')
        print('model saved successfully')
    exit(0)