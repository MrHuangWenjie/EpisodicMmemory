import torchvision
import utils.general
from python_speech_features import mfcc, logfbank
import scipy.io.wavfile as wav
import audio_monitor
import factory
from audio_monitor import load_audio, play_audio
from multiprocessing import Manager, Process
import time
import numpy as np
import torch
import cv2
import os
import struct
import librosa
import pyaudio
from ctypes import c_char_p
import zmq
import ctypes
import wandb
from matplotlib import cm
import matplotlib.pyplot as plt


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


def sort_keys_files(name):
    return int(name[5:-4])


def sort_keys_nets(name):
    return int(name[3:])


def status(x):
    for i in range(len(x)):
        if len(x[i]) != 0:
            return True
    return False


def save_configurations(path, frame):
    for i in range(len(frame)):
        try:
            s = len(frame[i])
            for j in range(s):
                torch.save(frame[i][j], path + 'net' + str(i * s + j))
        except TypeError:
            # if the element of frame is not in pair, the try lines would incur TypeError
            torch.save(frame[i], path + 'net' + str(i))


# >>>> Visual >>>>
# the image is firstly processed by an object filtering network (e.g, You-Only-Look-Once)
# then the visual system work on these proposals
# e.g., the proposed objects would go through a salience filter to
# identify objects, generate Semantic representations of them and extract corresponding attributes/relations
class Visual:

    external_perception = Manager().list([])
    candidate = Manager().list()
    # store the output of local processors
    salience = Manager().Value(typecode=float, value=0.0)
    cache_change_flg = Manager().Value(typecode=int, value=0)
    episodic_recall = Manager().list([[], []])
    source_gate = Manager().Value(typecode=int, value=0)    # 0 indicates vision, 1 indicates memory
    encoder = torch.load('./trained_models/vision/encoder', map_location=torch.device('cpu'))
    # if want to load this model, user has to import the class from another file explicitly here
    encoder.eval()

    def __init__(self):
        self.device = torch.device('cpu')
        # self.detector = torch.hub.load('/home/wenjie/PycharmProjects/yolov5', 'yolov5s',
        #                                pretrained=True, source='local').to(self.device)

        # self.detector = torch.load('./trained_models/vision/new_detector', map_location=self.device)
        # self.detector.eval()
        # self.detector = torch.load('./trained_models/vision/detector', map_location=self.device)
        # self.detector = torch.hub.load('/home/wenjie/PycharmProjects/yolov5',
        # 'yolov5s', pretrained=True, source='local').to(self.device)
        # self.detector.eval()
        self.detector = torch.load('./trained_models/vision/detector_detectron2', map_location=self.device)
        self.space = np.zeros([480, 640, 2])  # size of the image
        mid_x = 640 / 2.
        mid_y = 480 / 2.
        for i in range(480):
            for j in range(640):
                # self.space[i][j] = np.log10((j - mid_x) ** 2 + (i - mid_y) ** 2 + 1) / 5.3
                self.space[i][j][0] = (i + 1)/480.
                self.space[i][j][1] = (j + 1)/640.

        self.salience_threshold = 1.
        self.cache = list()

    def _encoder(self, x):
        # return the encoded result
        if len(x) == 0:
            return []
        # return torch.reshape(self.encoder(x).round(), [-1])
        return self.encoder(x).round()

    @staticmethod
    def square_padding(top_left, bottom_right):
        x = bottom_right[0] - top_left[0]
        y = bottom_right[1] - top_left[1]
        if x < y:
            new_x = top_left[0] - (y-x)//2
            if new_x >= 0:
                top_left[0] = new_x
            else:
                top_left[0] = 0
            new_x = bottom_right[0] + (y-x)//2
            if new_x <= 480:
                bottom_right[0] = new_x
            else:
                bottom_right[0] = 480
        if x > y:
            new_y = top_left[1] - (x-y)//2
            if new_y >= 0:
                top_left[1] = new_y
            else:
                top_left[1] = 0
            new_y = bottom_right[1] + (x-y)//2
            if new_y <= 640:
                bottom_right[1] = new_y
            else:
                bottom_right[1] = 640
        return top_left, bottom_right

    def object_filter(self, image):
        # return a list of coordinates of sub-areas containing objects
        # x = torch.tensor(x, dtype=torch.float32)
        # image = torch.tensor(image, dtype=torch.float32).permute([2, 0, 1]).unsqueeze(dim=0)
        # prediction = self.detector(image)[0]
        # results = utils.general.non_max_suppression(prediction)[0]
        results = self.detector(image)
        # model has two outputs, one of (1, 18900, 85), another is of (3, 3, 60, 80, 85)
        # prediction is the first one for detection
        # pass prediction to the non_max_suppression function to get the detection result;
        # result is of size [num_pic, num_box, 6], each row is [class, conf, *xyxy]
        # boxes = 0
        # for *xyxy, conf, cls in reversed(results):
        #     if conf > 0.75:
        #         boxes += 1
        # results.show()
        # return results.xyxy[0].data[:, 0:4]
        if len(results['instances'].pred_boxes) != 0:
            boxes = torch.cat([x.unsqueeze(dim=0) for x in results['instances'].pred_boxes], dim=0)
        else:
            boxes = torch.tensor([])
        return boxes

    def salience_filter(self, coordinates, external_p):
        # take perception as input which is numpy, return numpy
        img_size = np.shape(external_p)
        coordinates = np.int_(coordinates.cpu().detach().numpy())
        proposal_n = np.shape(coordinates)[0]
        salience_list = list()
        external_p = np.array(external_p)
        episodic_elements = list()
        sub_img = None
        top_left = None
        bottom_right = None
        salience_max = None
        # make it a numpy array, otherwise cannot dice the image for sub-img
        for i in range(proposal_n):
            # calculate the salience of each sub-area
            coordinate = coordinates[i]
            coordinate = [coordinate[1], coordinate[0], coordinate[3], coordinate[2]]
            # the coordinates calculated by YOLO is reversed
            top_left = coordinate[:2]
            bottom_right = coordinate[2:]
            top_left, bottom_right = self.square_padding(top_left, bottom_right)
            height = bottom_right[0] - top_left[0]
            width = bottom_right[1] - top_left[1]
            top_left_local = [max(top_left[0] - height, 0), max(top_left[1] - width, 0)]
            bottom_right_local = [min(bottom_right[0] + height, img_size[0]), min(bottom_right[1] + width, img_size[1])]
            # sum_sub = y, sum_expand = x, x-y = sum_back, then x-2y = salience relative to background
            sub = external_p[top_left[0]:bottom_right[0] + 1, top_left[1]: bottom_right[1] + 1]

            # cv2.namedWindow('sub')
            # cv2.imshow('sub', sub)
            # cv2.waitKey()
            # cv2.destroyWindow('sub')

            location = self.space[top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1]
            # sub_img = np.append(sub / 255., np.expand_dims(location, axis=2), axis=2)
            # print(np.shape(sub), np.shape(location))
            sub_img = np.append(sub / 255., location, axis=2)
            sub_img = cv2.resize(src=sub_img, dsize=(64, 64))
            # 64 64 5

            # annotation = input('Please input the annotation(num_object_color, 0 to abandon): ')
            # if annotation != '0':
            #     num, object_name, color_name = annotation.strip().split(' ')
            #     num = num.zfill(5)
            #     np.save('rep_dete/' + num + '_' + object_name + '_' + color_name, np.reshape(sub_img, [-1]))

            episodic_elements.append(sub_img)
            sum_sub = np.sum(sub)
            local = external_p[top_left_local[0]:bottom_right_local[0] + 1, top_left_local[1]: bottom_right_local[1] + 1]
            sum_local = np.sum(local)
            salience = abs(sum_local - 2 * sum_sub)
            salience_list.append(salience)
        if len(episodic_elements) > 0:
            episodic_elements = np.transpose(episodic_elements, (0, 3, 1, 2))
        return episodic_elements, salience_list

    def forward(self):
        try:
            while True:
                if self.source_gate.value == 0:
                    # take external perception
                    external_p = self.external_perception[:]
                    # print('externa_p: ', external_p)
                    if len(external_p) != 0:
                        # print('visual stimulus from external world')
                        # self.external_perception[:] = []
                        coordinates = self.object_filter(np.array(external_p.copy()))
                        # make the input into yoloV5 numpy array
                        # otherwise the detector duplicates twice the final dimension of the image
                        # the sub-image has been resized to [128, 128]
                        # sub_img, vertex1, vertex2, max_salience = self.salience_filter(coordinates, external_p)
                        episodic_elements, salience_list = self.salience_filter(coordinates, external_p)
                        # print('visual episodic_eles: ', np.shape(episodic_elements))
                    else:
                        [episodic_elements, salience_list] = [[], []]
                    if len(episodic_elements) > 0:
                        GW.source_flg[0] = 0
                else:
                    # take episodic_recall
                    episodic_elements, salience_list = self.episodic_recall[:]
                    # ===========================================
                    # episodic_elements_copy = torch.tensor(episodic_elements, dtype=torch.float32).to(self.device)
                    # encoded_elements = torch.reshape(self._encoder(episodic_elements_copy),
                    #                                  [len(episodic_elements_copy), -1, 1, 1]).detach()
                    # labels = Knowledge.label_infer(encoded_elements).round().detach().numpy().tolist()
                    # targets = [speak.out_concept_phone, speak.out_concept_mouse, speak.out_concept]
                    # ===========================================
                    external_p = episodic_elements
                    if len(episodic_elements) > 0:
                        GW.source_flg[0] = 1
                    # time.sleep(2)
                # if external_p is different from cache, store wm and cache to external_p
                if isinstance(self.cache, list):
                    cache_copy = np.array(self.cache)
                else:
                    cache_copy = self.cache
                if isinstance(external_p, list):
                    perception_copy = np.array(external_p)
                else:
                    perception_copy = external_p
                if np.shape(cache_copy) == np.shape(perception_copy):
                    change_value = np.sum((cache_copy / 255. - perception_copy / 255.) ** 2)
                else:
                    change_value = 999999
                if change_value > 10000:
                    print('change value:', change_value)
                    self.cache = perception_copy
                    self.cache_change_flg.value = 1

                if len(episodic_elements) == 0:
                    # print('Oh, no object attended')
                    continue
                # attentional coordination from working memory (unconsciously)
                wm_copy, wm_flag_copy = np.array(GW.wm[:]).copy(), np.array(GW.wm_flags[:]).copy()
                # print('wm.flags: ', GW.wm_flags[:])
                for i in range(len(episodic_elements)):
                    print(i * '===')
                    flag_var = 0
                    for j in range(len(wm_copy)):
                        tmp_epi = episodic_elements[i]
                        tmp_wm_epi = wm_copy[j]
                        if np.shape(tmp_epi) == np.shape(tmp_wm_epi):
                            dif = np.sum(np.abs(tmp_epi - tmp_wm_epi))
                            # print(dif)
                            if dif < 1000.:
                                if wm_flag_copy[j] >= 0.5:
                                    salience_list[i] = min(salience_list) / 2
                                    flag_var = 1
                                    break
                        if flag_var == 0:
                            salience_list[i] *= 2
                # print('saliences:', salience_list)
                max_salience = np.max(salience_list)
                episodic_elements = torch.tensor(episodic_elements, dtype=torch.float32).to(self.device)
                # GW.working_memory_v[:] = episodic_elements.detach().cpu().numpy()

                salient_index = int(np.argmax(salience_list))
                sub_img = episodic_elements[salient_index]
                # img_show = np.transpose(sub_img[:3].detach().numpy(), [1, 2, 0])
                # cv2.namedWindow('attended')
                # cv2.imshow('attended', img_show)
                # cv2.waitKey()
                handle = torch.reshape(self._encoder(sub_img.unsqueeze(dim=0)).squeeze(dim=0), [-1, 1, 1]).detach().numpy()
                # to transfer these signals, they need to be converted into numpy data first
                # as tensor data is not allowed to be transferred between processes

                # lag-1 sparing check & salience processing
                sub_img = sub_img.detach().cpu().numpy()

                if len(self.candidate[:]) == 0:
                    self.candidate.append([sub_img, handle, []])
                    self.salience.value = (max_salience/self.salience_threshold)
                else:
                    # print('lag-1 sparing activated')
                    s = max_salience/self.salience_threshold
                    if s > self.salience.value:
                        self.candidate[:] = []
                        self.candidate.append([sub_img, handle, []])
                        self.salience.value = s
                # update the salience threshold
                if self.salience.value >= 1.:
                    self.salience_threshold = 1.5 * max_salience
        except KeyboardInterrupt:
            print('Vision is exiting!')
            exit(0)
# <<<< Visual <<<<


# >>>> Auditory >>>>
class Auditory:
    # detector would automatically split the input sentences into voices with random length
    # uniformer will squeeze or un-squeeze the single word into a uniform size
    # each word will be encoded as an internal representation along with the encoder_, there would be a decoder for each
    # encoded sentence then would be used for semantic analysis to determine the event {sub, relation, obj}
    # the event would attempt to enter the global workspace
    # auditory module only for simplest language understanding here, further would be processed by language module
    # all input consist of 3 voices {sub, relation, obj}
    external_perception = Manager().list([])
    candidate = Manager().list()
    # store the outputs of local processors
    salience = Manager().Value(typecode=float, value=0.0)
    cache_change_flg = Manager().Value(typecode=int, value=0)
    source_gate = Manager().Value(typecode=int, value=0)

    def __init__(self):
        self.device = torch.device('cpu')
        self.encoder = torch.load('./trained_models/hearing/encoder', map_location=self.device)
        self.encoder.eval()
        self.salience_threshold = 1
        # self.task_analyser = factory.Analyser().eval()
        # self.cache = list()

    def _encoder(self, x):
        # input should be size of (chunk_num, chunk_size)
        # return the encoded result
        if x is None:
            return None
        else:
            return self.encoder(self.get_mfcc(x)).round()

    @staticmethod
    def get_mfcc(x):
        # input is a tensor sound signal
        # output is a tensor of mfcc of size [1, 2, 64, 64], which is suitable for feeding into encoder directly
        x = np.reshape(x, -1)
        mfcc_feat = mfcc(x, Ears.rate)
        resized_mfcc_feat = cv2.resize(np.array(mfcc_feat), (64, 64))

        fig, ax = plt.subplots()
        mfcc_data = np.swapaxes(resized_mfcc_feat.copy(), 0, 1)
        cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
        ax.set_title('MFCC')
        plt.show()

        resized_mfcc_feat = torch.tensor(resized_mfcc_feat, dtype=torch.float32).unsqueeze(dim=0)
        voice_tensor = torch.cat((resized_mfcc_feat, resized_mfcc_feat), dim=0).unsqueeze(dim=0)

        return voice_tensor

    @staticmethod
    def signal2int(x):
        for i in range(len(x)):
            data_int = np.array(struct.unpack(str(len(x[i])) + 'B', x[i]), dtype='b')
            # convert strings of hexadecimals into integers
            x[i] = data_int
        x = np.reshape(x, [-1])
        data_amplitude = librosa.db_to_amplitude(x)
        max_amplitude = np.max(data_amplitude)

        return x, max_amplitude

    def forward(self):
        try:
            while True:
                external_p = self.external_perception[:]

                # last priority is the external information
                if len(external_p) != 0:
                    # if external_p is different from cache, store wm and cache to external_p
                    # if len(self.cache) != 0:
                    #     if not isinstance(self.cache, list):
                    #         cache_copy = self.cache.tolist()
                    #     else:
                    #         cache_copy = self.cache
                    #     if not isinstance(external_p, list):
                    #         perception_copy = external_p.tolist()
                    #     else:
                    #         perception_copy = external_p
                    #     if cache_copy != perception_copy:
                    # self.cache = external_p
                    # self.cache_change_flg.value = 1

                    # print('hearing from external')
                    self.external_perception[:] = []
                    x, max_amplitude = self.signal2int(external_p)
                    # x is the int signal list, concatenating all chunks together
                    # GW.working_memory_a = x

                    rep = torch.reshape(self._encoder(x), [1, -1, 1, 1])
                    task = [0, 1, 0]
                    # task = torch.reshape(self.task_analyser(rep).round(), [-1])[:3].detach().cpu().numpy()
                    # tell if this requires language process

                    rep = rep.squeeze(dim=0).detach().cpu().numpy()

                    attention_factor = 1.
                    wm_copy = GW.wm[:]
                    wm_flags_copy = GW.wm_flags[:]
                    for i in range(len(wm_copy)):
                        if np.shape(x) == np.shape(wm_copy[i]):
                            if np.sum(np.abs(x - wm_copy[i])) < 1000.:
                                if wm_flags_copy[i] == 1:
                                    attention_factor = 0.5
                                    break
                    # lag-1 sparing check
                    if len(self.candidate[:]) == 0:
                        # self.candidate.append([x, rep, task])
                        self.candidate.append([[], rep, task])
                        self.salience.value = (max_amplitude / self.salience_threshold) * attention_factor
                    else:
                        print('lag-1 sparing activated')
                        s = (max_amplitude / self.salience_threshold) * attention_factor
                        if s > self.salience.value:
                            self.candidate[:] = []
                            # self.candidate.append([x, rep, task])
                            self.candidate.append([[], rep, task])
                            self.salience.value = s

                    # update the salience threshold
                    if self.salience.value > 1:
                        self.salience_threshold = max_amplitude
        except KeyboardInterrupt:
            print('Hearing is exiting...')
            exit(0)
# <<<< Auditory <<<<


class Language:

    candidate = Manager().list()
    # store the outputs of local processors
    salience = Manager().Value(typecode=float, value=0.0)
    cue = Manager().list()
    source_gate = Manager().Value(typecode=int, value=0)
    clock_number = Manager().Value(typecode=int, value=0)
    recall_mode = Manager().list([0, 0])

    objects = ['mouse', 'phone', 'apple', 'lemon', 'banana', 'sofa', 'book', 'clock', 'car', 'toy']
    concepts = list()
    for i in range(len(objects)):
        # concepts.append(np.expand_dims(np.expand_dims(np.loadtxt('rep_speech/' + objects[i] + '.txt'),
        #                                               axis=-1), axis=-1).tolist())
        concepts.append(np.expand_dims(np.expand_dims(np.loadtxt('rep_speech/' + objects[i] + '.txt'),
                                                      axis=-1), axis=-1))

    def __init__(self):
        self.device = torch.device('cpu')
        # self.task_analyser = factory.Analyser().eval().to(self.device)
        # self.task_clock_analyser = factory.Analyser().eval().to(self.device)
        # self.source_analyser = factory.Analyser().eval().to(self.device)
        # self.control_signals_analyser = torch.load('trained_models/language/control_analyser',
        #                                            map_location=torch.device('cpu')).eval()

        # self.material_analyser = factory.Analyser().eval().to(self.device)
        # self.cue_analyser = torch.load('trained_models/language/cue_analyser', map_location=torch.device('cpu')).eval()
        self.analyser = torch.load('trained_models/language/analyser', map_location=torch.device('cpu')).eval()
        # print(self.analyser)

    def forward(self):
        try:
            while True:
                x = GW.broadcast[2]
                if len(x) != 0:
                    GW.broadcast[2] = []
                    GW.block.append(1)
                    if np.array(x[1]).tolist() == [0, 1, 0]:
                        # if the task requires language processing
                        x = torch.tensor(x[0], dtype=torch.float32).unsqueeze(dim=0)

                        # material = self.material_analyser(x).round().squeeze(dim=0).detach().cpu().numpy()
                        material = []
                        analyser_results = self.analyser(x).round().squeeze(dim=0).detach().cpu()
                        # self.cue[:] = analyser_results[:8192].numpy()
                        cue_tmp = analyser_results[:8192].numpy()
                        control_signals = analyser_results[8192:].squeeze(dim=-1).squeeze(dim=-1).numpy()
                        # self.cue[:] = self.cue_analyser(x).round().squeeze(dim=0).detach().cpu().numpy()

                        # if language module wins the competition, these would be done in GW
                        # Episodic.cue[:] = cue
                        # GW.cue[:] = Episodic.cue[:]
                        # QA.hold[:] = Episodic.cue[:]
                        # Visual.source_gate[:] = source_gate
                        # GW.clock_num.value = clock_number
                        # Episodic.recall_mode = recall_mode

                        # output the results of analyser
                        # cue_tmp = self.cue[:]
                        # cue_tmp = np.repeat(np.expand_dims(cue_tmp, axis=0), len(self.objects), axis=0)
                        diffs = [np.sum(np.abs(cue_tmp - concept_ele)) for concept_ele in self.concepts]
                        cue_index = np.argmin(diffs)
                        cue_name = self.objects[cue_index]
                        print('the language cue is ', cue_name)
                        self.cue[:] = self.concepts[cue_index]

                        # control_signals = self.control_signals_analyser(x).squeeze(dim=0).round().detach().cpu().numpy()
                        task = control_signals[:3]
                        self.source_gate.value = control_signals[3]
                        self.recall_mode[:] = control_signals[4:6]
                        clk_num = sum(control_signals[6:])
                        if clk_num == 1:
                            self.clock_number.value = 1
                        else:
                            self.clock_number.value = 10
                        # try:
                        #     print(np.shape(self.concepts))
                        #     print(np.shape(cue_tmp))
                        #     position = self.concepts.index(cue_tmp)
                        #     print(position)
                        #     print('the language cue is ', self.objects[position])
                        # except ValueError:
                        #     print('cue analyser failed, language processing is abandoned!')
                        #     self.cue[:] = list()
                        #     self.source_gate.value = 0
                        #     self.clock_number.value = 0
                        #     self.recall_mode[:] = [0, 0]
                        #     GW.block.pop()
                        #     continue
                        print('the task, source gate, recall mode and clock num are ', task, self.source_gate.value, self.recall_mode[:], self.clock_number.value)

                        self.candidate.append([[], material, task])
                        self.salience.value = 1
                        GW.clock_num.value -= 1
                        if GW.clock_num.value < 1:
                            print('task finished!')
                    GW.block.pop()

        except KeyboardInterrupt:
            print('Language is exiting...')
            exit(0)


# >>>> Knowledge >>>>
class Knowledge:

    candidate = Manager().list()
    # store the output of local processors
    salience = Manager().Value(typecode=float, value=0.0)
    examine_flag = Manager().Value(typecode=int, value=None)
    # this flag indicates if the received information is right or not
    # three statuses, None, 1, -1, 0 indicating no result, right, wrong and do-not-know respectively
    response_flag = Manager().Value(typecode=int, value=0)
    # this flag when -1 indicating no response/ result from inference, 0 indicates nothing, 1 indicates response
    color_infer = torch.load('trained_models/knowledge/color_inf', map_location=torch.device('cpu')).eval()
    label_infer = torch.load('trained_models/knowledge/object_inf', map_location=torch.device('cpu')).eval()

    def __init__(self):
        self.device = torch.device('cpu')

    def forward(self):
        try:
            while True:
                tmp = GW.broadcast[3]
                if len(tmp) != 0:
                    # print('knowledge time:', time.time())
                    GW.broadcast[3] = []
                    GW.block.append(1)
                    if len(tmp[1]) == 0 or len(tmp[0]) == 0:
                        # no task or no material
                        GW.block.pop()
                        continue

                    # if the task is for knowledge module, there are tasks of color_infer, label_infer
                    # maybe and color_learning and label learning
                    # for infer task, how to tell the result is valid or not
                    if np.array(tmp[1]).tolist() == [0, 0, 0]:
                        # task color_infer
                        re = self.color_infer(torch.tensor(tmp[0], dtype=torch.float32).unsqueeze(dim=0)
                                              ).round().squeeze(dim=0).detach().cpu().numpy()
                        # if the result is not valid, set re to None
                        self.candidate.append([[], re, []])
                        self.salience.value = 1.
                        gw.ef_copy_num.value += 1
                        GW.clock_num.value -= 1
                        if GW.clock_num.value < 1:
                            print('task finished!')
                    elif np.array(tmp[1]).tolist() == [0, 0, 1]:
                        # task label_infer
                        re = self.label_infer(torch.tensor(tmp[0], dtype=torch.float32).unsqueeze(dim=0)
                                              ).round().squeeze(dim=0).detach().cpu().numpy()
                        # if the result is not valid, set re to None
                        self.candidate.append([[], re, []])
                        self.salience.value = 1.
                        gw.ef_copy_num.value += 1
                        GW.clock_num.value -= 1
                        if GW.clock_num.value < 1:
                            print('task finished!')
                    GW.block.pop()
        except KeyboardInterrupt:
            print('Knowledge is exiting...')
            exit(0)
# <<<< Knowledge <<<<


# >>>> Episodic Module >>>>
class Episodic:

    cue = Manager().list()
    recall_mode = Manager().list([0, 0])
    candidate = Manager().list()
    # store the output of local processors
    salience = Manager().Value(typecode=float, value=0.0)
    time_clue = Manager().Value(typecode=int, value=-1)
    memory_strength = Manager().list()

    def __init__(self):
        self.storage = 'trained_models/episodic_memory/'
        self.memory_content = list()
        # self.memory_strength = list()
        self.memory_index = 0
        self.load_memory()
        self.memory_strength_tracker = []

    def load_memory(self):
        memory_tracks = os.listdir(self.storage)
        memory_tracks.sort()
        for track in memory_tracks:
            low_dim = np.loadtxt(self.storage + track)
            if len(np.shape(low_dim)) == 1:
                low_dim = np.expand_dims(low_dim, axis=0)
            track_strength = [low_dim_ele[-1] for low_dim_ele in low_dim]
            low_dim = [low_dim_ele[:-1] for low_dim_ele in low_dim]
            self.memory_content.append(np.reshape(low_dim, [len(low_dim), 5, 64, 64]).tolist())
            self.memory_strength.append(track_strength)
        self.memory_index = len(memory_tracks)

    def retrieve(self):
        for track in self.memory_content:
            if self.cue[:] in track:
                salience_list = np.ones(len(track))
                Visual.episodic_recall[:] = [track, salience_list]
        return None

    def retrieve_(self):
        try:
            while True:
                if len(GW.block[:]) == 0 and len(self.cue[:]) != 0:
                    cue_tmp = self.cue[:]
                    print('start memory retrieval...')
                    for track_index in range(len(self.memory_content)):
                        print('recalling according to the cue in ', track_index)
                        # filter the desired memory
                        track = self.memory_content[track_index]
                        alive_track_fraction = []
                        for track_fraction_index in range(len(track)):
                            # only alive memory would be recalled
                            if self.memory_strength[track_index][track_fraction_index] > 0:
                                alive_track_fraction.append(track[track_fraction_index])
                        if len(alive_track_fraction) != 0:
                            encoded_rep = torch.reshape(Visual.encoder(
                                torch.tensor(alive_track_fraction, dtype=torch.float32)).round(), [-1, 8192, 1, 1])
                            re = Knowledge.label_infer(encoded_rep).round().detach().cpu().numpy()

                            p = None
                            for i in range(len(re)):
                                if re[i].tolist() == cue_tmp:
                                    p = i
                                    img_show = np.transpose(alive_track_fraction[p][:3], [1, 2, 0])
                                    cv2.namedWindow('retrieved')
                                    cv2.imshow('retrieved', img_show)
                                    cv2.waitKey()
                            if p is not None:
                                print('experience recalled according to the cue in ', track_index)
                                self.time_clue.value = track_index
                                salience_list = np.ones(len(alive_track_fraction)) * 0.8
                                salience_list[p] = 1
                                Visual.episodic_recall[:] = [alive_track_fraction, salience_list]
                                if self.recall_mode[:] == [0, 0]:
                                    # non-temporal mode
                                    print('retrieved memory contains element: ', np.shape(alive_track_fraction))
                                    self.cue[:] = []
                                    break
                            else:
                                print('nothing recalled in ', track_index)
                        else:
                            print('nothing recalled in ', track_index)
                        time.sleep(3)
                        if track_index == len(self.memory_content) - 1:
                            print('retrieval completed')
                            self.cue[:] = []

        except KeyboardInterrupt:
            print('Episodic memory is exiting...')
            exit(0)

    def store(self, memory, store_mode):
        print('the memory to be stored: ', len(memory))
        if len(memory) != 0:
            memory_content_store = [memory_ele[0] for memory_ele in memory]
            memory_strength_store = [memory_ele[1] for memory_ele in memory]
            if store_mode == 1:
                print('strengthen recalled memory')
                # strengthen recalled memory
                for i in range(len(memory_content_store)):
                    tmp_a = memory_content_store[i]
                    for j in range(len(self.memory_content)):
                        for k in range(len(self.memory_content[j])):
                            tmp_b = self.memory_content[j][k]
                            if not isinstance(tmp_b, list):
                                tmp_b.tolist()
                            try:
                                if tmp_a == tmp_b:
                                    self.memory_strength[j][k] += memory_strength_store[i]
                            except ValueError:
                                print('tmp_a and b cannot be compared:', np.shape(tmp_a), np.shape(tmp_b))
                                print('content and strength:', np.shape(memory_content_store),
                                      np.shape(memory_strength_store))
            else:
                print('new memory created')
                for i in range(len(memory_content_store)):
                    tmp_a = memory_content_store[i]
                    for j in range(len(self.memory_content)):
                        for k in range(len(self.memory_content[j])):
                            tmp_b = self.memory_content[j][k]
                            if not isinstance(tmp_b, list):
                                tmp_b.tolist()
                            try:
                                if tmp_a == tmp_b:
                                    self.memory_strength[j][k] += memory_strength_store[i]
                                    memory_strength_store[i] = self.memory_strength[j][k]
                            except ValueError:
                                print('tmp_a and b cannot be compared:', np.shape(tmp_a), np.shape(tmp_b))
                                print('content and strength:', np.shape(memory_content_store),
                                      np.shape(memory_strength_store))
                # store new memory traces
                self.memory_content.append(memory_content_store)
                # memory is of [memory_num, ele_num]
                self.memory_strength.append(memory_strength_store)
                # memory_str is of [memory_num, ele_num]
                memory_content_store = np.reshape(memory_content_store, [len(memory), -1])
                # print('memory content:', np.shape(memory_content_store))
                store_memory = np.concatenate((memory_content_store,
                                               np.expand_dims(memory_strength_store, axis=1)), axis=1)
                # print('memory content appended with strength:', np.shape(store_memory))
                # for i in range(len(store_memory)):
                #     # append the strength of each memory ele at the end
                #     store_memory[i].append(self.memory_strength[i])
                np.savetxt('trained_models/episodic_memory/' + str(self.memory_index).zfill(6) + '.txt', store_memory)
                self.memory_index += 1
                print('memory to store: ', np.shape(memory), ' successfully stored')
        else:
            print('nothing to store!!!')

    def forgetting(self):
        try:
            while True:
                # tmp_memory_strength = np.reshape(self.memory_strength.copy(), -1)
                # for i in range(len(tmp_memory_strength)):
                #     if tmp_memory_strength[i] <= 2:
                #         tmp_memory_strength[i] -= 0.1
                # self.memory_strength = np.reshape(tmp_memory_strength, (2, 3))
                # self.memory_strength[:] -= 0.001
                if len(self.memory_content) == 0:
                    continue
                memory_strength_track = np.concatenate(self.memory_strength[:], axis=0)
                # print(memory_strength_track)
                self.memory_strength_tracker.append(memory_strength_track)
                memory_strength_copy = self.memory_strength[:]
                # self.memory_strength[:] -= 0.01
                for i in range(len(memory_strength_copy)):
                    for j in range(len(memory_strength_copy[i])):
                        memory_strength_copy[i][j] -= 0.01
                        if memory_strength_copy[i][j] < 0.:
                            memory_strength_copy[i][j] = 0.
                self.memory_strength[:] = memory_strength_copy
                time.sleep(3)
        except KeyboardInterrupt:
            print('Episodic forgetting is saving memory strength tracker...')

            np.save('memory_strength_tracker_interactive', self.memory_strength_tracker)
            print('Episodic forgetting exits')
            exit(0)

# <<<< Episodic Module <<<<


class Goal:
    candidate = Manager().list()
    # store the output of local processors
    salience = Manager().Value(typecode=float, value=0.0)

    def __init__(self):
        self.emotion_factor = np.expand_dims(np.expand_dims(np.loadtxt('rep_speech/red.txt'), axis=-1),
                                             axis=-1).tolist()
        self.goal_factor = np.expand_dims(np.expand_dims(np.loadtxt('rep_speech/apple.txt'), axis=-1),
                                          axis=-1).tolist()

    def wait_goal(self):
        try:
            while True:
                tmp = GW.broadcast[5]
                if len(tmp) != 0:
                    # print('knowledge time:', time.time())
                    GW.broadcast[5] = []
                    GW.block.append(1)
                    tmp = tmp[0]
                    if len(tmp) == 0:
                        GW.block.pop()
                        continue
                    # print('come in signal: ', np.shape(tmp))
                    if not isinstance(tmp, list):
                        tmp = tmp.tolist()
                    if tmp == self.goal_factor or tmp == self.emotion_factor:
                        # if processed here, feedback to GW the ef_copy
                        gw.ef_copy_num.value += 10
                    try:
                        re = Knowledge.label_infer(
                            torch.tensor(tmp, dtype=torch.float32
                                         ).unsqueeze(dim=0)).squeeze(dim=0).round().detach().cpu().numpy()
                        if re.tolist() == self.goal_factor or re.tolist() == self.emotion_factor:
                            # if processed here, feedback to GW the ef_copy
                            gw.ef_copy_num.value += 10

                        re = Knowledge.color_infer(
                            torch.tensor(tmp, dtype=torch.float32
                                         ).unsqueeze(dim=0)).squeeze(dim=0).round().detach().cpu().numpy()
                        if re.tolist() == self.goal_factor or re.tolist() == self.emotion_factor:
                            # if processed here, feedback to GW the ef_copy
                            gw.ef_copy_num.value += 10
                    except RuntimeError:
                        pass
                    GW.block.pop()
        except KeyboardInterrupt:
            print('Goal is exiting...')
            exit(0)


class MoveModel:
    candidate = Manager().list()
    salience = Manager().Value(typecode=float, value=0.)

    def __init__(self):
        self.move_analyser = torch.load('trained_models/move/move_an', map_location=torch.device('cpu')).eval()
        # print(self.move_analyser)
        self.temporal_cache = []
        self.temporal_tags = []

    def forward(self):
        try:
            while True:
                x = GW.broadcast[6]
                if len(x) != 0:
                    GW.broadcast[6] = []
                    GW.block.append(1)
                    revisit = 0
                    material = np.array(x[0])
                    if np.array(x[1]).tolist() == [0, 1, 1]:
                        if len(material) != 0:
                            time_clue = Episodic.time_clue.value
                            for temporal_step_index in range(len(self.temporal_cache)):
                                if self.temporal_cache[temporal_step_index] == material.tolist() or \
                                        self.temporal_tags[temporal_step_index] == time_clue:
                                    revisit = 1
                                    break
                            if revisit == 0:
                                # print('non-revisited')
                                cue = GW.cue[:]
                                # if the task requires MoveModel processing
                                x = torch.tensor(material, dtype=torch.float32).unsqueeze(dim=0)
                                re = Knowledge.label_infer(x).round().squeeze(dim=0).detach().cpu().numpy()
                                if cue == re.tolist():
                                    # print('this is the cued info')
                                    self.temporal_cache.append(x)
                                    self.temporal_tags.append(time_clue)
                                # else:
                                #     print(np.shape(cue))
                                #     print('this is not the cued info')

                            # if len(self.temporal_cache) >= 2:
                            epi_cue = Episodic.cue[:]
                            if len(epi_cue) == 0:
                                if len(self.temporal_tags) >= 2:
                                    # print('temporal clues:', self.temporal_tags)
                                    # ready to analyse the movement
                                    # first step is to sort the steps by time_clue
                                    move_steps = []
                                    # sort the steps by temporal sequence
                                    for sorted_num in range(len(self.temporal_cache)):
                                        oldest_index = np.argmin(self.temporal_tags)
                                        if oldest_index == 9999:
                                            break
                                        move_steps.append(self.temporal_cache[oldest_index])
                                        self.temporal_tags[oldest_index] = 9999

                                    # do the analysis on the first and last step
                                    movement = torch.tensor(np.concatenate((move_steps[0], move_steps[-1]), axis=1), dtype=torch.float32)
                                    # print(movement.size())
                                    move_desc = self.move_analyser(movement).squeeze(dim=0).round().detach().cpu().numpy()
                                    print(move_desc)
                                    self.candidate.append([[], move_desc, []])
                                    self.salience.value = 1
                                else:
                                    print(len(self.temporal_tags))
                                    print('it did not move')
                                self.temporal_cache = []
                                self.temporal_tags = []
                                GW.clock_num.value -= 1
                                if GW.clock_num.value < 1:
                                    print('task finished!')
                    GW.block.pop()

        except KeyboardInterrupt:
            print('MoveModel is exiting...')
            exit(0)

# >>>> Speaking >>>>


class QA:

    candidate = Manager().list()
    # store the output of local processors
    salience = Manager().Value(typecode=float, value=0)
    hold = Manager().list()
    sound = Manager().list()

    def __init__(self):
        # self.speaker = factory.Analyser()
        self.out_concept_mouse = np.expand_dims(np.expand_dims(np.loadtxt('rep_speech/mouse.txt'), axis=-1),
                                                axis=-1).tolist()
        self.out_concept_phone = np.expand_dims(np.expand_dims(np.loadtxt('rep_speech/phone.txt'), axis=-1),
                                                axis=-1).tolist()
        self.out_concept_car = np.expand_dims(np.expand_dims(np.loadtxt('rep_speech/car.txt'), axis=-1),
                                                axis=-1).tolist()
        self.out_concept_toy = np.expand_dims(np.expand_dims(np.loadtxt('rep_speech/toy.txt'), axis=-1),
                                                axis=-1).tolist()
        self.out_concept_red = np.expand_dims(np.expand_dims(np.loadtxt('rep_speech/red.txt'), axis=-1),
                                              axis=-1).tolist()
        self.out_concept_yellow = np.expand_dims(np.expand_dims(np.loadtxt('rep_speech/yellow.txt'), axis=-1),
                                                 axis=-1).tolist()
        self.out_concept_apple = np.expand_dims(np.expand_dims(np.loadtxt('rep_speech/apple.txt'), axis=-1),
                                                axis=-1).tolist()
        self.out_concept_lemon = np.expand_dims(np.expand_dims(np.loadtxt('rep_speech/lemon.txt'), axis=-1),
                                                axis=-1).tolist()
        self.out_concept_banana = np.expand_dims(np.expand_dims(np.loadtxt('rep_speech/banana.txt'), axis=-1),
                                                 axis=-1).tolist()
        self.out_concept_sofa = np.expand_dims(np.expand_dims(np.loadtxt('rep_speech/sofa.txt'), axis=-1),
                                               axis=-1).tolist()
        self.out_concept_book = np.expand_dims(np.expand_dims(np.loadtxt('rep_speech/book.txt'), axis=-1),
                                               axis=-1).tolist()
        self.out_concept_clock = np.expand_dims(np.expand_dims(np.loadtxt('rep_speech/clock.txt'), axis=-1),
                                                axis=-1).tolist()
        self.out_concept_silver = np.expand_dims(np.expand_dims(np.loadtxt('rep_speech/silver.txt'), axis=-1),
                                                 axis=-1).tolist()
        self.out_concept_black = np.expand_dims(np.expand_dims(np.loadtxt('rep_speech/black.txt'), axis=-1),
                                                axis=-1).tolist()
        self.description_closer = np.array([1, 0, 0, 0])
        self.description_farther = np.array([0, 1, 0, 0])
        self.description_right = np.array([0, 0, 1, 0])
        self.description_left = np.array([0, 0, 0, 1])
        self.description_still = np.array([0, 0, 0, 0])

        self.ans_cache = list()

    def forward(self):
        # >>> two modes of QA, with inference result or with check result <<<

        try:
            while True:
                x = GW.broadcast[7]
                if len(x) != 0:
                    GW.broadcast[7] = []
                    GW.block.append(1)
                    speech = ''
                    # ans = np.array(x[0])
                    ans = x[0].tolist()
                    try:
                        position = self.ans_cache.index(ans)
                        GW.block.pop()
                        if gw.clock_num.value == 0:
                            self.ans_cache = list()
                        continue
                    except ValueError:
                        pass
                    self.ans_cache.append(ans)
                    # ans_list = ans.tolist()
                    ans_list = ans
                    if ans_list != self.hold[:]:
                        # speech = self.speaker(x[0])
                        # audio_monitor.play_audio(speech)
                        if ans_list == self.out_concept_mouse:
                            speech = 'mouse'
                            print('mouse')
                        elif ans_list == self.out_concept_phone:
                            speech = 'phone'
                            print('phone')
                        elif ans_list == self.out_concept_apple:
                            speech = 'apple'
                            print('apple')
                        elif ans_list == self.out_concept_lemon:
                            speech = 'lemon'
                            print('lemon')
                        elif ans_list == self.out_concept_car:
                            speech = 'car'
                            print('car')
                        elif ans_list == self.out_concept_toy:
                            speech = 'toy'
                            print('toy')
                        elif ans_list == self.out_concept_banana:
                            speech = 'banana'
                            print('banana')
                        elif ans_list == self.out_concept_sofa:
                            speech = 'sofa'
                            print('sofa')
                        elif ans_list == self.out_concept_book:
                            speech = 'book'
                            print('book')
                        elif ans_list == self.out_concept_clock:
                            speech = 'clock'
                            print('clock')
                        elif ans_list == self.out_concept_red:
                            speech = 'red'
                            print('red')
                        elif ans_list == self.out_concept_yellow:
                            speech = 'yellow'
                            print('yellow')
                        elif ans_list == self.out_concept_black:
                            speech = 'black'
                            print('black')
                        elif ans_list == self.out_concept_silver:
                            speech = 'silver'
                            print('silver')
                        elif ans_list == self.description_farther.tolist():
                            speech = 'itmovesaway'
                            print('it moves away')
                        elif ans_list == self.description_closer.tolist():
                            speech = 'itmovescloser'
                            print('it moves closer')
                        elif ans_list == self.description_right.tolist():
                            speech = 'itmovestoright'
                            print('it moves to right')
                        elif ans_list == self.description_left.tolist():
                            print('it moves to left')
                        elif ans_list == self.description_still.tolist():
                            print('still')
                            speech = 'itisstill'
                        elif ans_list == (self.description_closer + self.description_right).tolist():
                            print('closer and right')
                            speech = 'itmovedclosertotheright'
                        elif ans_list == (self.description_closer + self.description_left).tolist():
                            print('closer and left')
                            speech = 'itmovedclosertotheleft'
                        elif ans_list == (self.description_farther + self.description_right).tolist():
                            print('it moved farther to the right')
                            speech = 'itmovedfarthertotheright'
                        elif ans_list == (self.description_farther + self.description_left).tolist():
                            print('it moved farther to the left')
                            speech = 'itmovedfarthertotheleft'
                        else:
                            print('wrong ans')
                        # if len(speech) > 0:
                        #     cmd = 'sshpass -p \'nao\' ssh nao@192.168.0.107 \'qicli call ALAudioPlayer.playFile
                        #     /home/nao/wenjie/audios/' + speech + '001.wav\''
                        #     os.system(cmd)
                    if gw.clock_num.value == 0:
                        self.ans_cache = list()
                    GW.block.pop()
        except KeyboardInterrupt:
            print('Speaking(QA) is exiting')
            exit(0)
# <<<< Speaking <<<<


# >>>> GW >>>>
class GW:

    broadcast = Manager().list([[], [], [], [], [], [], [], []])
    block = Manager().list()
    attention = Manager().list([1, 1, 1, 1, 1, 1, 1, 1])
    source_flg = Manager().list([0, 0, 0, 0, 0, 0, 0, 0])
    # the signal would be stored into memory needs to indicate this flag each time
    # working_memory_v = Manager().list()
    # working_memory_a = Manager().list()
    wm = Manager().list()
    wm_flags = Manager().list()
    wm_voluntary_attention = Manager().list()
    clock_num = Manager().Value(typecode=int, value=0)
    cue = Manager().list()
    ef_copy_num = Manager().Value(typecode=int, value=0)

    def __init__(self):
        # self.connected = Manager().list()
        # # gw is connected to all specialists
        self.candidates = [[], [], [], [], [], [], [], []]
        self.salience_list = np.zeros(8)
        self.specialists = [Visual, Auditory, Language, Knowledge, Episodic, Goal, MoveModel, QA]
        self.task = []
        self.transient_memory_index = -1
        self.perception_source = [0, 0]

    def read_signals(self):
        for i in range(len(self.specialists)):
            if len(self.specialists[i].candidate) == 0:
                self.candidates[i] = []
                self.salience_list[i] = -10
            else:
                self.candidates[i] = self.specialists[i].candidate.pop(0)
                self.salience_list[i] = self.specialists[i].salience.value

    # def refresh_wm(self):
    #     print('GW refreshes working memory...')
    #     # for episode in self.working_memory_a[:]:
    #     #     self.wm[:].append(episode)
    #     tmp = self.working_memory_v[:]
    #     tmp_wm = list()
    #     for episode in tmp:
    #         tmp_wm.append(episode)
    #     self.wm[:] = tmp_wm
    #     return

    def voluntary_attention(self):
        if len(self.cue[:]) != 0 and len(self.wm[:]) != 0:
            encoded_rep = torch.reshape(Visual.encoder(torch.tensor(self.wm[:], dtype=torch.float32)), [-1, 8192, 1, 1])
            re = Knowledge.label_infer(encoded_rep.round()).round().detach().cpu().numpy()

            p = None
            for i in range(len(re)):
                if re[i].tolist() == self.cue[:]:
                    p = i
            self.wm_voluntary_attention[:] = np.zeros(len(self.wm[:]))
            if p is not None:
                self.wm_voluntary_attention[p] = 1.

    def decaying(self):
        # the decaying rate in working memory is as double fast as that in episodic memory
        try:
            while True:
                # tmp_memory_strength = np.reshape(self.memory_strength.copy(), -1)
                # for i in range(len(tmp_memory_strength)):
                #     if tmp_memory_strength[i] <= 2:
                #         tmp_memory_strength[i] -= 0.1
                # self.memory_strength = np.reshape(tmp_memory_strength, (2, 3))
                # self.memory_strength[:] -= 0.001
                if len(self.wm) == 0:
                    continue
                # memory_strength_track = np.concatenate(self.wm_flags[:], axis=0)
                # self.memory_strength_tracker.append(memory_strength_track)
                memory_strength_copy = self.wm_flags[:]
                # self.memory_strength[:] -= 0.01
                for i in range(len(memory_strength_copy)):
                    memory_strength_copy[i] -= 0.02
                    if memory_strength_copy[i] < 0.:
                        memory_strength_copy[i] = 0.
                self.wm_flags[:] = memory_strength_copy
                time.sleep(0.1)
        except KeyboardInterrupt:
            print('GW decaying is exiting')
            exit(0)

    def compete(self):
        # read out the forwarded signals from all specialists, competing for access to GW
        # the winner is broadcast to all specialists except the origin of the certain signal
        try:
            while True:
                if len(self.block[:]) != 0:
                    continue
                else:
                    if self.transient_memory_index != -1:
                        if self.ef_copy_num.value != 0:
                            self.wm_flags[self.transient_memory_index] += self.ef_copy_num.value
                            if self.wm_flags[self.transient_memory_index] > 100.:
                                self.wm_flags[self.transient_memory_index] = 100.
                    self.transient_memory_index = -1    # reset the index to transient episodic memory
                    self.ef_copy_num.value = 0  # reset the num of ef_copy_feedback

                    # read signals from specialists
                    self.read_signals()
                    # >>>> competition >>>>
                    self.salience_list = self.salience_list + self.attention[:]
                    if np.max(self.salience_list) == -10:
                        continue
                    winner_index = np.argmax(self.salience_list)
                    winner = self.candidates[winner_index]
                    # <<<< competition <<<<
                    if len(winner) == 0:
                        continue
                    # self.voluntary_attention()
                    # update voluntary attention signal

                    # >>>> working memory - episodic memory interface >>>>
                    # winner is [raw info, encoded_info, task]
                    # if attended info is one of the elements in working memory, add wm_flag, otherwise reset wm
                    reset_fag = 0
                    # if winner_index in [0, 1]:
                    # only vision and hearing needs to indicate the source of signal
                    # only function when the signal is from modules will change the working memory
                    if winner_index == 0:
                        # only vision
                        reset_fag = self.specialists[winner_index].cache_change_flg.value
                        self.specialists[winner_index].cache_change_flg.value = 0
                        self.perception_source.append(self.source_flg[winner_index])
                        self.perception_source.pop(0)
                        print('perception source his: ', self.perception_source)
                        print('the sig is from: ', winner_index)
                        print(len(winner))

                    if reset_fag == 1:
                        print('the working memory holds elements: ', np.shape(self.wm[:]))
                        print('the flags for them are: ', self.wm_flags[:])
                        episodic_element = list()
                        for i in range(len(self.wm_flags[:])):
                            if self.wm_flags[i] >= 0.05:
                                episodic_element.append([self.wm[i].tolist(), self.wm_flags[i]])
                        print('the elements to be stored:', len(episodic_element))
                        if self.perception_source[0] == 0:
                            # the info from last cycle is from external
                            episodic.store(memory=episodic_element, store_mode=0)
                            # store new traces
                        elif self.perception_source[0] == 1:
                            # the info from last cycle is from memory
                            episodic.store(memory=episodic_element, store_mode=1)
                            # store recalled traces

                        # self.refresh_wm()
                        print('GW wipes working memory...')
                        self.wm[:] = list()
                        self.wm_flags[:] = list()
                    # <<<< working memory - episodic memory interface <<<<

                    # >>>> update working memory >>>>
                    if len(winner[0]) != 0:
                        revisited_flg = 0
                        for i in range(len(self.wm[:])):
                            # if winner[0].tolist() == self.wm[i].tolist():
                            try:
                                if np.sum((winner[0] - self.wm[i]) ** 2) < 1000.:
                                    if self.wm_flags[i] < 5.:
                                        self.wm_flags[i] += 5.
                                    self.transient_memory_index = i
                                    revisited_flg = 1
                                    break
                            except ValueError:
                                print(np.shape(winner[0]), np.shape(self.wm[i]), winner_index)
                                pass
                        if revisited_flg == 0:
                            self.wm.append(winner[0])
                            self.wm_flags.append(5.)
                            self.transient_memory_index = len(self.wm[:]) - 1
                    # <<<< update working memory <<<<

                    # >>>> refresh signals >>>>
                    if winner_index == 2:
                        # if language wins
                        # this stresses that the signals from one module to others can be spread
                        # only if it attends consciousness
                        Episodic.cue[:] = Language.cue[:].copy()
                        Episodic.recall_mode[:] = Language.recall_mode[:].copy()
                        GW.cue[:] = Language.cue[:].copy()
                        QA.hold[:] = Language.cue[:].copy()
                        Visual.source_gate.value = language.source_gate.value
                        GW.clock_num.value = language.clock_number.value
                    else:
                        if len(winner[2]) != 0:
                            GW.clock_num.value = 1

                    if self.clock_num.value < 1:
                        self.task = []
                        # Episodic.cue[:] = []
                        # GW.cue[:] = []
                        # QA.hold[:] = []
                        Visual.episodic_recall[:] = list([[], []])
                        Visual.source_gate.value = 0
                        self.clock_num.value = 0

                    # >>>> broadcast >>>>
                    winner = winner[1:]
                    # winner became [encoded_info, task]
                    if len(winner[1]) == 0:
                        if winner_index in [0, 1]:
                            winner[1] = self.task
                    else:
                        self.task = winner[1]

                    signal_pattern = list()
                    for signal in winner:
                        if len(signal) == 0:
                            signal_pattern.append(0)
                        else:
                            signal_pattern.append(1)
                    # specialists = [vision, hearing, language, knowledge, episodic, speak]
                    if signal_pattern == [1, 1]:
                        self.attention[:] = [0, 0, 1, 1, 0, 0, 1, 0]
                    elif signal_pattern == [0, 1]:
                        # enhance the signal from perception
                        self.attention[:] = [1, 1, 0., 0, 0, 0, 0, 0]
                    elif signal_pattern == [1, 0]:
                        self.attention[:] = [0, 0, 0., 0, 0, 0, 0, 0]

                    # with conditions, write into broadcast caches

                    for specialist_index in range(8):
                        if specialist_index == winner_index:
                            GW.broadcast[specialist_index] = []
                        else:
                            if specialist_index == 7:
                                # if the signal is not from knowledge and move_model, QA does not take it
                                if winner_index not in [3, 6]:
                                    GW.broadcast[7] = []
                                else:
                                    GW.broadcast[7] = winner
                            else:
                                GW.broadcast[specialist_index] = winner
                    # if winner_index == 3:
                    #     self.clock_num.value -= 1
                    # print(self.clock_num.value)
                    if self.clock_num.value < 1:
                        self.clock_num.value = 0
                        self.task = []
                    # <<<< broadcast <<<<
                    # <<<< refresh signals <<<<

        except KeyboardInterrupt:
            print('GW saves working memory')
            episodic_element = list()
            for i in range(len(self.wm_flags[:])):
                if self.wm_flags[i] >= 0.05:
                    episodic_element.append([self.wm[i].tolist(), self.wm_flags[i]])
            print('the elements to be stored:', len(episodic_element))
            if self.perception_source[1] == 1:
                source_flg_tmp = 1
            else:
                source_flg_tmp = 0
            episodic.store(memory=episodic_element, store_mode=source_flg_tmp)
            print('GW is exiting')
            exit(0)
# <<<< GW <<<<


class Eyes:

    presented_img_file = Manager().Value(c_char_p, './pepper_comm/photo_pepper_2.JPG')
    sent_msg = Manager().Value(ctypes.c_char_p, '')

    def __init__(self):
        self.msg = 'Error in opening the camera!'

    def monitor(self):
        # capture = cv2.VideoCapture(0)
        # if not capture.isOpened():
        #     print(self.msg)
        #     exit()
        index = 0
        imgs_to_show = os.listdir('pepper_comm/')
        imgs_to_show.sort()
        while True:
            try:
                print(index)
                # ref, frame = capture.read()
                # ref is False if no frame is returned, otherwise True
                # Visual.external_perception[:] = frame
                # print('the perceived is ', self.presented_img_file.value)
                # img = cv2.imread(self.presented_img_file.value)
                img = cv2.imread('pepper_comm/' + imgs_to_show[index//100])
                img = cv2.resize(img, (640, 480))
                # img = cv2.imread(presented_img_file)
                Visual.external_perception[:] = img
                time.sleep(0.1)
                index += 1
                if index//100 > len(imgs_to_show):
                    print('imgs are run out')
                    break
            except KeyboardInterrupt:
                print('Eyes is exiting')
                exit(0)

        # context = zmq.Context()
        # socket = context.socket(zmq.REP)
        # socket.bind("tcp://*:5555")
        # frame_index = 0
        # try:
        #     while True:
        #         # continue
        #         message = socket.recv()
        #         try:
        #             img = cv2.imread('/home/n40729wh/projects/episodicMemory/pepper_comm/photo_pepper.JPG')
        #             # img = cv2.imread('/home/n40729wh/projects/episodicMemory/pepper_comm/non_vision.JPG')
        #             # img = cv2.resize(img, (640, 480))
        #             Visual.external_perception[:] = img
        #             cmd = 'mv /home/n40729wh/projects/episodicMemory/pepper_comm/photo_pepper.JPG ' \
        #                   '/home/n40729wh/projects/episodicMemory/pepper_comm/photo_pepper_' + str(frame_index) + '.JPG'
        #             os.system(cmd)
        #             frame_index += 1
        #         except FileNotFoundError:
        #             pass
        #         # socket.send(b'img processed')
        #         # a = input('stop_point: ')
        #         socket.send_string(self.sent_msg.value)
        #         self.sent_msg.value = ''
        # except KeyboardInterrupt:
        #     print('Eyes is exiting')
        #     exit(0)


class Ears:
    background_amplitude = 33000.
    rate = 4096
    chunk = 1024
    data_format = pyaudio.paInt16
    channels = 1

    def __init__(self):
        self.chunk = 1024
        self.data_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 4096
        self.record_seconds = 10

    def monitor(self):
        # >>>> detect and record audio >>>>
        # detect sentences which are divided by 8-silent chunks with each other
        p = pyaudio.PyAudio()

        stream = p.open(format=self.data_format,
                        channels=self.channels,
                        rate=self.rate,
                        input=True,
                        frames_per_buffer=self.chunk)

        flag = False
        pause_num = 0
        # >>>> detect and record audio >>>>
        print("Microphone starts listening")

        frames = list()
        sentence = list()
        chunk_flags = list()

        # calculate the background sound level
        # the first chunk is abandoned as circuit bias
        background = 0.
        for i in range(20):
            data = stream.read(self.chunk)
            data_int = np.array(struct.unpack(str(len(data)) + 'B', data), dtype='b')
            data_amplitude = librosa.db_to_amplitude(data_int)
            ave_amplitude = np.average(data_amplitude)
            if 4 < i < 19:
                background += ave_amplitude
        ave_background_level = background / 14
        print('the background average level is: %f' % ave_background_level)
        self.background_amplitude = ave_background_level
        while True:
            try:
                data = stream.read(self.chunk)
                data_int = np.array(struct.unpack(str(len(data)) + 'B', data), dtype='b')
                data_amplitude = librosa.db_to_amplitude(data_int)
                ave_amplitude = np.average(data_amplitude)

                if flag is False:
                    if ave_amplitude > 2 * self.background_amplitude:
                        flag = True
                        pause_num = 0
                        sentence.append(data)
                        chunk_flags.append(1)
                else:
                    sentence.append(data)
                    if ave_amplitude > 2 * self.background_amplitude:
                        chunk_flags.append(1)
                        pause_num = 0
                    else:
                        chunk_flags.append(0)
                        pause_num += 1
                        if pause_num >= 8:
                            # refine the voice by clipping the silent chunks
                            start = chunk_flags.index(1, 0, len(chunk_flags))
                            chunk_flags.reverse()
                            end = chunk_flags.index(1, 0, len(chunk_flags))
                            Auditory.external_perception[:] = sentence[start: -end+2]
                            print('sentence detected!')
                            # frames.append(sentence[start:end + 1])
                            # replace this append with feeding to external perception of hearing
                            sentence = list()
                            chunk_flags = list()
                            flag = False
                            pause_num = 0
            except KeyboardInterrupt:
                print('Ears is exiting')
                exit(0)
        # <<<< detect and record audio <<<<


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method("spawn")
    print('this is the entry')
    # ears = Ears()
    # eyes = Eyes()
    # vision = Visual()
    hearing = Auditory()
    # language = Language()
    # knowledge = Knowledge()
    # episodic = Episodic()
    # goal = Goal()
    # move = MoveModel()
    # speak = QA()
    # gw = GW()
    print('initialisation is okay')

    # p0 = Process(target=gw.compete)
    # p1 = Process(target=vision.forward)
    p2 = Process(target=hearing.forward)
    # p3 = Process(target=language.forward)
    # p4 = Process(target=knowledge.forward)
    # p5 = Process(target=episodic.retrieve_)
    # p6 = Process(target=speak.forward)
    # p7 = Process(target=ears.monitor)
    # p8 = Process(target=eyes.monitor)
    # p9 = Process(target=move.forward)
    # p10 = Process(target=goal.wait_goal)
    # p11 = Process(target=episodic.forgetting)
    # p12 = Process(target=gw.decaying)
    print('processes creating is okay')
    # p0.start()
    # p1.start()
    p2.start()
    # p3.start()
    # p4.start()
    # p5.start()
    # p6.start()
    # p7.start()
    # p8.start()
    # p9.start()
    # p10.start()
    # p11.start()
    # p12.start()
    print('all processes are already started')

    # imgs = os.listdir('imgs/')
    # voices = os.listdir('recorded_voices/voices')
    # voices.sort(key=sort_keys_files)
    # imgs.sort(key=sort_keys_files)

    i = 0
    j = 0
    while True:
        try:
            # continue
            try:
                file_path = input('please input the stimuli(e.g., image, audio_words):')
            except NameError:
                print('input is broken')
                continue
            # print([p0.is_alive(), p1.is_alive(), p2.is_alive(),
            #        p3.is_alive(), p4.is_alive(), p5.is_alive(),
            #        p6.is_alive()])
            # one image and one or more words/sounds
            file_names = file_path.split(',')
            # if len(file_names[0]) != 0:
            #     # img = cv2.imread('./imgs/' + file_names[0] + '.jpg')
            #     Eyes.presented_img_file.value = './imgs/' + file_names[0] + '.jpg'
            #     # presented_img_file = './imgs/' + file_names[0] + '.jpg'
            #     # if img is None:
            #     #     img = []
            # words = list()
            # sentence = list()
            # audio_files = file_names[1].strip().split(' ')
            # # print(audio_files)
            # if len(file_names[1]) != 0:
            #     # silence_chunk = load_audio('./recorded_voices/silence_chunk.wav')
            #     # silence_chunk = wav.read('recorded_voices/silence_chunk.wav')
            #     # word_split = silence_chunk + silence_chunk + silence_chunk
            #     for audio in audio_files:
            #         try:
            #             tmp = load_audio('./recorded_voices/speeches/' + audio + '00001.wav')
            #             # tmp = load_audio('./recorded_voices/voices/' + audio + '.wav')
            #         except FileNotFoundError:
            #             tmp = []
            #         words.append(tmp)
            #
            #     for ele in words:
            #         # play_audio(sound_stream=ele)
            #         if len(sentence) == 0:
            #             appendix = ele
            #         else:
            #             appendix = word_split + ele
            #         sentence = sentence + appendix
            # Visual.external_perception[:] = img
            audio = load_audio('./recorded_voices/speeches/' + file_names[1].strip() + '.wav')
            Auditory.external_perception[:] = audio
        except KeyboardInterrupt:
            print('All modules will store the configuration first before exiting...')
            # while any([p0.is_alive(), p1.is_alive(), p2.is_alive(),
            #            p3.is_alive(), p4.is_alive(), p5.is_alive(),
            #            p6.is_alive()]) is True:
            #     continue
            break