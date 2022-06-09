import os
import random
import sys
import pickle
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
import math
import matplotlib.pyplot as plt
import numpy as np
import pygame as pg
import tensorflow as tf
from gensim.models import KeyedVectors
from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4
from pygame.locals import *
from translate import Translator #https://pypi.org/project/translate/
#https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
from nltk.corpus import wordnet 
#https://medium.com/pyladies-taiwan/nltk-%E5%88%9D%E5%AD%B8%E6%8C%87%E5%8D%97-%E4%B8%89-%E5%9F%BA%E6%96%BC-wordnet-%E7%9A%84%E8%AA%9E%E7%BE%A9%E9%97%9C%E4%BF%82%E8%A1%A8%E7%A4%BA%E6%B3%95-%E4%B8%8A%E4%B8%8B%E4%BD%8D%E8%A9%9E%E7%B5%90%E6%A7%8B%E7%AF%87-4874fb9b167a
from gtts import gTTS # text to speech
from PIL import Image, ImageFont, ImageDraw #https://yugioh-card.linziyou.info/
os.environ["CUDA_VISIBLE_DEVICES"] = ""

class OptionBox():

    def __init__(self, x, y, w, h, color, highlight_color, font, option_list, selected = 0):
        self.color = color
        self.highlight_color = highlight_color
        self.rect = pg.Rect(x, y, w, h)
        self.font = font
        self.option_list = option_list
        self.selected = selected
        self.draw_menu = False
        self.menu_active = False
        self.active_option = -1

    def draw(self, surf):
        pg.draw.rect(surf, self.highlight_color if self.menu_active else self.color, self.rect)
        pg.draw.rect(surf, (0, 0, 0), self.rect, 2)
        msg = self.font.render(self.option_list[self.selected], 1, (0, 0, 0))
        surf.blit(msg, msg.get_rect(center = self.rect.center))

        if self.draw_menu:
            for i, text in enumerate(self.option_list):
                rect = self.rect.copy()
                rect.y += (i+1) * self.rect.height
                pg.draw.rect(surf, self.highlight_color if i == self.active_option else self.color, rect)
                msg = self.font.render(text, 1, (0, 0, 0))
                surf.blit(msg, msg.get_rect(center = rect.center))
            outer_rect = (self.rect.x, self.rect.y + self.rect.height, self.rect.width, 
                          self.rect.height * len(self.option_list))
            pg.draw.rect(surf, (0, 0, 0), outer_rect, 2)

    def update(self, event_list):
        mpos = pg.mouse.get_pos()
        self.menu_active = self.rect.collidepoint(mpos)
        
        self.active_option = -1
        for i in range(len(self.option_list)):
            rect = self.rect.copy()
            rect.y += (i+1) * self.rect.height
            if rect.collidepoint(mpos):
                self.active_option = i
                break

        if not self.menu_active and self.active_option == -1:
            self.draw_menu = False

        for event in event_list:
            if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                if self.menu_active:
                    self.draw_menu = not self.draw_menu
                elif self.draw_menu and self.active_option >= 0:
                    self.selected = self.active_option
                    self.draw_menu = False
                    return self.active_option
        return self.selected

class Button():
    def __init__(self, image, pos, text_input, font, base_color, hover_color):
        self.image = image
        self.x = pos[0]
        self.y = pos[1]
        self.font = font
        self.base_color = base_color
        self.hover_color = hover_color
        self.text_input = text_input
        self.text = self.font.render(self.text_input, True, self.base_color)
        if self.image is None:
            self.image = self.text
        self.rect = self.image.get_rect(center=(self.x, self.y))
        self.text_rect = self.text.get_rect(center=(self.x, self.y))

    def update(self, screen):
        if self.image is not None:
            screen.blit(self.image, self.rect)
        screen.blit(self.text, self.text_rect)

    def checkForInput(self, pos):
        if pos[0] in range(self.rect.left, self.rect.right) and pos[1] in range(self.rect.top, self.rect.bottom):
            return True
        return False

    def changeColor(self, pos):
        if pos[0] in range(self.rect.left, self.rect.right) and pos[1] in range(self.rect.top, self.rect.bottom):
            self.text = self.font.render(self.text_input, True, self.hover_color)
        else:
            self.text = self.font.render(self.text_input, True, self.base_color)

class Player(pg.sprite.Sprite):
    def __init__(self):
        self.pg = pg
        self.horse = np.random.randint(low = 1, high = 5)
        self.total_room = 1 # initial total room number, but will change by number of error_history
        self.room = 1 # initial room number
        self.direction = 'right' # initial player direction
        self.image = self.pg.image.load(f'pony_player/player{self.horse}_0.png').convert_alpha()
        self.image = self.pg.transform.scale(self.image, (100,80))
        self.rect = self.image.get_rect()
        self.rect.top = 500
        self.speed = 0

        self.walk_left_images = [self.pg.image.load(f'pony_player/player{self.horse}_{i}.png').convert_alpha() 
                                 for i in range(16)]
        self.walk_left_images = [self.pg.transform.scale(self.walk_left_images[i], (100,80)) 
                                 for i in range(16)]
        self.walk_right_images = [self.pg.transform.flip(
                                  self.pg.image.load(f'pony_player/player{self.horse}_{i}.png').convert_alpha(), 
                                  True, False)
                                  for i in range(16)]
        self.walk_right_images = [self.pg.transform.scale(self.walk_right_images[i], (100,80)) 
                                  for i in range(16)]

    def update(self):
        if self.speed:
            self.rect.left += self.speed
            if self.direction == 'left':
                self.index = (self.index + 1) % len(self.walk_left_images)
            elif self.direction == 'right':
                self.index = (self.index + 1) % len(self.walk_right_images)
        else:
            self.index = 0

        if self.direction == 'left':
            self.image = self.walk_left_images[self.index]
        elif self.direction == 'right':
            self.image = self.walk_right_images[self.index]

class Main_function():
    def __init__(self, config):
        # 初始化
        self.pg = pg   
        self.config = config
        self.pg.init()
        self.pg.mixer.init()
        self.screen = self.pg.display.set_mode((self.config['Game_WIDTH'], 
                                                self.config['Game_HEIGHT']))
        self.pg.display.set_caption("LEARN WITH THE LITTLE PONY!")
        self.pg.mixer.music.set_volume(self.config['music_volume'])
        self.bg_music = self.config['music']
        self.bg_museum_music = self.config['museum_music']
        self.bg_museum_scene_music = self.config['ro_music']
        # 讀取圖片
        self.background_img = self.pg.image.load(self.config['bg_img']).convert()
        self.player_img = self.pg.image.load(self.config['player_img']).convert_alpha()
        self.player_mini_img = self.pg.transform.scale(self.player_img, 
                                                       tuple(self.config['pg_scale']))
        self.pg.display.set_icon(self.player_mini_img)
        self.correct_img = self.pg.image.load(self.config['correct_img']).convert()
        self.error_img = self.pg.image.load(self.config['error_img']).convert()

        # 讀取YOLO
        self.classes_list = config['yolo_classes_list']
        self.colors = self.config['yolo_colors']
        self.detect_num = self.config['yolo_detect_num']
        self.model = YOLOv4(
            input_shape=(self.config['yolo_height'], 
            self.config['yolo_width'], self.config['yolo_channel']),
            anchors=YOLOV4_ANCHORS,
            num_classes=self.config['yolo_num_classes'],
            training=self.config['yolo_training'],
            yolo_max_boxes=self.config['yolo_max_boxes'],
            yolo_iou_threshold=self.config['yolo_iou_threshold'],
            yolo_score_threshold=self.config['yolo_score_threshold'],
        )
        self.model.load_weights(self.config['yolo_weight_path'])
        self.yolo_detect_image_path = self.config['yolo_detect_image_path']

        # 讀取word2vector model
        self.w2v_model = KeyedVectors.load("word2vec.model", mmap='r')

        # 讀取設定
        self.clock = self.pg.time.Clock()
        self.prob_pic_size = self.config['prob_pic_size']
        self.title_text = ""
        self.translator = None
        self.WIDTH = self.config['Game_WIDTH']
        self.HEIGHT = self.config['Game_HEIGHT']
        self.FPS = self.config['FPS']
        self.user_font_size = self.config['user_font_size']
        self.exhibition_font_size = self.config['exhibition_font_size']
        self.offset_list = self.config['offset_list']
        self.rw_def_offset = self.config['rw_def_offset']
        self.prob_pic = None
        self.WIDTH_offset = self.config['WIDTH_offset']
        self.HEIGHT_offset = self.config['HEIGHT_offset']
        self.answer_block_width = self.config['answer_block_width']
        self.user_answer_history = []

        self.Game_language = None
        self.Game_language_code = None
        self.Game_language_original_code = self.config['Game_language_original_code']
        self.speech_original_path =  f"Speechs/{self.Game_language_original_code}/"
        self.speech_tg_path =  None
        self.language_list = OptionBox(700, 245, 160, 40, (150, 150, 150), 
                                       (100, 200, 255), 
                                       self.pg.font.SysFont(None, 30), 
                                       ["Deutsch", "French"])

        # check error history file exists or not
        if not os.path.exists(config['user_error_answer_history']):
            self.user_error_answer_history = {}
            self.user_error_answer_history['string'] = []
            self.user_error_answer_history['image'] = []

            with open(self.config['user_error_answer_history'], 'wb') as f:
                pickle.dump(self.user_error_answer_history,f)
        else:
            with open(self.config['user_error_answer_history'], 'rb') as f:
                self.user_error_answer_history = pickle.load(f)             

        # 讀取顏色設定
        self.GOLD = tuple(self.config['GOLD'])
        self.RED = tuple(self.config['RED'])
        self.YELLOW = tuple(self.config['YELLOW'])
        self.GREEN = tuple(self.config['GREEN'])
        self.W_GREEN = tuple(self.config['W_GREEN'])
        self.D_GREEN = tuple(self.config['D_GREEN'])
        self.PURPLE = tuple(self.config['PURPLE'])
        self.D_PURPLE = tuple(self.config['D_PURPLE'])
        self.PINK = tuple(self.config['PINK'])
        self.S_PINK = tuple(self.config['S_PINK'])
        self.WHITE = tuple(self.config['WHITE'])
        self.BLACK = tuple(self.config['BLACK'])
        self.NAVYBLUE = tuple(self.config['NAVYBLUE'])
        self.card_color_list = [self.RED, self.NAVYBLUE]

        # 讀取卡片設定
        self.fake_card_list = ['card_apple', 'card_banana', 'card_orange', 
                               'card_lemon', 'card_papaya']
        self.card_pixel_list = None
        self.real_card_list = None
        self.card_width_num = 4
        self.card_height_num = 4
        self.card_height = 150
        self.card_width = 103
        self.check_card_flag = None
        self.check_card_timing = 10
        self.click_card_history = None
        self.card_score = 0
        self.delay_time = config['Delay_time']
        self.Default_YuGiOh_Card_path = config['Default_YuGiOh_Card']
        self.Default_YuGiOh_Card_font = config['Default_YuGiOh_Card_font']
        self.Default_YuGiOh_Card_font_size = config['Default_YuGiOh_Card_font_size']
        self.Default_YuGiOh_Card_font_width = config['Default_YuGiOh_Card_font_width']
        self.Default_YuGiOh_Card_font_height = config['Default_YuGiOh_Card_font_height']
        self.Card_Back = self.pg.transform.scale(self.pg.image.load(config['Card_Back']).convert(), 
                                                 (self.card_width, self.card_height))
        self.Card_Apple = self.pg.transform.scale(self.pg.image.load(config['Card_Apple']).convert(), 
                                                 (self.card_width, self.card_height))
        self.Card_Banana = self.pg.transform.scale(self.pg.image.load(config['Card_Banana']).convert(), 
                                                 (self.card_width, self.card_height))
        self.Card_Orange = self.pg.transform.scale(self.pg.image.load(config['Card_Orange']).convert(), 
                                                 (self.card_width, self.card_height))
        self.Card_Lemon = self.pg.transform.scale(self.pg.image.load(config['Card_Lemon']).convert(), 
                                                 (self.card_width, self.card_height))
        self.Card_Papaya = self.pg.transform.scale(self.pg.image.load(config['Card_Papaya']).convert(), 
                                                 (self.card_width, self.card_height))
        self.width_pixel_list = [self.card_width * i for i in range(self.card_width_num)]
        self.height_pixel_list = [self.card_height * i for i in range(self.card_height_num)]
        self.Card_Image = [self.Card_Apple, self.Card_Banana, self.Card_Orange, 
                           self.Card_Lemon, self.Card_Papaya]
        self.card_status = None

        #讀取博物館設定
        self.pony_speed = self.config['pony_speed']
        self.yolo_museum_frame_size = self.config['yolo_museum_frame_size']
        self.yolo_museum_exhibition_width_size = self.config['yolo_museum_exhibition_width_size']
        self.yolo_museum_exhibition_height_size = self.config['yolo_museum_exhibition_height_size']
        self.background_museum_img = self.pg.image.load(self.config['bg_img_museum']).convert()
        self.museum_frame = self.pg.image.load(self.config['bg_img_frame']).convert_alpha()
        self.museum_frame = self.pg.transform.scale(self.museum_frame, 
                                                    (self.yolo_museum_frame_size, 
                                                     self.yolo_museum_frame_size))
        self.meseum_exhibition = self.pg.image.load(self.config['bg_img_exhibition']).convert_alpha()
        self.meseum_exhibition = self.pg.transform.scale(self.meseum_exhibition, 
                                                        (self.yolo_museum_exhibition_width_size, 
                                                         self.yolo_museum_exhibition_height_size))
        self.meseum_exhibition_height = self.config['yolo_meseum_exhibition_height']
        self.yolo_museum_frame_height = self.config['yolo_museum_frame_height']
        self.yolo_museum_width_interval = self.config['yolo_museum_width_interval']

    def plot_results(self, pil_img, boxes, scores, classes, classes_list, colors):
        text_list = []
        plt.figure(figsize=(16,10))
        plt.imshow(pil_img)
        ax = plt.gca()

        for (xmin, ymin, xmax, ymax), score, cl in zip(boxes.tolist(), scores.tolist(), classes.tolist()):
            if score > 0:
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                        fill=False, color=colors[cl % 6], linewidth=3))
                # text = f'{classes_list[cl]}: {score:0.2f}'
                text = f'{classes_list[cl]}'
                ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
                text_list.append(text)
        plt.axis('off')
        # plt.show()
        plt.savefig(self.yolo_detect_image_path,bbox_inches='tight')
        plt.close()
        
        text_list = [i.split(':')[0].upper() for i in text_list]
        return text_list

    def init(self):
        self.play_music()
        self.FPS = 60
        while True:
            self.screen.blit(self.background_img, (0, 0))
            MOUSE_POS = self.pg.mouse.get_pos()
            self.draw_text_center(self.get_font(50), 'PONY PLAY TIME',self.GOLD, self.WIDTH//2, self.HEIGHT//2-200)
            
            # button
            play_button = Button(image=self.pg.image.load("assets/Menu_Rect.png"), pos=(self.WIDTH//2, self.HEIGHT//2),
                                text_input="PLAY", font=self.get_font(75), base_color=self.W_GREEN, 
                                hover_color=self.WHITE)
            History_button = Button(image=self.pg.image.load("assets/Menu_Rect.png"), pos=(self.WIDTH//2, self.HEIGHT//2+120),
                                text_input="Museum", font=self.get_font(50), base_color=self.W_GREEN, 
                                hover_color=self.WHITE)
            quit_button = Button(image=self.pg.image.load("assets/Menu_Rect.png"), pos=(self.WIDTH//2, self.HEIGHT//2+240),
                                text_input="QUIT", font=self.get_font(75), base_color=self.W_GREEN, 
                                hover_color=self.WHITE)

            for button in [play_button, History_button, quit_button]:
                button.changeColor(MOUSE_POS)
                button.update(self.screen)
            # event
            event_list = self.pg.event.get()
            for event in event_list:
                if event.type == self.pg.QUIT:
                    self.pg.quit()
                    sys.exit()
                if event.type == self.pg.MOUSEBUTTONDOWN:
                    if play_button.checkForInput(MOUSE_POS):
                        self.games()
                    if History_button.checkForInput(MOUSE_POS):
                        self.history_museum()
                    if quit_button.checkForInput(MOUSE_POS):
                        self.pg.quit()
                        sys.exit()
                        
            selected_language = self.language_list.update(event_list)
            if selected_language == 0:
                self.Game_language = "Deutsch"
                self.Game_language_code = "de"
            elif selected_language == 1:
                self.Game_language = "French"
                self.Game_language_code = "fr"

            self.speech_tg_path =  f"Speechs/{self.Game_language_code}/"
            self.translator = Translator(to_lang = self.Game_language_code)
            self.language_list.draw(self.screen)
            self.clock.tick(self.FPS)
            self.pg.display.update()
            
        return None

    def play_music(self, museum = False, speechs_path = None):
        if speechs_path is None:
            if museum is False:
                self.pg.time.wait(2000)
                self.pg.mixer.music.load(self.bg_music)
                self.pg.mixer.music.play(-1)
            else:
                self.pg.mixer.music.load(self.bg_museum_music)
                self.pg.mixer.music.play(-1)
        else:
            self.pg.mixer.music.load(speechs_path)
            self.pg.mixer.music.play()
        return None

    def get_speech_path(self, rw, path = None, language_code='en'):
        if path == None:save_path = self.speech_original_path
        else:save_path = path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        speechs_path = f"{save_path}{rw}_{language_code}.mp3"
        return speechs_path

    def get_font(self, size):
        return self.pg.font.Font(os.path.join("assets", "font.ttf"), size)

    def user_font(self, size):
        return self.pg.font.Font(None, size)

    def draw_text_center(self, font, text, color, center_x, center_y):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=(center_x, center_y))
        self.screen.blit(text_surface, text_rect)   
        return None

    def draw_text(self, font, text, color, x, y):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(left=x, top=y)
        self.screen.blit(text_surface, text_rect)
        return None

    def games(self):

        self.prob_pic = None
        self.title_text = 'Plz Load JPG Image'
        self.FPS = 60

        while True:
            self.screen.fill(self.D_PURPLE)
            MOUSE_POS = self.pg.mouse.get_pos()

            self.draw_text_center(self.get_font(40), self.title_text, self.PINK, self.WIDTH//2, self.HEIGHT//2-200)

            # button
            play_back = Button(image=None, pos=(self.WIDTH-200, self.HEIGHT-60), text_input="BACK",
                            font=self.get_font(75), base_color=self.WHITE, hover_color=self.GREEN)

            LoadImage = Button(image=None, pos=(self.WIDTH//2, self.HEIGHT//2-80), text_input="LoadImage",
                        font=self.get_font(60), base_color=self.WHITE, hover_color=self.GREEN)
                        
            match = Button(image=None, pos=(self.WIDTH//2, self.HEIGHT//2+20), text_input="MATCH",
                        font=self.get_font(60), base_color=self.WHITE, hover_color=self.GREEN)
            card = Button(image=None, pos=(self.WIDTH//2, self.HEIGHT//2+100), text_input="CARD",
                            font=self.get_font(60), base_color=self.WHITE, hover_color=self.GREEN)

            if self.prob_pic != None:
                for button_variable in [match, card]:
                    button_variable.changeColor(MOUSE_POS)
                    button_variable.update(self.screen)

            for button_variable in [play_back, LoadImage]:
                button_variable.changeColor(MOUSE_POS)
                button_variable.update(self.screen)

            # event
            for event in self.pg.event.get():
                if event.type == self.pg.QUIT:
                    self.pg.quit()
                    sys.exit()
                elif event.type == self.pg.MOUSEBUTTONDOWN:
                    if play_back.checkForInput(MOUSE_POS):
                        self.init()
                    elif LoadImage.checkForInput(MOUSE_POS):
                        root = Tk()
                        root.geometry("1x1") # windows size
                        root.withdraw() # hide tkinter windows
                        prob_pic_path = askopenfilename()
                        root.destroy() # close tkinter windows
                        # print(prob_pic_path)

                        if os.path.exists(prob_pic_path):# if user not choose image
                            image = tf.io.read_file(prob_pic_path)
                            image = tf.image.decode_image(image)
                            image = tf.image.resize(image, (self.config['yolo_height'], self.config['yolo_width']))
                            image = tf.expand_dims(image, axis=0) / 255.0

                            boxes, scores, classes, valid_detections = self.model.predict(image)
                            valid_detections = int(valid_detections) ## array([13]) -> 13

                            boxes = boxes[0][:valid_detections]
                            scores = scores[0][:valid_detections]
                            classes = classes[0][:valid_detections]

                            combine_dict, index_dict = ({} for _ in range(2))
                            for idx,(i,j) in enumerate(zip(classes.astype(int),scores)):
                                if f'{i}' not in combine_dict.keys(): 
                                    combine_dict[f'{i}'] = j
                                    index_dict[f'{i}'] = idx
                                elif j > combine_dict[f'{i}']:
                                    combine_dict[f'{i}'] = j
                                    index_dict[f'{i}'] = idx
                                    
                            # print(combine_dict)

                            few_boxes = np.array([s for idx,s in enumerate(boxes) if idx in list(index_dict.values())])[:self.detect_num]
                            few_scores = np.array([s for idx,s in enumerate(scores) if idx in list(index_dict.values())])[:self.detect_num]
                            few_classes = np.array([s for idx,s in enumerate(classes) if idx in list(index_dict.values())])[:self.detect_num]

                            temp_text_list = self.plot_results(image[0], few_boxes * [self.config['yolo_width'], 
                                                               self.config['yolo_height'], self.config['yolo_width'], 
                                                               self.config['yolo_height']], few_scores, 
                                                               few_classes.astype(int), self.classes_list, 
                                                               self.colors)

                            if len(temp_text_list) == 3:
                                text_list = [i.capitalize() for i in temp_text_list.copy()]
                                self.prob_pic = self.pg.image.load(self.yolo_detect_image_path).convert()
                                self.prob_pic = self.pg.transform.scale(self.prob_pic, (self.WIDTH//2, self.HEIGHT//2))
                                self.title_text = 'CHOOSE A GAME'
                            else:
                                self.title_text = 'Try Other JPG Image'
                                print(len(temp_text_list))
                                continue

                    elif match.checkForInput(MOUSE_POS):
                        if self.prob_pic != None:
                            self.matching(text_list)
                    elif card.checkForInput(MOUSE_POS):
                        self.card(text_list)
            self.clock.tick(self.FPS)
            self.pg.display.update()
        return None

    def matching(self, recognize_word):
        # initialize history
        self.user_answer_history = []

        # answering zone
        ans_block1 = self.pg.Rect(self.WIDTH-self.WIDTH_offset, 70,  self.WIDTH//3, self.HEIGHT//8)
        ans_block2 = self.pg.Rect(self.WIDTH-self.WIDTH_offset, 170, self.WIDTH//3, self.HEIGHT//8)
        ans_block3 = self.pg.Rect(self.WIDTH-self.WIDTH_offset, 270, self.WIDTH//3, self.HEIGHT//8)

        word_rect1 = self.pg.Rect(self.WIDTH-self.WIDTH_offset, 70, self.WIDTH//3, 28)
        word_rect2 = self.pg.Rect(self.WIDTH-self.WIDTH_offset, 170, self.WIDTH//3, 28)
        word_rect3 = self.pg.Rect(self.WIDTH-self.WIDTH_offset, 270, self.WIDTH//3, 28)

        # recogize_word in target language, and upper the first letter.
        ans = [self.translator.translate(i).capitalize() for i in recognize_word]

        rand_ans_pos = [self.WIDTH//2-self.WIDTH_offset, self.WIDTH//2-100, self.WIDTH//2+200]
        random.shuffle(rand_ans_pos)
        ans_rect1 = self.pg.Rect(rand_ans_pos[0], self.HEIGHT-self.HEIGHT_offset, 
                                 self.answer_block_width, self.user_font_size)
        ans_rect2 = self.pg.Rect(rand_ans_pos[1], self.HEIGHT-self.HEIGHT_offset, 
                                 self.answer_block_width, self.user_font_size)
        ans_rect3 = self.pg.Rect(rand_ans_pos[2], self.HEIGHT-self.HEIGHT_offset, 
                                 self.answer_block_width, self.user_font_size)

        language_block = self.pg.Surface((self.WIDTH-100, self.HEIGHT//8)).convert()
        language_block.fill(self.PINK)
        pic_edge = self.pg.Surface((self.WIDTH//2+10, self.HEIGHT//2+10)).convert()
        pic_edge.fill(self.GOLD)

        move1,move2,move3 = (False for _ in range(3))

        while True:
            self.screen.fill(self.D_PURPLE)
            self.screen.blit(pic_edge, (45, 45))
            self.screen.blit(self.prob_pic, (self.prob_pic_size, self.prob_pic_size))
            self.screen.blit(language_block, (50, self.HEIGHT//2+100))
            MOUSE_POS = self.pg.mouse.get_pos()
            play_back = Button(image=None, pos=(self.WIDTH-200, self.HEIGHT-60), text_input="BACK",
                            font=self.get_font(75), base_color=self.WHITE, hover_color=self.GREEN)
            play_next = Button(image=None, pos=(200, self.HEIGHT-60), text_input="NEXT",
                            font=self.get_font(75), base_color=self.WHITE, hover_color=self.GREEN)

            for play_variable in [play_back, play_next]:
                play_variable.changeColor(MOUSE_POS)
                play_variable.update(self.screen)

            # answering zone
            ans_block_list = [ans_block1, ans_block2, ans_block3]
            for ans_block_each in ans_block_list:
                self.pg.draw.rect(self.screen, self.WHITE, ans_block_each)

            # recognize_word
            word_rect_list = [word_rect1, word_rect2, word_rect3]
            for idx,word_rect_each in enumerate(word_rect_list):
                self.pg.draw.rect(self.screen, self.D_GREEN, word_rect_each)
                self.draw_text(self.user_font(self.user_font_size), recognize_word[idx], 
                               self.BLACK, word_rect_each.x+5, word_rect_each.y+5)
            # ans
            ans_list = [ans_rect1, ans_rect2, ans_rect3]
            for idx,ans_rect_each in enumerate(ans_list):
                self.pg.draw.rect(self.screen, self.GREEN, ans_rect_each)
                self.draw_text(self.user_font(self.user_font_size), ans[idx], self.BLACK, 
                               ans_rect_each.x+5, ans_rect_each.y+5)

            # choosing language
            instruction = f'Match the {self.Game_language} word to the corresponding English word'
            self.draw_text(self.user_font(self.user_font_size), instruction, self.BLACK, 
                           self.WIDTH//2-370, self.HEIGHT//2+70)                              

            self.pg.display.flip()
            self.clock.tick(self.FPS)
            self.pg.display.update()

            # event
            for event in self.pg.event.get():
                if event.type == self.pg.QUIT:
                    self.pg.quit()
                    sys.exit()
                elif event.type == self.pg.MOUSEBUTTONDOWN:
                    if play_back.checkForInput(MOUSE_POS):self.games()
                    elif ans_rect1.collidepoint(event.pos):move1 = True
                    elif ans_rect2.collidepoint(event.pos):move2 = True
                    elif ans_rect3.collidepoint(event.pos):move3 = True
                    elif play_next.checkForInput(MOUSE_POS):
                        #record history of user answer 
                        for i,block in enumerate(ans_block_list):
                            for j,answer in enumerate(ans_list):
                                if block.contains(answer):
                                    self.user_answer_history.append(ans[j])
                        # print(self.user_answer_history)
                        self.matching_result(recognize_word, ans)
                elif event.type == self.pg.MOUSEBUTTONUP:
                    move1, move2, move3 = (False for _ in range(3))
                elif event.type == self.pg.MOUSEMOTION:
                    if move1:ans_rect1.move_ip(event.rel)
                    elif move2:ans_rect2.move_ip(event.rel)
                    elif move3:ans_rect3.move_ip(event.rel)
        return None

    def matching_result(self, recognize_word, ans):
        if not os.path.exists(self.yolo_detect_image_path):
            self.pg.quit()
            sys.exit()           
        yolo_image_array = np.array(Image.open(self.yolo_detect_image_path))

        #rw = recognize_word, get rw definition
        rw_define = ([wordnet.synset(f'{i}.n.01').definition() 
                     if len(wordnet.synsets(i)) != 0 else '' 
                     for i in recognize_word])

        # check there had 3 element in rw_define list
        assert len(rw_define) == 3

        while True:
            self.screen.fill(self.D_PURPLE)
            MOUSE_POS = self.pg.mouse.get_pos()
            menu_back = Button(image=None, pos=(self.WIDTH-200, self.HEIGHT-60), text_input="MENU",
                                font=self.get_font(50), base_color=self.WHITE, hover_color=self.GREEN)
            play_again = Button(image=None, pos=(200, self.HEIGHT-60), text_input="AGAIN",
                                font=self.get_font(50), base_color=self.WHITE, hover_color=self.GREEN)
            menu_back.changeColor(MOUSE_POS)
            menu_back.update(self.screen)
            play_again.changeColor(MOUSE_POS)
            play_again.update(self.screen)

            # rw = YOLO detect recognize_word
            play_rw_speech1 = Button(image=None, pos=((self.WIDTH//2)+50, 
                                    self.HEIGHT//4 - (self.offset_list[2])), text_input="Speech",
                                    font=self.get_font(10), base_color=self.WHITE, hover_color=self.GREEN)
            play_rw_speech2 = Button(image=None, pos=((self.WIDTH//2)+50, 
                                    self.HEIGHT//4 - (self.offset_list[1])), text_input="Speech",
                                    font=self.get_font(10), base_color=self.WHITE, hover_color=self.GREEN)
            play_rw_speech3 = Button(image=None, pos=((self.WIDTH//2)+50, 
                                self.HEIGHT//4 - (self.offset_list[0])), text_input="Speech",
                                font=self.get_font(10), base_color=self.WHITE, hover_color=self.GREEN)
            # tg = target language

            play_tg_speech1 = Button(image=None, pos=((self.WIDTH//1.3), 
                                    self.HEIGHT//1.5 + self.offset_list[2]), text_input="Speech",
                                    font=self.get_font(10), base_color=self.WHITE, hover_color=self.GREEN)
            play_tg_speech2 = Button(image=None, pos=((self.WIDTH//1.3), 
                                    self.HEIGHT//1.5 + self.offset_list[1]), text_input="Speech",
                                    font=self.get_font(10), base_color=self.WHITE, hover_color=self.GREEN)
            play_tg_speech3 = Button(image=None, pos=((self.WIDTH//1.3), 
                                    self.HEIGHT//1.5 + self.offset_list[0]), text_input="Speech",
                                    font=self.get_font(10), base_color=self.WHITE, hover_color=self.GREEN)  


            for speech_variable in [play_rw_speech1, play_rw_speech2, play_rw_speech3]:
                speech_variable.changeColor(MOUSE_POS)
                speech_variable.update(self.screen)

            for speech_variable in [play_tg_speech1, play_tg_speech2, play_tg_speech3]:
                speech_variable.changeColor(MOUSE_POS)
                speech_variable.update(self.screen) 

            #Show message box of rw_define, rw = recognize_word
            if len(rw_define[0]) != 0:
                show_rw_def1 = Button(image=None, pos=((self.WIDTH//2)-50, 
                                      self.HEIGHT//4 - (self.offset_list[2])), text_input="Definition",
                                      font=self.get_font(10), base_color=self.WHITE, hover_color=self.GREEN)

                show_rw_def1.changeColor(MOUSE_POS)
                show_rw_def1.update(self.screen)
            if len(rw_define[1]) != 0:
                show_rw_def2 = Button(image=None, pos=((self.WIDTH//2)-50, 
                                      self.HEIGHT//4 - (self.offset_list[1])), text_input="Definition",
                                      font=self.get_font(10), base_color=self.WHITE, hover_color=self.GREEN)

                show_rw_def2.changeColor(MOUSE_POS)
                show_rw_def2.update(self.screen)
            if len(rw_define[2]) != 0:
                show_rw_def3 = Button(image=None, pos=((self.WIDTH//2)-50, 
                                      self.HEIGHT//4 - (self.offset_list[0])), text_input="Definition",
                                      font=self.get_font(10), base_color=self.WHITE, hover_color=self.GREEN)                                   
                show_rw_def3.changeColor(MOUSE_POS)
                show_rw_def3.update(self.screen)

            # event
            for event in self.pg.event.get():
                if event.type == self.pg.QUIT:
                    with open(self.config['user_error_answer_history'], 'wb') as f:
                        pickle.dump(self.user_error_answer_history,f)
                    self.pg.quit()
                    sys.exit()
                elif event.type == self.pg.MOUSEBUTTONDOWN:
                    if menu_back.checkForInput(MOUSE_POS):
                        with open(self.config['user_error_answer_history'], 'wb') as f:
                            pickle.dump(self.user_error_answer_history,f)                        
                        self.init()
                    if play_again.checkForInput(MOUSE_POS):
                        with open(self.config['user_error_answer_history'], 'wb') as f:
                            pickle.dump(self.user_error_answer_history,f)                          
                        self.matching(recognize_word)

                    if len(rw_define[0]) != 0:
                        if show_rw_def1.checkForInput(MOUSE_POS):
                            Tk().wm_withdraw() #to hide the main window
                            messagebox.showinfo(f'Definition of {recognize_word[0]}',f'{rw_define[0]}')
                    if len(rw_define[1]) != 0:
                        if show_rw_def2.checkForInput(MOUSE_POS):
                            Tk().wm_withdraw() #to hide the main window
                            messagebox.showinfo(f'Definition of {recognize_word[1]}',f'{rw_define[1]}')                            
                    if len(rw_define[2]) != 0:
                        if show_rw_def3.checkForInput(MOUSE_POS):
                            Tk().wm_withdraw() #to hide the main window
                            messagebox.showinfo(f'Definition of {recognize_word[2]}',f'{rw_define[2]}')    

                    for rw, speech_variable in zip(recognize_word, [play_rw_speech1, play_rw_speech2, play_rw_speech3]):
                        if speech_variable.checkForInput(MOUSE_POS):
                            speechs_path = self.get_speech_path(rw)
                            if not os.path.exists(speechs_path):
                                speech_obj = gTTS(text = rw, 
                                                lang = self.Game_language_original_code, 
                                                slow = False)
                                speech_obj.save(speechs_path)
                                self.play_music(speechs_path = speechs_path)
                                self.play_music(speechs_path = None)
                            else:
                                self.play_music(speechs_path = speechs_path)
                                self.play_music(speechs_path = None)

                    #[::-1] is inverse list element
                    for an, speech_variable in zip(ans[::-1], [play_tg_speech1, play_tg_speech2, play_tg_speech3]):
                        if speech_variable.checkForInput(MOUSE_POS):
                            speechs_path = self.get_speech_path(an, 
                                                                path = self.speech_tg_path, 
                                                                language_code = self.Game_language_code)
                            if not os.path.exists(speechs_path):
                                speech_obj = gTTS(text = an, 
                                                lang = self.Game_language_original_code, 
                                                slow = False)
                                speech_obj.save(speechs_path)
                                self.play_music(speechs_path = speechs_path)
                                self.play_music(speechs_path = None)
                            else:
                                self.play_music(speechs_path = speechs_path)
                                self.play_music(speechs_path = None)

            # draw titles
            self.draw_text_center(self.user_font(self.user_font_size), 'English Words', self.S_PINK, 
                                  self.WIDTH//2, self.HEIGHT//5 -100)

            self.draw_text_center(self.user_font(self.user_font_size), f'Your {self.Game_language} answer', 
                                  self.S_PINK, self.WIDTH//4, self.HEIGHT//2 -50)   

            self.draw_text_center(self.user_font(self.user_font_size), f'Correct {self.Game_language} answer', 
                                  self.S_PINK, self.WIDTH//1.3, self.HEIGHT//2 -50)

            # draw three words for English and correct answer
            for an, rw, ol in zip(ans, recognize_word, self.offset_list):
                self.draw_text_center(self.user_font(self.user_font_size), rw, self.S_PINK, 
                                      self.WIDTH//2, self.HEIGHT//4.5 + ol)
                self.draw_text_center(self.user_font(self.user_font_size), an, self.S_PINK, 
                                      self.WIDTH//1.3, self.HEIGHT//1.6 + ol)                                     

            # draw thress words for user answer, since user may not give any answer.
            for idx, (rw, ua, ol) in enumerate(zip(recognize_word, self.user_answer_history, self.offset_list)):

                self.draw_text_center(self.user_font(self.user_font_size), ua, self.S_PINK, 
                                      self.WIDTH//4, self.HEIGHT//1.6 + ol)

                # image for correct and in correct
                if ans[idx] == ua:
                    self.screen.blit(self.correct_img, (self.WIDTH//2, self.HEIGHT//1.65 + ol))
                else:
                    self.screen.blit(self.error_img, (self.WIDTH//2, self.HEIGHT//1.65 + ol))
                    record_string = f'en:{rw}_{self.Game_language_code}:{ans[idx]}_UserAnswer:{ua}'
                    if record_string not in self.user_error_answer_history['string']:
                        self.user_error_answer_history['string'].append(record_string)
                        self.user_error_answer_history['image'].append(yolo_image_array)


            self.clock.tick(self.FPS)
            self.pg.display.update()  
        return None

    def YuGiOh_maker(self, title_text, language='en'):
        create_image = Image.open(self.Default_YuGiOh_Card_path)
        title_font = ImageFont.truetype(self.Default_YuGiOh_Card_font, 
                                        self.Default_YuGiOh_Card_font_size)
        image_editable = ImageDraw.Draw(create_image)
        #width, hight
        image_editable.text((self.Default_YuGiOh_Card_font_width,self.Default_YuGiOh_Card_font_height), 
                             title_text, self.BLACK, font=title_font)

        save_path = f"YuGiOh_Card/{language}_card/{title_text}.jpg"
        create_image.save(save_path)
        create_image_resize = self.pg.transform.scale(self.pg.image.load(save_path).convert(), 
                                                     (self.card_width, self.card_height))
        self.Card_Image.append(create_image_resize)
        return None

    def card(self, recognize_word):
        for default_card_path in ['en_card', f'{self.Game_language_code}_card']:
            if not os.path.exists(f'YuGiOh_Card/{default_card_path}'):
                os.makedirs(f'YuGiOh_Card/{default_card_path}')
        self.check_card_flag = False
        self.check_card_timing = 10
        self.card_status = dict()
        self.click_card_history = dict()
        self.card_score = 0
        self.Card_Image = [self.Card_Apple, self.Card_Banana, self.Card_Orange, 
                           self.Card_Lemon, self.Card_Papaya]
        # recogize_word in target language, and upper the first letter.
        ans = [self.translator.translate(i).capitalize() for i in recognize_word]
        _ = [self.YuGiOh_maker(card_title_text, language='en') for card_title_text in recognize_word]
        _ = [self.YuGiOh_maker(card_title_text, language=self.Game_language_code) for card_title_text in ans]
        mainBoard = self.card_random_board(recognize_word, ans)

        while True:
            self.screen.fill(self.D_PURPLE)
            MOUSE_POS = self.pg.mouse.get_pos()
            play_back = Button(image=None, pos=(self.WIDTH-200, self.HEIGHT-60), text_input="BACK",
                            font=self.get_font(75), base_color=self.WHITE, hover_color=self.GREEN)

            assert len(self.click_card_history) <= 2

            if len(self.click_card_history) == 2:
                self.card_evaluation()
            
            if (not self.check_card_flag) and (self.check_card_timing > 0): # start button not click
                play_start = Button(image=None, pos=(self.WIDTH-300, self.HEIGHT-500), text_input="START",
                                    font=self.get_font(75), base_color=self.WHITE, hover_color=self.GREEN)  
                button_list = [play_back,play_start]
                self.card_plot_board(mainBoard, plot_type = 'plot_card_back') # plot card back
            else: # start button was clicked
                button_list = [play_back]
                if self.check_card_timing > 0:
                    self.card_plot_board(mainBoard, plot_type = 'plot_card') # plot card image
                    self.check_card_timing -= 1
                    if self.check_card_timing == 0:
                        self.card_plot_board(mainBoard, plot_type = 'plot_card_back')
                    self.FPS = 1
                elif self.check_card_timing == 0:
                    self.card_plot_board(mainBoard, plot_type = 'plot_game') # plot card
                    self.FPS = 60

            for play_variable in button_list:
                play_variable.changeColor(MOUSE_POS)
                play_variable.update(self.screen)
            if self.check_card_timing > 0:
                self.draw_text_center(self.user_font(50), f'Remain check time : {self.check_card_timing}', 
                                  self.S_PINK, self.WIDTH-300, self.HEIGHT-400)
            else:
                self.draw_text_center(self.user_font(50), f'Score : {self.card_score}', 
                                  self.S_PINK, self.WIDTH-300, self.HEIGHT-300)
            for event in self.pg.event.get():
                if event.type == self.pg.QUIT:
                    self.pg.quit()
                    sys.exit()
                if event.type == self.pg.MOUSEBUTTONDOWN:
                    if play_back.checkForInput(MOUSE_POS):
                        self.games()
                    if play_start.checkForInput(MOUSE_POS):
                        self.check_card_flag = True
                if (event.type == self.pg.MOUSEBUTTONUP) and (self.check_card_timing == 0):
                    self.card_click_card()

            self.pg.display.flip()
            self.clock.tick(self.FPS)
            self.pg.display.update()
        return None

    def card_random_board(self, recognize_word, answer_word):
        recognize_word = [f'card_en_{i}' for i in recognize_word]
        answer_word = [f'card_{self.Game_language_code}_{i}' for i in answer_word]
        self.real_card_list = recognize_word + answer_word

        english_word = recognize_word + self.fake_card_list
        target_word = answer_word + self.fake_card_list

        board = []
        for idx, (en_w, tar_w) in enumerate(zip(english_word, target_word)):
            board.append((en_w, idx))
            board.append((tar_w, idx))

        random.shuffle(board)
        self.card_pixel_list = [[wp,hp] for wp in self.width_pixel_list 
                                        for hp in self.height_pixel_list]

        return board

    def card_plot_board(self, board, plot_type = 'plot_card_back'):
        for card_type, card_position in zip(board, self.card_pixel_list):
            # card_type[0] = name
            # card_type[1] = label number
            if plot_type == 'plot_card_back':
                self.screen.blit(self.Card_Back, (card_position[0], 
                                                  card_position[1]))
                # 0 is mean this card is close
                self.card_status[f'{card_type[0]}_{card_position[0]}_{card_position[1]}'] = 0
            elif plot_type == 'plot_card':
                for img, card_name in zip(self.Card_Image, self.fake_card_list + self.real_card_list):
                    if card_type[0] == card_name:
                        self.screen.blit(img, (card_position[0], 
                                               card_position[1]))
                        # 1 is mean this card is open
                        self.card_status[f'{card_type[0]}_{card_position[0]}_{card_position[1]}'] = 1
            elif plot_type == 'plot_game':
                if self.card_status[f'{card_type[0]}_{card_position[0]}_{card_position[1]}'] == 0:
                    self.screen.blit(self.Card_Back, (card_position[0], card_position[1]))
                elif self.card_status[f'{card_type[0]}_{card_position[0]}_{card_position[1]}'] == 1:
                    self.click_card_history[f'{card_type[0]}_{card_position[0]}_{card_position[1]}'] = card_type[1]
                    for img, card_name in zip(self.Card_Image, self.fake_card_list + self.real_card_list):
                        if card_type[0] == card_name:
                            self.screen.blit(img, (card_position[0], card_position[1]))
                elif self.card_status[f'{card_type[0]}_{card_position[0]}_{card_position[1]}'] == 2:
                    for img, card_name in zip(self.Card_Image, self.fake_card_list + self.real_card_list):
                        if card_type[0] == card_name:
                            self.screen.blit(img, (card_position[0], card_position[1]))
            else:
                break
        return None

    def card_click_card(self):
        self.click_card_number = np.sum([value for key,value in self.card_status.items()])
        mouse_position = self.pg.mouse.get_pos()
        mouse_y = mouse_position[0]
        mouse_x = mouse_position[1]
        for key in self.card_status.keys():
            split_key = key.split('_')
            key_length_list = [4,5]
            for kll in key_length_list:
                if len(split_key) == kll:
                    width = int(split_key[kll-2])
                    height = int(split_key[kll-1])
                    if (mouse_x >= height) and (mouse_x <= height+self.card_height) and \
                        (mouse_y >= width) and (mouse_y <= width+self.card_width):
                        
                        if (self.card_status[key] == 0):
                            self.card_status[key] = 1
        return None

    def card_evaluation(self):
        # two card are same
        if len(set([value for key,value in self.click_card_history.items()])) == 1:
            self.card_score += 1
            for key in self.click_card_history.keys():
                self.card_status[key] = 2
            self.click_card_history = {}
        # two card not same
        else:
            self.pg.time.delay(self.delay_time)
            for key in self.click_card_history.keys():
                self.card_status[key] = 0
            self.click_card_history = dict()

        
        return None

    def history_museum(self):
        assert len(self.user_error_answer_history['string']) == \
        len(self.user_error_answer_history['image']), \
        'the length of string and image not match'
        self.play_music(museum = True)
        self.player = Player()
        # according to the length of user_error_answer_history to create number of room.
        self.player.total_room = math.ceil(len(self.user_error_answer_history['string']) / 4)
        # to avoid 1/0 and increase room number.
        if self.player.total_room == 0:
            self.player.total_room = 1

        en_word = [i.split('_')[0].split(':')[1] for i in self.user_error_answer_history['string']]
        correct_word = [i.split('_')[1].split(':')[1] for i in self.user_error_answer_history['string']]
        target_language = [i.split('_')[1].split(':')[0] for i in self.user_error_answer_history['string']]
        user_word = [i.split('_')[2].split(':')[1] for i in self.user_error_answer_history['string']]

        similarity_word = []
        for i in en_word:
            temp = []
            try:
                three_words = self.w2v_model.most_similar(i, topn=3)
                for j in three_words:temp.append(j[0])
            except:
                for j in range(3):temp.append('')
            similarity_word.append(temp)

        while True:
            # get mouse coordinate
            MOUSE_POS = self.pg.mouse.get_pos()
            # build background image
            self.screen.blit(self.background_museum_img, [0, 0])
            # draw text of room number and keyboard a and d for walk
            self.draw_text_center(self.user_font(self.user_font_size),
                                  f'Room {self.player.room} / {self.player.total_room}', 
                                  self.S_PINK, self.WIDTH//2, self.HEIGHT//5 -100)
            self.draw_text_center(self.user_font(self.user_font_size),
                                  f'press a/d to go left/right', 
                                  self.S_PINK, (self.WIDTH//2), self.HEIGHT//5 -80) 

            # YOLO museum image index, [0,1,2,3] or [4,5,6,7].. and so on.
            room_yolo_image_index_list = list((np.arange(1,5) + (4 * (self.player.room-1)))-1)
            # get image array from error_history.
            yolo_image_array_list = [self.user_error_answer_history['image'][i] 
                                        for i in room_yolo_image_index_list 
                                        if i < len(self.user_error_answer_history['image'])]
            yolo_en_word_list = [en_word[i] 
                                 for i in room_yolo_image_index_list 
                                 if i < len(self.user_error_answer_history['string'])]

            yolo_correct_word_list = [correct_word[i] 
                                     for i in room_yolo_image_index_list 
                                     if i < len(self.user_error_answer_history['string'])]
            yolo_target_language_list = [target_language[i] 
                                         for i in room_yolo_image_index_list 
                                         if i < len(self.user_error_answer_history['string'])]

            yolo_similarity_word_list = [similarity_word[i] 
                                         for i in room_yolo_image_index_list 
                                         if i < len(self.user_error_answer_history['string'])]

            # one line mapping de to Deutsch and fr to French which within in the list.
            yolo_target_language_list = [{'de': 'Deutsch', 'fr': 'French'}.get(i, 'none') 
                                         for i in yolo_target_language_list]

            yolo_user_word_list = [user_word[i] 
                                 for i in room_yolo_image_index_list 
                                 if i < len(self.user_error_answer_history['string'])]


            #transfer numpy array to surface                
            yolo_surf_list = [self.pg.surfarray.make_surface(i) for i in yolo_image_array_list]
            # scale image
            yolo_surf_list = [self.pg.transform.scale(i, (165,165)) for i in yolo_surf_list]
            # image counterclockwise 270 degree
            yolo_surf_list = [self.pg.transform.rotate(i, 270) for i in yolo_surf_list]


            # build frame
            for meseum_width in self.yolo_museum_width_interval:
                self.screen.blit(self.museum_frame, [meseum_width, self.yolo_museum_frame_height])
            # build exhibition, that exhibition to match the frame.
            for meseum_width in self.yolo_museum_width_interval:
                self.screen.blit(self.meseum_exhibition, [meseum_width, self.meseum_exhibition_height])
            # offset yolo image 43 pixel to match the frame
            for museum_width, museum_img, museum_en_text, museum_language, museum_correct, us_ans, three_word \
                in zip(self.yolo_museum_width_interval, yolo_surf_list, yolo_en_word_list, \
                yolo_target_language_list, yolo_correct_word_list, yolo_user_word_list,yolo_similarity_word_list):

                self.screen.blit(museum_img, [museum_width+43, self.yolo_museum_frame_height+43])
                # draw text on the exhibition
                self.draw_text_center(self.user_font(self.exhibition_font_size),
                                    f'English:{museum_en_text}', self.BLACK, 
                                    museum_width+120, self.yolo_museum_frame_height+260)
                self.draw_text_center(self.user_font(self.exhibition_font_size),
                                    f'{museum_language}:{museum_correct}', 
                                    self.BLACK, museum_width+120, 
                                    self.yolo_museum_frame_height+275)
                self.draw_text_center(self.user_font(self.exhibition_font_size),
                                    f'User:{us_ans}', self.BLACK, museum_width+120, 
                                    self.yolo_museum_frame_height+290)
                self.draw_text_center(self.user_font(self.exhibition_font_size),
                                    f'-'*40, self.BLACK, museum_width+125, 
                                    self.yolo_museum_frame_height+300)

                for each_word,offset in zip(three_word, [320,335,350]):
                    self.draw_text_center(self.user_font(self.exhibition_font_size),
                                        each_word, self.BLACK, museum_width+125, 
                                        self.yolo_museum_frame_height+offset)

            # event of player at fat left or fat right and room number
            # self.WIDTH is define at config, value is 1000
            if self.player.rect.left < 0:
                # not at room 1 and want to go back to previous room
                if self.player.room != 1:
                    self.player.room -= 1
                    self.player.rect.right = (self.WIDTH - 1) # if 1000, can not change room.
                    self.pg.mixer.Channel(1).play(self.pg.mixer.Sound(self.bg_museum_scene_music))
                # at room 1 and want to go back to wall
                elif self.player.room == 1:
                    self.player.rect.left = 0
            elif self.player.rect.right > self.WIDTH:
                if self.player.room != (self.player.total_room):
                    self.player.room +=1
                    self.player.rect.left = 0
                    self.pg.mixer.Channel(1).play(self.pg.mixer.Sound(self.bg_museum_scene_music))
                elif self.player.room == self.player.total_room:
                    self.player.rect.right = self.WIDTH

            # build player image
            self.screen.blit(self.player.image, self.player.rect)

            # button for back to menu
            play_back = Button(image=None, pos=(self.WIDTH-50, 20), text_input="BACK",
                               font=self.get_font(25), base_color=self.WHITE, 
                               hover_color=self.GREEN)
            play_back.changeColor(MOUSE_POS)
            play_back.update(self.screen)

            # event of press key
            for event in self.pg.event.get():
                if event.type == self.pg.QUIT:
                    self.pg.quit()
                    sys.exit()
                elif event.type == self.pg.MOUSEBUTTONDOWN:
                    if play_back.checkForInput(MOUSE_POS):
                        self.init()
                elif event.type == self.pg.KEYDOWN:
                    if event.key == self.pg.K_a:
                        self.player.direction = 'left'
                        self.player.speed = - self.pony_speed
                    elif event.key == self.pg.K_d:
                        self.player.direction = 'right'
                        self.player.speed = self.pony_speed
                elif event.type == self.pg.KEYUP:
                    self.player.speed = 0

            # update screen
            self.pg.display.flip()
            self.clock.tick(self.FPS)
            self.player.update()
            self.pg.display.update()

        return None
