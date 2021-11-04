import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pymysql
import pytz
import time
from datetime import datetime, timedelta, timezone
import cv2
from libs.hoheto.ml.table import CcxtCandles
from PIL import Image, ImageDraw, ImageFont
from perlin_noise import PerlinNoise
import matplotlib.pyplot as plt


class StrategiesImageGenerator(object):

    def __init__(self, directory, candle_generator):
        self.directory = directory
        self.candle_generator = candle_generator
        self.width = 2000
        self.height = 2000
        self.x_center = int(self.width / 2)
        self.y_center = int(self.height / 2)
        self.size = (self.width, self.height, 3)
        self.square = np.array([[0, 0], [0, self.height], [self.width, self.height], [self.width, 0]])
        self.color = {
            "white": (255, 255, 255),
            "gray": (128, 128, 128),
        }
        self.resource_dir = self.directory + "resources/"
        self.image_dir = self.resource_dir + "images/"
        self.csv_dir = self.resource_dir + "csv/"
        self.logic_dir = self.resource_dir + "logics/"
        self.font_dir = self.resource_dir + "fonts/"
        self.product_dir = self.directory + "products/"
        self.csv = []
        self.daily_candles = None
        self.hourly_candles = None
        self.bar_tick = None
        self.bar_width = None
        self.bar_padding = None
        self.bar_start = None
        self.price_range = None
        self.price_base = None
        # Gradation.
        self.gradation_radius = 800
        self.gradation_thickness = -1
        self.gradation_blur = (2051, 2051)
        # Grids.
        self.grids_thickness = 1
        self.grids_blur = (5, 5)
        self.y_grids_num = 4
        # PnL.
        self.pnl_thickness = 5
        self.pnl_glow_thickness = 20
        self.pnl_glow_blur = (51, 51)
        # Text.
        self.text_font = self.font_dir + "Magnolia Script.otf"
        self.text_size = 45
        # csv data.
        self.csv = None

    def _load_ftx_candles(self):
        self.hourly_candles = self.candle_generator.generate()
        self.daily_candles = pd.concat([
            self.hourly_candles["o"].resample("24h").first(),
            self.hourly_candles["h"].resample("24h").max(),
            self.hourly_candles["l"].resample("24h").min(),
            self.hourly_candles["c"].resample("24h").last(),
            self.hourly_candles["v"].resample("24h").sum(),
        ], axis=1)
        self.bar_tick = int(self.width / len(self.daily_candles))  # キャンドル1つ分のtick.
        self.bar_width = math.ceil(self.bar_tick / 2)  # 偶数になるようにceil
        self.bar_padding = self.bar_tick - self.bar_width
        self.bar_start = int((self.width - self.bar_tick * len(self.daily_candles) + self.bar_padding) / 2)
        self.price_range = self.daily_candles["h"].max() - self.daily_candles["l"].min()
        self.price_base = int((self.daily_candles["h"].max() + self.daily_candles["l"].min()) / 2)

    def prepare(self):
        self._load_csv()
        self._load_images()
        self.noise_image = cv2.imread(self.image_dir + "noise.png")
        self.perlin_noise_bg = cv2.imread(self.image_dir + "perlin_noise_bg.png")
        self.perlin_noise_fg = cv2.imread(self.image_dir + "perlin_noise_fg.png")
        self.gradation_image = cv2.imread(self.image_dir + "gradation.png")
        self.volume_image = cv2.imread(self.image_dir + "volume.png")
        self.candle_image = cv2.imread(self.image_dir + "candles.png")
        self.grid_image = cv2.imread(self.image_dir + "grids.png")

    def _write_image(self, filename, image):
        cv2.imwrite(self.image_dir + filename, image)

    def _save_product(self, filename, image):
        cv2.imwrite(self.product_dir + filename, image)

    def _exists_image(self, filename):
        return os.path.isfile(self.image_dir + filename)

    def _create_empty_image(self):
        return np.zeros(self.size, dtype=np.uint8)

    def _load_images(self):
        if not self._exists_image("volume.png") or not self._exists_image("candles.png"):
            self._load_ftx_candles()
        if not self._exists_image("grids.png"):
            self._create_grid_image("grids.png")

    def _create_gradation_image(self, filename):
        tmp_image = self._create_empty_image()
        cv2.circle(tmp_image, (self.x_center, self.y_center), self.gradation_radius, self.color["white"], thickness=self.gradation_thickness)
        tmp_image = cv2.GaussianBlur(tmp_image, self.gradation_blur, 0)
        self._write_image(filename, tmp_image)

    def _create_volume_image(self, filename):
        max_height = int(self.height * 0.5)
        bars = []
        base_y = self.height
        for i in range(len(self.daily_candles)):
            base_x = i * self.bar_tick + self.bar_start
            x1 = base_x
            y1 = base_y
            x2 = base_x + self.bar_width
            y2 = int(base_y - self.daily_candles["volume"].iloc[i] / self.daily_candles["volume"].max() * max_height)
            points = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])
            bars.append(points)
        tmp_image = self._create_empty_image()
        cv2.fillPoly(tmp_image, pts=bars, color=self.color["white"])
        self._write_image(filename, tmp_image)

    def _create_candle_image(self, filename):
        base_y = int(self.height / 2)
        max_height = int(self.height * (2 / 3))
        hige_up = []
        hige_down = []
        box_up = []
        box_down = []
        for i in range(len(self.daily_candles)):
            base_x = i * self.bar_tick + self.bar_start
            # ヒゲ.
            x1 = base_x + int(self.bar_width / 2)
            y1 = int(base_y - (self.daily_candles["low_price"].iloc[i] - self.price_base) / self.price_range * max_height)
            x2 = base_x + int(self.bar_width / 2)
            y2 = int(base_y - (self.daily_candles["high_price"].iloc[i] - self.price_base) / self.price_range * max_height)
            points = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])
            if self.daily_candles["open_price"].iloc[i] <= self.daily_candles["close_price"].iloc[i]:
                hige_up.append(points)
            else:
                hige_down.append(points)
            # 箱.
            x1 = base_x
            y1 = int(base_y - (self.daily_candles["open_price"].iloc[i] - self.price_base) / self.price_range * max_height)
            x2 = base_x + self.bar_width
            y2 = int(base_y - (self.daily_candles["close_price"].iloc[i] - self.price_base) / self.price_range * max_height)
            points = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])
            if self.daily_candles["open_price"].iloc[i] <= self.daily_candles["close_price"].iloc[i]:
                box_up.append(points)
            else:
                box_down.append(points)
        tmp_image = self._create_empty_image()
        cv2.fillPoly(tmp_image, pts=hige_up, color=self.color["white"])
        cv2.fillPoly(tmp_image, pts=hige_down, color=self.color["gray"])
        cv2.fillPoly(tmp_image, pts=box_up, color=self.color["white"])
        cv2.fillPoly(tmp_image, pts=box_down, color=self.color["gray"])
        self._write_image(filename, tmp_image)

    def _create_grid_image(self, filename):
        # 縦軸.
        day_padding = int(len(self.daily_candles) / self.y_grids_num)
        y_grids_x = [self.bar_start + day_padding * i * self.bar_tick + int(self.bar_width / 2) for i in
                     range(self.y_grids_num)]
        # 横軸.
        base_y = int(self.height / 2)
        max_height = int(self.height * (2 / 3))
        price_padding = 10000  # 10000ドル.
        x_grids_y = []
        i = 0
        while True:
            price = price_padding * i
            y = int(base_y - (price - self.price_base) / self.price_range * max_height)
            if y < 0:
                break
            x_grids_y.append(y)
            i += 1
        print(y_grids_x)
        print(x_grids_y)
        points_array = []
        for x in y_grids_x:
            points_array.append(np.array([[x, 0], [x, self.height]]))
        for y in x_grids_y:
            points_array.append(np.array([[0, y], [self.width, y]]))

        tmp_image = self._create_empty_image()
        cv2.polylines(tmp_image, points_array, False, self.color["white"], thickness=self.grids_thickness)
        tmp_image = cv2.GaussianBlur(tmp_image, self.grids_blur, 0)
        self._write_image(filename, tmp_image)

    #     def _create_noise_image(self, i):
    #         filename = f"noise_{i}.png"
    #         tmp_image = self._create_empty_image()
    #         tmp_image = self._add_noise(tmp_image, mean=0, sd=50, low=50, high=100)
    #         self._write_image(filename, tmp_image)

    def _create_nebula_image(self, i):
        last_balance = self.csv[i]["balance"][-1]
        np.random.seed(seed=int(1000 * (last_balance + 10)))
        x, y, rotete, hue = np.random.rand(4)
        x = int(x * 8000)
        y = int(y * 8000)
        rotete = int(rotete * 360)
        hue = int(hue * 255) - 128

        perlin_noise_bg = self.perlin_noise_bg[x: x + self.width, y: y + self.height]
        result = self._hard_light(self.noise_image, perlin_noise_bg)
        result = self._soft_light(result, self.perlin_noise_fg)
        result = self._soft_light(result, self.perlin_noise_fg)
        result_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

        result_trans = []
        for x in range(len(result_hsv)):
            row = []
            for y in range(len(result_hsv[x])):
                hue_trans = result_hsv[x][y][0] + hue
                hue_trans = np.clip(hue_trans, 0, 255)
                hsv = np.array([hue_trans, result_hsv[x][y][1], result_hsv[x][y][2]], dtype=np.uint8)
                row.append(hsv)
            result_trans.append(np.array(row))
        result_trans = np.array(result_trans)
        result_trans = cv2.cvtColor(result_trans, cv2.COLOR_HSV2BGR)
        filename = f"nebula_{i}.png"
        self._write_image(filename, result_trans)

    def _create_pnl_image(self, i):
        balance_range = 1.5  # -1.5～1.5 までを0～2000に描画.
        balances = self.csv[i]["balance"].resample("4h").mean()

        # HUE 90～180 blue to red
        hue = np.clip(int(-balances[-1] * 25 + 145), 110, 180)
        result = []
        for x in range(10):
            row = []
            for y in range(10):
                hsv = np.array([hue, 255, 200], dtype=np.uint8)
                row.append(hsv)
            result.append(np.array(row))
        result = np.array(result)
        result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
        glow_color = [int(item) for item in result[0][0]]

        balances = balances - balances[-1] * 0.5
        points = []
        base_y = int(self.height / 2)
        x_tick = self.width / (len(balances) - 1)
        for index, balance in enumerate(balances):
            x = int(x_tick * index)
            y = (balance + balance_range) / (balance_range * 2) * self.height - base_y
            y = int(base_y - y)
            points.append([x, y])
        # PnL画像.
        tmp_image = self._create_empty_image()
        cv2.polylines(tmp_image, np.array([points]), False, self.color["white"], thickness=self.pnl_thickness)
        filename = f"pnl_{i}.png"
        self._write_image(filename, tmp_image)
        # PnL画像 光彩.
        tmp_image = self._create_empty_image()
        cv2.polylines(tmp_image, np.array([points]), False, glow_color, self.pnl_glow_thickness)
        tmp_image = cv2.GaussianBlur(tmp_image, self.pnl_glow_blur, 0)
        filename = f"pnl_{i}_glow.png"
        self._write_image(filename, tmp_image)

    def generate_all_images(self):
        for i in range(100):
            print("processing", i)
            self.generate_image(i)

    def generate_image(self, i):
        # PnL作成.
        self._create_pnl_image(i)
        pnl_image = cv2.imread(f"{self.image_dir}pnl_{i}.png")
        pnl_glow_image = cv2.imread(f"{self.image_dir}pnl_{i}_glow.png")
        # テキスト画像作成.
        self._create_text_image(i)
        text_image = cv2.imread(f"{self.image_dir}text_{i}.png")
        # Nebula画像作成.
        if not self._exists_image(f"nebula_{i}.png"):
            self._create_nebula_image(i)
        nebula_image = cv2.imread(f"{self.image_dir}nebula_{i}.png")
        # 合成.
        result = self._screen(nebula_image, self.grid_image * 0.5)
        result = self._screen(result, self.volume_image * 0.4)
        result = self._screen(result, self.candle_image * 0.8)
        result = self._multiply(result, self.gradation_image)
        result = self._screen(result, pnl_glow_image)
        result = self._screen(result, pnl_image)
        result = self._screen(result, text_image)
        filename = f"strategies_{i:0=3}.png"
        self._save_product(filename, result)

    def _add_noise(self, image, mean, sd, low, high):
        # ノイズ画像作成.
        noise_img = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.randn(noise_img, mean, sd)
        # ノイズ適用.
        image[high < noise_img] = 255
        image[noise_img < low] = 0
        return image

    def _load_csv(self):
        self.strategies = pd.read_csv(self.csv_dir + "strategies.csv")
        self.csv = []
        for i in range(100):
            filename = f"{i:03}.csv"
            data = pd.read_csv(self.csv_dir + f"{i}.csv")
            data.index = pd.to_datetime(data["ts"], unit="s", utc=True)
            self.csv.append(data)

    def _multiply(self, bg_img, fg_img):
        # ref. https://optie.hatenablog.com/entry/2018/03/15/212107
        bg_img = bg_img / 255
        fg_img = fg_img / 255
        result = (bg_img * fg_img).clip(0, 1)
        return np.array(result * 255, dtype=np.uint8)

    def _screen(self, bg_img, fg_img):
        bg_img = bg_img / 255
        fg_img = fg_img / 255
        result = 1 - ((1 - bg_img) * (1 - fg_img))
        return np.array(result * 255, dtype=np.uint8)

    def _exclusion(self, bg_img, fg_img):
        bg_img = bg_img / 255
        fg_img = fg_img / 255
        result = np.zeros(bg_img.shape)
        result = bg_img + fg_img - 2 * bg_img * fg_img
        return np.array(result * 255, dtype=np.uint8)

    def _dodge(self, bg_img, fg_img):
        bg_img = bg_img / 255
        fg_img = fg_img / 255
        fg_reverse = 1 - fg_img
        non_zero = fg_reverse != 0
        result = np.zeros(bg_img.shape)
        result[non_zero] = bg_img[non_zero] / fg_reverse[non_zero]
        result[~non_zero] = 1
        return np.array(result * 255, dtype=np.uint8)

    def _overlay(self, bg_img, fg_img):
        bg_img = bg_img / 255
        fg_img = fg_img / 255
        darker = bg_img < 0.5
        bg_inverse = 1 - bg_img
        fg_inverse = 1 - fg_img
        result = np.zeros(bg_img.shape)
        result[darker] = bg_img[darker] * fg_img[darker] * 2
        result[~darker] = 1 - bg_inverse[~darker] * fg_inverse[~darker] * 2
        return np.array(result * 255, dtype=np.uint8)

    def _hard_light(self, bg_img, fg_img):
        bg_img = bg_img / 255
        fg_img = fg_img / 255
        darker = fg_img < 0.5
        result = np.zeros(bg_img.shape)
        bg_inverse = 1 - bg_img
        fg_inverse = 1 - fg_img
        result[darker] = 2 * bg_img[darker] * fg_img[darker]
        result[~darker] = 1 - 2 * bg_inverse[~darker] * fg_inverse[~darker]
        return np.array(result * 255, dtype=np.uint8)

    def _soft_light(self, bg_img, fg_img):
        bg_img = bg_img / 255
        fg_img = fg_img / 255
        darker = fg_img < 0.5
        result = np.zeros(bg_img.shape)
        result[darker] = 2 * fg_img[darker] * bg_img[darker] + 2 * (0.5 - fg_img[darker]) * np.square(bg_img[darker])
        result[~darker] = 2 * (1 - fg_img[~darker]) * bg_img[~darker] + 2 * (fg_img[~darker] - 0.5) * np.sqrt(bg_img[~darker])
        return np.array(result * 255, dtype=np.uint8)

    def _put_text(self, img, text, org, font_face, font_size, color):
        # ref. https://qiita.com/mo256man/items/b6e17b5a66d1ea13b5e3
        font = ImageFont.truetype(font=font_face, size=font_size)
        dummy_draw = ImageDraw.Draw(Image.new("RGB", (0, 0)))
        text_w, text_h = dummy_draw.textsize(text, font=font)
        text_b = int(0.1 * text_h)
        x, y = org
        offset_x = text_w // 2
        offset_y = (text_h + text_b) // 2
        x0 = x - offset_x
        y0 = y - offset_y
        img_h, img_w = img.shape[:2]
        x1, y1 = max(x0, 0), max(y0, 0)
        x2, y2 = min(x0 + text_w, img_w), min(y0 + text_h + text_b, img_h)
        text_area = np.full((text_h + text_b, text_w, 3), (0, 0, 0), dtype=np.uint8)
        text_area[y1 - y0:y2 - y0, x1 - x0:x2 - x0] = img[y1:y2, x1:x2]
        img_pillow = Image.fromarray(text_area)
        draw = ImageDraw.Draw(img_pillow)
        draw.text(xy=(0, 0), text=text, fill=color, font=font)
        text_area = np.array(img_pillow, dtype=np.uint8)
        img[y1:y2, x1:x2] = text_area[y1 - y0:y2 - y0, x1 - x0:x2 - x0]
        return img

    def _create_text_image(self, i):
        filename = f"text_{i}.png"
        tmp_image = self._create_empty_image()
        text = f"#{i:0=3}. " + self.strategies["name"].iloc[i]
        tmp_image = self._put_text(tmp_image, text, (int(self.width / 2), self.height - 100), self.text_font, self.text_size, self.color["white"])
        self._write_image(filename, tmp_image)

